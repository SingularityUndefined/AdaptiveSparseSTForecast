"""Basic modules for unrolled optimization models."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOperator = Callable[..., torch.Tensor]


def _identity(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return x


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _validate_undirected_edge_pairs(
    neighbor_list: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> None:
    num_graphs = int(neighbor_mask.size(0))
    for graph in range(num_graphs):
        directed_edges = set()
        undirected_counts = {}
        graph_mask = neighbor_mask[graph]
        rows, cols = graph_mask.nonzero(as_tuple=True)

        for row, col in zip(rows.tolist(), cols.tolist()):
            neighbor = int(neighbor_list[row, col])
            if neighbor == row:
                raise ValueError(
                    f"graph {graph} contains a self-loop at node {row}."
                )

            directed_edge = (row, neighbor)
            if directed_edge in directed_edges:
                raise ValueError(
                    f"graph {graph} contains duplicate directed edge "
                    f"{row}->{neighbor}."
                )
            directed_edges.add(directed_edge)

            undirected_edge = (min(row, neighbor), max(row, neighbor))
            undirected_counts[undirected_edge] = (
                undirected_counts.get(undirected_edge, 0) + 1
            )

        for u, v in undirected_counts:
            count = undirected_counts[(u, v)]
            has_forward = (u, v) in directed_edges
            has_backward = (v, u) in directed_edges
            if count != 2 or not has_forward or not has_backward:
                raise ValueError(
                    "Each valid undirected edge must appear exactly twice, "
                    f"once per direction. In graph {graph}, edge ({u}, {v}) "
                    f"appears {count} time(s)."
                )


def _validate_operator_neighbor_inputs(
    neighbor_list: torch.Tensor,
    neighbor_mask: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neighbor_list = torch.as_tensor(neighbor_list, dtype=torch.long)
    neighbor_mask = torch.as_tensor(neighbor_mask, dtype=torch.bool)
    if neighbor_list.dim() != 2 or neighbor_list.size(0) != num_nodes:
        raise ValueError("neighbor_list must have shape (num_nodes, num_neighbors).")
    if neighbor_mask.dim() != 3 or tuple(neighbor_mask.shape[1:]) != tuple(
        neighbor_list.shape
    ):
        raise ValueError(
            "neighbor_mask must have shape (num_graphs, num_nodes, num_neighbors)."
        )

    if bool((neighbor_list < -1).any()):
        raise ValueError("neighbor_list entries must be -1 or valid node indices.")
    if bool((neighbor_list >= num_nodes).any()):
        bad_index = int(neighbor_list[neighbor_list >= num_nodes][0])
        raise ValueError(
            f"neighbor_list contains node index {bad_index}, but num_nodes={num_nodes}."
        )
    active_mask = neighbor_mask.any(dim=0)
    if bool((active_mask & (neighbor_list < 0)).any()):
        raise ValueError("neighbor_mask=True entries must have valid node indices.")
    _validate_undirected_edge_pairs(neighbor_list, neighbor_mask)

    safe_neighbor_list = neighbor_list.masked_fill(~active_mask, 0)
    return neighbor_list, neighbor_mask, safe_neighbor_list


class UnrolledCG(nn.Module):
    """Learned, fixed-depth preconditioned conjugate-gradient update.

    ``B_vecs`` must have shape ``(batch_size, num_heads, num_nodes)``.
    ``A_func`` and ``Minv_func``/``M_func`` must preserve this shape.  Extra
    operator arguments can be passed through ``A_args``/``A_kwargs`` and
    ``Minv_args``/``Minv_kwargs`` or ``M_args``/``M_kwargs``.
    """

    def __init__(
        self,
        CG_iters: int,
        alpha_init: float,
        beta_init: float,
        num_heads: int,
        init_method: str = "constant",
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if CG_iters <= 0:
            raise ValueError("CG_iters must be a positive integer.")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")
        if init_scale < 0.0:
            raise ValueError("init_scale must be non-negative.")

        self.num_iters = int(CG_iters)
        self.CG_iters = self.num_iters  # Backward-compatible attribute name.
        self.num_heads = int(num_heads)
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.init_method = init_method
        self.init_scale = float(init_scale)

        parameter_shape = (self.num_iters, self.num_heads)
        self.alpha = nn.Parameter(torch.empty(parameter_shape))
        self.beta = nn.Parameter(torch.empty(parameter_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_method == "constant":
            nn.init.constant_(self.alpha, self.alpha_init)
            nn.init.constant_(self.beta, self.beta_init)
        elif self.init_method == "normal":
            nn.init.normal_(self.alpha, mean=self.alpha_init, std=self.init_scale)
            nn.init.normal_(self.beta, mean=self.beta_init, std=self.init_scale)
        elif self.init_method == "uniform":
            nn.init.uniform_(
                self.alpha,
                self.alpha_init - self.init_scale,
                self.alpha_init + self.init_scale,
            )
            nn.init.uniform_(
                self.beta,
                self.beta_init - self.init_scale,
                self.beta_init + self.init_scale,
            )
        elif self.init_method == "xavier_uniform":
            self._init_xavier_uniform_around(self.alpha, self.alpha_init)
            self._init_xavier_uniform_around(self.beta, self.beta_init)
        else:
            raise ValueError(
                "init_method must be one of: "
                "'constant', 'normal', 'uniform', 'xavier_uniform'."
            )

    def _init_xavier_uniform_around(
        self,
        parameter: nn.Parameter,
        center: float,
    ) -> None:
        init_view = parameter.unsqueeze(0) if parameter.dim() == 1 else parameter
        nn.init.xavier_uniform_(init_view, gain=self.init_scale)
        with torch.no_grad():
            parameter.add_(center)

    def _validate_signal_shape(self, signal: torch.Tensor, name: str) -> None:
        if signal.dim() != 3 or signal.size(1) != self.num_heads:
            raise ValueError(
                f"{name} must have shape (batch_size, num_heads, num_nodes) "
                f"with num_heads={self.num_heads}."
            )

    def _apply_operator(
        self,
        operator: TensorOperator,
        signal: torch.Tensor,
        operator_name: str,
        operator_args: Tuple[Any, ...] = (),
        operator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if operator_kwargs is None:
            operator_kwargs = {}
        output = operator(signal, *operator_args, **operator_kwargs)
        if output.shape != signal.shape:
            raise ValueError(
                f"{operator_name} must return a tensor with the same shape as input: "
                f"expected {tuple(signal.shape)}, got {tuple(output.shape)}."
            )
        return output

    def _step_parameter(
        self,
        parameter: torch.Tensor,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        return parameter.view(1, parameter.size(0), 1)

    def forward(
        self,
        A_func: TensorOperator,
        B_vecs: torch.Tensor,
        Minv_func: Optional[TensorOperator] = None,
        X0: Optional[torch.Tensor] = None,
        *,
        A_args: Tuple[Any, ...] = (),
        A_kwargs: Optional[Dict[str, Any]] = None,
        Minv_args: Tuple[Any, ...] = (),
        Minv_kwargs: Optional[Dict[str, Any]] = None,
        M_func: Optional[TensorOperator] = None,
        M_args: Optional[Tuple[Any, ...]] = None,
        M_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        self._validate_signal_shape(B_vecs, "B_vecs")
        if X0 is not None and X0.shape != B_vecs.shape:
            raise ValueError("X0 must have the same shape as B_vecs.")
        if Minv_func is not None and M_func is not None:
            raise ValueError("Pass only one of Minv_func or M_func.")
        if M_func is not None:
            Minv_func = M_func
            Minv_args = () if M_args is None else M_args
            Minv_kwargs = M_kwargs
        elif M_args is not None or M_kwargs is not None:
            raise ValueError("M_args and M_kwargs require M_func.")

        preconditioner = Minv_func if Minv_func is not None else _identity
        x = torch.zeros_like(B_vecs) if X0 is None else X0.clone()

        residual = B_vecs - self._apply_operator(
            A_func,
            x,
            "A_func",
            A_args,
            A_kwargs,
        )
        direction = self._apply_operator(
            preconditioner,
            residual,
            "Minv_func",
            Minv_args,
            Minv_kwargs,
        )

        for alpha, beta in zip(self.alpha, self.beta):
            alpha = self._step_parameter(alpha, direction)
            beta = self._step_parameter(beta, direction)
            A_direction = self._apply_operator(
                A_func,
                direction,
                "A_func",
                A_args,
                A_kwargs,
            )
            x = x + alpha * direction
            residual = residual - alpha * A_direction
            direction = (
                self._apply_operator(
                    preconditioner,
                    residual,
                    "Minv_func",
                    Minv_args,
                    Minv_kwargs,
                )
                + beta * direction
            )

        return x


class GraphLearningModule(nn.Module):
    """Generate sparse graph weights from node embeddings and observations."""

    def __init__(
        self,
        num_nodes: int,
        num_neighbors: int,
        neighbor_list: torch.Tensor,
        neighbor_mask: torch.Tensor,
        emb_dim: int = 6,
        feature_dim: int = 3,
        theta: float = 0.5,
        theta_min: float = 1e-6,
        embedding_std: float = 1.0,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive.")
        if emb_dim <= 0:
            raise ValueError("emb_dim must be positive.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if theta_min <= 0.0:
            raise ValueError("theta_min must be positive.")
        if theta <= theta_min:
            raise ValueError("theta must be greater than theta_min.")
        if embedding_std < 0.0:
            raise ValueError("embedding_std must be non-negative.")

        self.num_nodes = int(num_nodes)
        self.num_neighbors = int(num_neighbors)
        self.emb_dim = int(emb_dim)
        self.feature_dim = int(feature_dim)
        self.theta_min = float(theta_min)
        self.embedding_std = float(embedding_std)

        neighbor_list, neighbor_mask, safe_neighbor_list = (
            _validate_operator_neighbor_inputs(
                neighbor_list,
                neighbor_mask,
                self.num_nodes,
            )
        )
        if neighbor_list.size(1) != self.num_neighbors:
            raise ValueError("neighbor_list must have shape (num_nodes, num_neighbors).")
        self.num_graphs = int(neighbor_mask.size(0))

        self.register_buffer("neighbor_list", neighbor_list)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self.register_buffer("safe_neighbor_list", safe_neighbor_list)

        self.node_embeddings = nn.Parameter(
            torch.empty(self.num_graphs, self.num_nodes, self.emb_dim)
        )
        self.fc = nn.Linear(self.emb_dim + 1, self.feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        theta_init = torch.full(
            (self.num_graphs,),
            float(theta - self.theta_min),
        )
        raw_theta = _inverse_softplus(theta_init)
        self.raw_theta = nn.Parameter(raw_theta)
        self.reset_parameters()

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + self.theta_min

    def reset_parameters(self) -> None:
        nn.init.normal_(self.node_embeddings, mean=0.0, std=self.embedding_std)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate graph weights from node values.

        ``x`` must have shape ``(batch_size, num_graphs, num_nodes)``.
        """
        if x.dim() != 3 or x.size(1) != self.num_graphs or x.size(2) != self.num_nodes:
            raise ValueError(
                "x must have shape (batch_size, num_graphs, num_nodes)."
            )
        return self._forward_multi_graph(x)

    def _forward_multi_graph(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_graphs, _ = x.shape
        node_embeddings = self.node_embeddings.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        node_input = torch.cat([x.unsqueeze(-1), node_embeddings], dim=-1)
        features = self.leakyrelu(self.fc(node_input))
        features_by_graph = features.permute(1, 0, 2, 3).contiguous()
        flat_features = features_by_graph.reshape(
            num_graphs * batch_size,
            self.num_nodes,
            self.feature_dim,
        )
        neighbor_features = flat_features[
            :,
            self.safe_neighbor_list.reshape(-1),
        ].reshape(
            num_graphs,
            batch_size,
            self.num_nodes,
            self.num_neighbors,
            self.feature_dim,
        )
        edge_diff = features_by_graph.unsqueeze(3) - neighbor_features

        theta = self.theta.to(device=x.device, dtype=features.dtype).view(
            num_graphs,
            1,
            1,
            1,
        )
        w = torch.exp(-(edge_diff.square()).sum(dim=-1) / (2.0 * theta)).mean(dim=1)
        if not torch.isfinite(w).all():
            raise ValueError("Non-finite values detected in learned graph weights.")

        return w * self.neighbor_mask.to(device=w.device, dtype=w.dtype)

class SparseGraphOperators(nn.Module):
    """Sparse Laplacian operators for GEM neighbor-list graphs."""

    def __init__(
        self,
        num_nodes: int,
        neighbor_list: torch.Tensor,
        neighbor_mask: torch.Tensor,
        c: float = 8,
        scale: bool = True,
        epsilon: float = 4e-3,
        norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if c <= 0.0:
            raise ValueError("c must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive.")

        self.num_nodes = int(num_nodes)
        self.c = float(c)
        self.scale = bool(scale)
        self.epsilon = float(epsilon)
        self.norm_eps = float(norm_eps)

        neighbor_list, neighbor_mask, safe_neighbor_list = (
            _validate_operator_neighbor_inputs(
                neighbor_list,
                neighbor_mask,
                self.num_nodes,
            )
        )

        self.num_neighbors = int(neighbor_list.size(1))
        self.register_buffer("neighbor_list", neighbor_list)
        self.register_buffer("neighbor_mask", neighbor_mask)
        self.register_buffer("safe_neighbor_list", safe_neighbor_list)

    def _validate_weight(self, w: torch.Tensor) -> torch.Tensor:
        if w.dim() != 3 or tuple(w.shape[1:]) != tuple(self.neighbor_list.shape):
            raise ValueError(
                "w must have shape (num_graphs, num_nodes, num_neighbors)."
            )
        if w.size(0) != self.neighbor_mask.size(0):
            raise ValueError("w and neighbor_mask must agree on num_graphs.")
        mask = self.neighbor_mask.to(device=w.device, dtype=w.dtype)
        return w * mask

    def _validate_node_batch(self, X: torch.Tensor, w: torch.Tensor) -> None:
        if X.dim() != 3 or X.size(1) != w.size(0) or X.size(2) != self.num_nodes:
            raise ValueError("X must have shape (batch_size, num_graphs, num_nodes).")

    def scale_w(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale weights so the sparse Laplacian has Frobenius norm ``c``."""
        w = self._validate_weight(w)
        if not self.scale:
            return w, w.new_ones(w.size(0))

        degree = w.sum(dim=-1)
        laplacian_norm_square = degree.square().sum(dim=1) + w.square().sum(
            dim=(1, 2)
        )
        norm = torch.sqrt(torch.clamp(laplacian_norm_square, min=self.norm_eps))
        scale_factor = self.c / norm
        return w * scale_factor.view(-1, 1, 1), scale_factor

    def apply_L(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply sparse graph Laplacian(s) to batched node signals.

        Args:
            X: ``(batch_size, num_graphs, num_nodes)``.
            w: ``(num_graphs, num_nodes, num_neighbors)``.

        Returns:
            ``(batch_size, num_graphs, num_nodes)``.
        """
        w = self._validate_weight(w).to(device=X.device, dtype=X.dtype)
        self._validate_node_batch(X, w)
        neighbor_index = self.safe_neighbor_list.reshape(-1)

        batch_size, num_graphs, _ = X.shape
        X_by_graph = X.permute(1, 0, 2).contiguous()
        flat_X = X_by_graph.reshape(num_graphs * batch_size, self.num_nodes)
        neighbor_X = flat_X.index_select(1, neighbor_index).reshape(
            num_graphs,
            batch_size,
            self.num_nodes,
            self.num_neighbors,
        )
        edge_diff = X_by_graph.unsqueeze(3) - neighbor_X
        output = (w.unsqueeze(1) * edge_diff).sum(dim=3)
        return output.permute(1, 0, 2).contiguous()

    def apply_L_plus_J(self, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply ``L + J``, where ``J`` maps each row to its mean value."""
        Lx = self.apply_L(X, w)
        Jx = X.mean(dim=-1, keepdim=True).expand_as(X)
        return Lx + Jx

    def apply_L_plus_epsilon_I(
        self,
        X: torch.Tensor,
        w: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply ``L + epsilon * I``."""
        epsilon_value = self.epsilon if epsilon is None else float(epsilon)
        if epsilon_value <= 0.0:
            raise ValueError("epsilon must be positive.")
        Lx = self.apply_L(X, w)
        return Lx + epsilon_value * X


def _demo_multi_head_unrolled_cg() -> None:
    torch.manual_seed(7)

    num_heads = 3
    batch_size = 4
    num_nodes = 6
    num_neighbors = 6
    mu = 0.1

    rows = []
    for node in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if j != node]
        rows.append(neighbors + [-1])
    neighbor_list = torch.tensor(rows, dtype=torch.long)

    neighbor_mask = torch.zeros(
        num_heads,
        num_nodes,
        num_neighbors,
        dtype=torch.bool,
    )
    edge_sets = (
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)),
        ((0, 1), (0, 2), (0, 3), (3, 4), (4, 5)),
        ((0, 5), (1, 5), (2, 5), (2, 3), (3, 4)),
    )
    neighbor_position = {
        (node, int(neighbor)): pos
        for node, row in enumerate(neighbor_list.tolist())
        for pos, neighbor in enumerate(row)
        if neighbor >= 0
    }
    for head, edges in enumerate(edge_sets):
        for u, v in edges:
            neighbor_mask[head, u, neighbor_position[(u, v)]] = True
            neighbor_mask[head, v, neighbor_position[(v, u)]] = True

    row_degrees = neighbor_mask.sum(dim=-1)
    assert all(torch.unique(row_degrees[head]).numel() > 1 for head in range(num_heads))
    assert not neighbor_mask[..., -1].any()

    x = torch.randn(batch_size, num_heads, num_nodes)
    graph_learner = GraphLearningModule(
        num_nodes,
        num_neighbors,
        neighbor_list,
        neighbor_mask,
        emb_dim=4,
        feature_dim=3,
    )
    graph_ops = SparseGraphOperators(
        num_nodes,
        neighbor_list,
        neighbor_mask,
        c=4.0,
        scale=True,
    )

    w_raw = graph_learner(x)
    w, scale = graph_ops.scale_w(w_raw)
    Lx = graph_ops.apply_L(x, w)
    LJx = graph_ops.apply_L_plus_J(x, w)
    Lepsx = graph_ops.apply_L_plus_epsilon_I(x, w, epsilon=0.05)

    assert w_raw.shape == (num_heads, num_nodes, num_neighbors)
    assert w.shape == (num_heads, num_nodes, num_neighbors)
    assert scale.shape == (num_heads,)
    assert Lx.shape == x.shape
    assert LJx.shape == x.shape
    assert Lepsx.shape == x.shape
    assert torch.all(w[~neighbor_mask] == 0)
    assert torch.allclose(LJx, Lx + x.mean(dim=-1, keepdim=True).expand_as(x))
    assert torch.allclose(Lepsx, Lx + 0.05 * x)

    def lhs(z: torch.Tensor) -> torch.Tensor:
        return z + mu * graph_ops.apply_L(z, w)

    solver = UnrolledCG(
        CG_iters=25,
        alpha_init=0.5,
        beta_init=0.0,
        num_heads=num_heads,
    )
    z = solver(lhs, x)
    initial_residual = (lhs(torch.zeros_like(x)) - x).norm(dim=(0, 2))
    final_residual = (lhs(z) - x).norm(dim=(0, 2))

    print("neighbor_list shape:", tuple(neighbor_list.shape))
    print("neighbor_list", neighbor_list)
    print("row degrees per head:")
    print(row_degrees)
    print("valid neighbor list per head:")
    print(neighbor_list.masked_fill(~neighbor_mask, -1))
    print("x shape:", tuple(x.shape))
    print("learned w shape:", tuple(w.shape))
    print("scale shape:", tuple(scale.shape), "scale:", scale.detach())
    print("Lx/LJx/Lepsx shapes:", tuple(Lx.shape), tuple(LJx.shape), tuple(Lepsx.shape))
    print("z shape:", tuple(z.shape))
    print("initial residual per head:", initial_residual.detach())
    print("final residual per head:", final_residual.detach())
    print("residual ratio per head:", (final_residual / initial_residual).detach())


__all__ = ["UnrolledCG", "GraphLearningModule", "SparseGraphOperators"]


if __name__ == "__main__":
    _demo_multi_head_unrolled_cg()
