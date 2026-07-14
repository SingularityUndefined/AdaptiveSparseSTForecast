import torch
import torch.nn as nn
from effective_resistance import MultiGraphEffectiveResistance
from effective_resistance_torch import TorchDenseMultiGraphEffectiveResistance
from utils_modules import UnrolledCG, GraphLearningModule, SparseGraphOperators


def _build_effective_resistance_solver(
    backend,
    neighbor_list,
    input_neighbor_mask,
    epsilon,
    backward_chunk_size,
):
    if backend == "sksparse":
        return MultiGraphEffectiveResistance(
            neighbor_list,
            inv_method="L+J",
            epsilon=epsilon,
            backward_chunk_size=backward_chunk_size,
        )
    if backend == "torch_dense":
        if input_neighbor_mask is None:
            raise ValueError("torch_dense ER backend requires input_neighbor_mask.")
        return TorchDenseMultiGraphEffectiveResistance(
            neighbor_list,
            input_neighbor_mask,
            shared_neighbor_list=True,
            inv_method="L+J",
            epsilon=epsilon,
            backward_chunk_size=backward_chunk_size,
        )
    raise ValueError("er_backend must be 'sksparse' or 'torch_dense'.")


class UnrolledGEMBlock(nn.Module):
    '''
    Unrolled Graph Embedding Module (GEM) Block for multi-graph learning.
    
    E-step: given signal y and current graph W^o, S, solve the smoothness problem $(I+\mu L)^{-1} x = y$ by unrolled CG.

    M-step: given signal x and current graph W^o, S, update the graph by unrolling the gradient descent steps for the learned graph W, and its connectivity mask S.

    Solve with multi-head graph learning.
    
    Shape of input y: (batch_size, num_nodes)
    Shape of input W^o: (num_graphs, num_nodes, k)
    Shape of input S: (num_graphs, num_nodes, k)
    Shape of neighbor list: (num_nodes, k) (possible neighbors are pre-defined)
    '''
    def __init__(self, num_nodes, neighbor_list, num_heads, E_iters=5, M_iters=5, GD_step_init=0.1, mu_init=0.2, gamma_init=0.4, c=20, scale=True, epsilon=0.2, xi=1, input_neighbor_mask=None, er_backend="sksparse", er_backward_chunk_size=1024):
        super(UnrolledGEMBlock, self).__init__()
        self.num_nodes = num_nodes
        self.neighbor_list = neighbor_list
        self.num_heads = num_heads
        self.E_iters = E_iters
        self.M_iters = M_iters
        self.er_backend = er_backend

        # E-step block: Unrolled Conjugate Gradient for solving smoothness problem
        self.CG_solver = UnrolledCG(self.E_iters, 0.1, 0, self.num_heads, init_method='uniform', init_scale=0.02)

        # M-step block: Graph Learning with GD for updating graph weights and connectivity mask
        self.c = c
        self.scale = scale
        self.epsilon = epsilon
        self.xi = xi # sparse penalty offset

        # PARAMETERS: learnable parameters for the unrolled optimization
        # TODO: check whether mu should be in (1,) or in (M_iters,)
        self.mu = nn.Parameter(torch.ones(1,) * mu_init, requires_grad=True)
        # TODO: check whether gamma should be in (1,) or in (M_iters,)
        self.gamma_list = nn.Parameter(torch.ones(self.M_iters) * gamma_init, requires_grad=True)
        self.step_size_list = nn.Parameter(torch.ones(self.M_iters) * GD_step_init, requires_grad=True)

        # Reusable block for computing effective resistance without parameters.
        self.ER_solver = _build_effective_resistance_solver(
            self.er_backend,
            self.neighbor_list,
            input_neighbor_mask,
            self.epsilon,
            er_backward_chunk_size,
        )

        # Graph Operator block
        # self.graph_op = SparseGraphOperators(self.num_nodes, self.neighbor_list, self.input_S, c=self.c, scale=self.scale, epsilon=self.epsilon)

    def edge_diff_square(self, x, input_S):
        """
        Compute the squared differences of node features across edges defined by the neighbor mask.
        x: Node features of shape (batch_size, num_heads, num_nodes)
        input_S: Neighbor mask of shape (num_heads, num_nodes, k)
        Returns: Squared differences of shape (num_heads, num_nodes, k)
        """
        batch_size, num_heads, num_nodes = x.size()
        k = input_S.size(2)  # Number of neighbors

        mask = input_S.to(device=x.device, dtype=x.dtype)
        edge_diffs = x.unsqueeze(-1) - x[:, :, self.neighbor_list.view(-1)].reshape(batch_size, num_heads, num_nodes, -1)  # Shape: (batch_size, num_heads, num_nodes, k)
        edge_diffs = edge_diffs * mask.unsqueeze(0)  # Apply neighbor mask
        return (edge_diffs ** 2).mean(dim=0)  # Shape: (num_heads, num_nodes, k)
        
    def apply_L(self, x, W, S):
        """
        Sparse implementation of graph Laplacian operator L applied to x, given graph weights W and connectivity mask S.
        input:
            x: Node features of shape (batch_size, num_heads, num_nodes)
            W: Graph weights of shape (num_heads, num_nodes, k)
            S: Connectivity mask of shape (num_heads, num_nodes, k)
        output:
            Lx: Graph Laplacian applied to x, of shape (batch_size, num_heads, num_nodes)
        """
        # compute edge differences
        batch_size, num_heads, num_nodes = x.size()
        k = S.size(2)  # Number of neighbors
        mask = S.to(device=x.device, dtype=x.dtype)
        edge_diffs = x.unsqueeze(-1) - x[:, :, self.neighbor_list.view(-1)].reshape(batch_size, num_heads, num_nodes, -1)  # Shape: (batch_size, num_heads, num_nodes, k)
        edge_diffs = edge_diffs * mask.unsqueeze(0)  # Apply connectivity mask
        # compute weighted sum of edge differences
        weighted_edge_diffs = edge_diffs * W.unsqueeze(0)  # Shape: (batch_size, num_heads, num_nodes, k)
        Lx = weighted_edge_diffs.sum(dim=-1)  # Sum over neighbors, shape: (batch_size, num_heads, num_nodes)
        return Lx
    
    ###### NOT USED ##################
    def apply_L_plus_epsilon_I(self, x, W, S):
        """
        Sparse implementation of (L + epsilon * I) applied to x, given graph weights W and connectivity mask S.
        input:
            x: Node features of shape (batch_size, num_heads, num_nodes)
            W: Graph weights of shape (num_heads, num_nodes, k)
            S: Connectivity mask of shape (num_heads, num_nodes, k)
        output:
            (L + epsilon * I)x: Graph Laplacian plus epsilon times identity applied to x, of shape (batch_size, num_heads, num_nodes)
        """
        Lx = self.apply_L(x, W, S)  # Apply graph Laplacian
        return Lx + self.epsilon * x  # Add epsilon * I * x 
    #############################################

    def scale_graph_weights(self, W, S):
        """
        Scale the graph weights W to meet the F-norm constraints of self.c
        """
        # Compute the Frobenius norm of W considering only the edges defined by S
        fro_norm = torch.sqrt((W * S).pow(2).sum(dim=(1, 2)))  # Shape: (num_heads,)
        # Compute scaling factors to meet the constraint
        scale_factors = self.c / (fro_norm + 1e-8)  # Shape: (num_heads,)
        # Scale W accordingly
        W_scaled = W * scale_factors.view(-1, 1, 1)  # Shape: (num_heads, num_nodes, k)
        return W_scaled

    def E_step(self, y, W, S):
        if y.ndim == 2:
            y = y.unsqueeze(1).repeat(1, self.num_heads, 1) # (batch_size, num_nodes) -> (batch_size, num_heads, num_nodes)

        def LHS(x):
            """
            Compute the left-hand side of the smoothness problem: (I + mu * L) * x
            where L is the graph Laplacian derived from W.
            """
            # Compute graph Laplacian L from W
            return x + self.mu * self.apply_L(x, W, S)  # (I + mu * L) * x

        x = self.CG_solver(LHS, y)
        return x

    def M_step(self, x, W, input_S):
        candidate_mask = input_S.bool()
        neighbor_mask = candidate_mask
        x_diff_square = self.edge_diff_square(x, input_S)  # Shape: (num_heads, num_nodes, k)
        for i in range(self.M_iters):
            # Compute effective resistance for the current graph W
            assert W[~candidate_mask].abs().sum() == 0, "W has non-zero entries outside the candidate mask"
            resistance = self.ER_solver(W, neighbor_mask)  # Shape: (num_heads, num_nodes, k)
            
            # TODO: check which epsilon to be used here, the one in ER_solver or the one in M_step
            sparse_penalty = self.gamma_list[i] * (W + self.xi)  # Shape: (num_heads, num_nodes, k)
            grad = x_diff_square + sparse_penalty - resistance  # Shape: (num_heads, num_nodes, k)
            grad = grad * neighbor_mask.to(dtype=grad.dtype, device=grad.device)  # Apply neighbor mask to gradient

            # gradient step
            W = torch.clamp(W - self.step_size_list[i] * grad, min=0)  # Shape: (num_heads, num_nodes, k)
            W = W * candidate_mask.to(dtype=W.dtype, device=W.device)
            
            neighbor_mask = (W > 0) & candidate_mask  # Update active mask based on non-zero candidate weights
            # Optionally apply constraints or normalization to W here if needed
            # W = self.graph_op.scale_w(W)
        return W
    
    
    def forward(self, y, W_o, input_S, default_threshold=1e-4):
        """
        y: Node features of shape (batch_size, num_nodes, in_channels)
        W_o: Initial graph weights of shape (num_heads, num_nodes, k)
        input_S: Connectivity mask of shape (num_heads, num_nodes, k)
        """
        batch_size = y.size(0)
        # E-step: Solve for x given y and current graph W_o
        x = self.E_step(y, W_o, input_S)  # Shape: (batch_size, num_heads, num_nodes)
        # M-step: Update graph weights W given x and current graph W_o
        W = self.M_step(x, W_o, input_S)  # Shape: (num_heads, num_nodes, k)
        # compute graph connectivity mask S based on updated W by percentile thresholding
        threshold = torch.clamp(torch.quantile(W.view(-1), 0.05, keepdim=True), min=default_threshold)  # 5th percentile threshold for sparsity

        W_new = torch.clamp(W, min=0)  # Ensure non-negativity
        W_new = torch.where(W < threshold, torch.zeros_like(W), W)  # Zero out weights below the threshold
        S = (W_new > 0).float()  # Shape: (num_heads, num_nodes, k)

        if self.scale:
            W_new = self.scale_graph_weights(W_new, S)  # Scale W to meet F-norm constraints

        return x, W_new, S # sparsified W by quantile thresholding, and its connectivity mask S


def _build_grid_window_neighbors(grid_size=7, window_size=5):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    radius = window_size // 2
    num_nodes = grid_size * grid_size
    max_neighbors = window_size * window_size - 1
    neighbor_list = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long)
    neighbor_mask = torch.zeros(1, num_nodes, max_neighbors, dtype=torch.bool)

    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            neighbors = []
            for drow in range(-radius, radius + 1):
                for dcol in range(-radius, radius + 1):
                    if drow == 0 and dcol == 0:
                        continue
                    nrow = row + drow
                    ncol = col + dcol
                    if 0 <= nrow < grid_size and 0 <= ncol < grid_size:
                        neighbors.append(nrow * grid_size + ncol)

            neighbor_list[node, : len(neighbors)] = torch.tensor(neighbors)
            neighbor_mask[0, node, : len(neighbors)] = True

    return neighbor_list, neighbor_mask


def _make_grid_signal(grid_size=7):
    coords = torch.linspace(-1.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    signal = (
        torch.sin(torch.pi * xx)
        + 0.6 * torch.cos(torch.pi * yy)
        + torch.exp(-5.0 * (xx.square() + yy.square()))
    )
    signal = (signal - signal.mean()) / signal.std()
    return signal.reshape(1, -1)


def _cg_with_residual_trace(block, y, W, S):
    if y.ndim == 2:
        y = y.unsqueeze(1).repeat(1, block.num_heads, 1)

    def lhs(z):
        return z + block.mu * block.apply_L(z, W, S)

    x = torch.zeros_like(y)
    residual = y - lhs(x)
    direction = residual.clone()
    print("E-step residuals:")
    print(f"  iter 00: {residual.norm().item():.6e}")

    for idx, (alpha, beta) in enumerate(
        zip(block.CG_solver.alpha, block.CG_solver.beta),
        start=1,
    ):
        alpha = alpha.view(1, -1, 1)
        beta = beta.view(1, -1, 1)
        A_direction = lhs(direction)
        x = x + alpha * direction
        residual = residual - alpha * A_direction
        direction = residual + beta * direction
        print(f"  iter {idx:02d}: {residual.norm().item():.6e}")

    return x


def _summarize_weights(prefix, W, S):
    active = S.bool()
    values = W[active]
    undirected_edges = int(active.sum().item() // 2)
    if values.numel() == 0:
        print(f"{prefix}: no active edges")
        return
    print(
        f"{prefix}: min={values.min().item():.6f}, "
        f"p25={values.quantile(0.25).item():.6f}, "
        f"mean={values.mean().item():.6f}, "
        f"p75={values.quantile(0.75).item():.6f}, "
        f"max={values.max().item():.6f}, "
        f"remaining_edges={undirected_edges}"
    )


def _m_step_with_weight_trace(block, x, W, S, default_threshold=1e-4):
    candidate_mask = S.bool()
    neighbor_mask = candidate_mask
    x_diff_square = block.edge_diff_square(x, neighbor_mask.float())

    print("M-step weight distributions:")
    _summarize_weights("  iter 00", W, neighbor_mask)
    for idx in range(block.M_iters):
        if W[~candidate_mask].abs().sum() != 0:
            raise ValueError("W has non-zero entries outside the candidate mask.")

        resistance = block.ER_solver(W, neighbor_mask)
        sparse_penalty = block.gamma_list[idx] * (W + block.xi)
        grad = x_diff_square + sparse_penalty - resistance
        grad = grad * neighbor_mask.to(dtype=grad.dtype, device=grad.device)
        W = torch.clamp(W - block.step_size_list[idx] * grad, min=0)
        W = W * candidate_mask.to(dtype=W.dtype, device=W.device)
        neighbor_mask = (W > 0) & candidate_mask

        threshold = torch.clamp(
            torch.quantile(W.view(-1), 0.05, keepdim=True),
            min=default_threshold,
        )
        W_sparse = torch.where(W < threshold, torch.zeros_like(W), W)
        S_sparse = W_sparse > 0
        _summarize_weights(f"  iter {idx + 1:02d}", W_sparse, S_sparse)

    return W


def _demo_grid_forward() -> None:
    torch.manual_seed(7)

    grid_size = 7
    num_heads = 1
    num_nodes = grid_size * grid_size
    neighbor_list, input_S = _build_grid_window_neighbors(grid_size, window_size=5)

    clean_signal = _make_grid_signal(grid_size)
    noisy_signal = clean_signal + 0.25 * torch.randn_like(clean_signal)

    W_o = input_S.float()
    block = UnrolledGEMBlock(
        num_nodes,
        neighbor_list,
        num_heads,
        E_iters=6,
        M_iters=10,
        GD_step_init=0.1,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
        xi=1.0,
    )

    print("7x7 grid GEM block demo")
    print("neighbor_list shape:", tuple(neighbor_list.shape))
    print("input_S shape:", tuple(input_S.shape))
    print("initial undirected edges:", int(input_S.sum().item() // 2))
    print("clean signal shape:", tuple(clean_signal.shape))
    print("noisy signal shape:", tuple(noisy_signal.shape))

    with torch.no_grad():
        x_trace = _cg_with_residual_trace(block, noisy_signal, W_o, input_S)
        _m_step_with_weight_trace(block, x_trace, W_o, input_S)
        x, W_new, S_new = block(noisy_signal, W_o, input_S)

    print("forward output shapes:")
    print("  x:", tuple(x.shape))
    print("  W_new:", tuple(W_new.shape))
    print("  S_new:", tuple(S_new.shape))
    print("final undirected edges from S:", int(S_new.sum().item() // 2))


if __name__ == "__main__":
    _demo_grid_forward()
