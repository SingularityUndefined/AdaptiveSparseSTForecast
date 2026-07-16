"""Dense upper-triangular effective resistance.

Inputs are upper-triangular matrices:

* ``weight[g, i, j]`` is the conductance of edge ``(i, j)`` for ``i < j``.
* ``mask[g, i, j]`` marks whether edge ``(i, j)`` is active.

The lower triangle and diagonal are ignored.  Outputs use the same
upper-triangular layout and are zero outside the active mask.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


_INV_METHODS = ("L+J", "L+eI")


def _validate_inv_method(inv_method: str, epsilon: float) -> None:
    if inv_method not in _INV_METHODS:
        raise ValueError("inv_method must be 'L+J' or 'L+eI'.")
    if inv_method == "L+eI" and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when inv_method='L+eI'.")


def _solve_rows(num_nodes: int, inv_method: str) -> int:
    return num_nodes - 1 if inv_method == "L+J" else num_nodes


def _normalize_upper_inputs(
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)
        squeezed = True
    elif weight.dim() == 3:
        squeezed = False
    else:
        raise ValueError("weight must have shape (N, N) or (num_graphs, N, N).")

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).expand(weight.size(0), -1, -1)
    elif mask.dim() != 3:
        raise ValueError("mask must have shape (N, N) or (num_graphs, N, N).")

    if tuple(weight.shape) != tuple(mask.shape):
        raise ValueError("weight and mask must have the same shape after expansion.")
    if weight.size(1) != weight.size(2):
        raise ValueError("weight and mask must be square matrices.")
    if not weight.is_floating_point():
        raise TypeError("weight must be a floating point tensor.")

    upper = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
    mask = mask.bool() & upper
    weight = torch.triu(weight, diagonal=1) * mask.to(dtype=weight.dtype)
    return weight, mask, squeezed


def _edge_solver_rows(
    num_nodes: int,
    edge_index: torch.Tensor,
    root: int,
    inv_method: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if inv_method == "L+eI":
        return edge_index[0].clone(), edge_index[1].clone()
    if inv_method != "L+J":
        raise ValueError(f"Unsupported inv_method: {inv_method}.")

    u = edge_index[0].clone()
    v = edge_index[1].clone()
    u[u == root] = -1
    v[v == root] = -1
    u[u > root] -= 1
    v[v > root] -= 1
    return u, v


def _dense_laplacian_from_upper(weight: torch.Tensor) -> torch.Tensor:
    adjacency = weight + weight.transpose(-1, -2)
    degree = adjacency.sum(dim=-1)
    return torch.diag_embed(degree) - adjacency


def _system_matrix_from_laplacian(
    laplacian: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
) -> torch.Tensor:
    num_nodes = int(laplacian.size(-1))
    if inv_method == "L+J":
        keep = torch.arange(num_nodes, device=laplacian.device)
        keep = keep[keep != root]
        return laplacian.index_select(1, keep).index_select(2, keep)

    eye = torch.eye(num_nodes, dtype=laplacian.dtype, device=laplacian.device)
    return laplacian + float(epsilon) * eye.unsqueeze(0)


def _cholesky_factor(matrix: torch.Tensor) -> torch.Tensor:
    chol, info = torch.linalg.cholesky_ex(matrix, upper=False)
    if bool((info != 0).any()):
        bad_graph = int((info != 0).nonzero(as_tuple=False)[0, 0])
        raise RuntimeError(
            "Dense ER Cholesky failed for graph "
            f"{bad_graph}. Check graph connectivity and positive active weights."
        )
    return chol


def _edge_rhs(
    solve_rows: int,
    edge_u_rows: torch.Tensor,
    edge_v_rows: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_edges = int(edge_u_rows.numel())
    rhs = torch.zeros((solve_rows, num_edges), dtype=dtype, device=device)
    if num_edges == 0:
        return rhs

    columns = torch.arange(num_edges, device=device)
    u_mask = edge_u_rows >= 0
    v_mask = edge_v_rows >= 0
    if bool(u_mask.any()):
        rhs[edge_u_rows[u_mask], columns[u_mask]] += 1.0
    if bool(v_mask.any()):
        rhs[edge_v_rows[v_mask], columns[v_mask]] -= 1.0
    return rhs


def _gather_solution_rows_batched(
    potentials: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    if bool((rows >= 0).all()):
        return potentials.index_select(1, rows)

    values = potentials.new_zeros(
        (potentials.size(0), rows.numel(), potentials.size(2))
    )
    mask = rows >= 0
    if bool(mask.any()):
        values[:, mask] = potentials.index_select(1, rows[mask])
    return values


class _DenseUpperEffectiveResistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        edge_weight: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        root: int,
        inv_method: str,
        epsilon: float,
    ) -> torch.Tensor:
        if edge_weight.dim() != 2:
            raise ValueError("edge_weight must have shape (num_graphs, num_edges).")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")
        if edge_index.size(1) != edge_weight.size(1):
            raise ValueError("edge_weight and edge_index disagree on num_edges.")
        _validate_inv_method(inv_method, epsilon)
        if root < 0 or root >= num_nodes:
            raise ValueError(f"root must be in [0, {num_nodes}), got {root}.")

        num_graphs = int(edge_weight.size(0))
        num_edges = int(edge_weight.size(1))
        solve_rows = _solve_rows(num_nodes, inv_method)
        if num_edges == 0:
            ctx.save_for_backward(
                edge_weight.new_empty((num_graphs, solve_rows, 0)),
                edge_weight.new_empty(0, dtype=torch.long),
                edge_weight.new_empty(0, dtype=torch.long),
            )
            return edge_weight.new_zeros((num_graphs, 0))

        if bool((edge_weight <= 0).any()):
            raise ValueError("Active edge weights must be positive.")

        device = edge_weight.device
        dtype = edge_weight.dtype
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_u_rows, edge_v_rows = _edge_solver_rows(
            num_nodes,
            edge_index,
            root,
            inv_method,
        )

        upper_weight = edge_weight.new_zeros((num_graphs, num_nodes, num_nodes))
        upper_weight[:, edge_index[0], edge_index[1]] = edge_weight
        laplacian = _dense_laplacian_from_upper(upper_weight)
        matrix = _system_matrix_from_laplacian(
            laplacian,
            root=root,
            inv_method=inv_method,
            epsilon=epsilon,
        )
        chol = _cholesky_factor(matrix)

        rhs = _edge_rhs(
            solve_rows,
            edge_u_rows,
            edge_v_rows,
            dtype=dtype,
            device=device,
        )
        potentials = torch.cholesky_solve(
            rhs.unsqueeze(0).expand(num_graphs, -1, -1),
            chol,
            upper=False,
        )
        resistance = (rhs.unsqueeze(0) * potentials).sum(dim=1)
        ctx.save_for_backward(potentials, edge_u_rows, edge_v_rows)
        return resistance

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        potentials, edge_u_rows, edge_v_rows = ctx.saved_tensors
        num_graphs = int(potentials.size(0))
        num_edges = int(edge_u_rows.numel())
        grad_edge_weight = grad_output.new_zeros((num_graphs, num_edges))
        if num_edges == 0 or not ctx.needs_input_grad[0]:
            return grad_edge_weight, None, None, None, None, None

        potential_u = _gather_solution_rows_batched(potentials, edge_u_rows)
        potential_v = _gather_solution_rows_batched(potentials, edge_v_rows)
        transfer = potential_u - potential_v
        grad_edge_weight -= (
            transfer.square() * grad_output.contiguous().unsqueeze(2)
        ).sum(dim=1)
        return grad_edge_weight, None, None, None, None, None


def dense_upper_effective_resistance(
    weight: torch.Tensor,
    mask: torch.Tensor,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
) -> torch.Tensor:
    weight, mask, squeezed = _normalize_upper_inputs(weight, mask)
    num_graphs, num_nodes, _ = weight.shape
    output = weight.new_zeros(weight.shape)

    for graph in range(num_graphs):
        rows, cols = mask[graph].nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        edge_index = torch.stack((rows, cols), dim=0)
        edge_weight = weight[graph, rows, cols].unsqueeze(0)
        edge_resistance = _DenseUpperEffectiveResistanceFunction.apply(
            edge_weight,
            edge_index,
            int(num_nodes),
            int(root),
            inv_method,
            float(epsilon),
        )
        output[graph, rows, cols] = edge_resistance.squeeze(0)

    if squeezed:
        return output.squeeze(0)
    return output


class DenseUpperEffectiveResistance(nn.Module):
    """Exact ER for dense upper-triangular graph matrices."""

    def __init__(
        self,
        *,
        root: int = 0,
        inv_method: str = "L+J",
        epsilon: float = 0.2,
    ) -> None:
        super().__init__()
        self.root = int(root)
        self.inv_method = inv_method
        self.epsilon = float(epsilon)
        _validate_inv_method(self.inv_method, self.epsilon)

    def forward(self, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return dense_upper_effective_resistance(
            weight,
            mask,
            root=self.root,
            inv_method=self.inv_method,
            epsilon=self.epsilon,
        )

