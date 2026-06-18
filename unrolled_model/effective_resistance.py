"""Exact effective resistance for GEM-style sparse neighbor weights.

``GEM_sparse/GEM_module_sparse_new.py`` stores sparse graph weights as
``neighbor_list`` and ``weight`` tensors with shape ``(N, k)``:

* ``neighbor_list[i, j]`` is the neighbor node id of node ``i``.
* ``neighbor_list[i, j] == -1`` marks padding / no edge.
* ``weight[i, j]`` is the corresponding edge conductance and is masked to zero
  at padding positions.

This module first converts that representation to a unique undirected edge
list:

    edge_weight: (m,)
    edge_index:  (2, m)

The effective resistance is computed on the unique edge list with CHOLMOD in
the forward pass, then scattered back to the original ``(N, k)`` shape.  The
manual backward is implemented at the edge-list level; PyTorch handles the
chain rule from ``edge_weight`` back to the original ``weight`` tensor.

Two inverse conventions are supported, matching ``GEM_module_sparse_new.py``:

* ``inv_method="L+J"`` factors a grounded/reduced Laplacian.  For zero-sum
  edge RHS vectors this is equivalent to applying ``(L + J)^-1``.
* ``inv_method="L+eI"`` factors the full regularized matrix
  ``L + epsilon * I``.

For the factored matrix ``A(c)`` and edge incidence vectors ``b_e``:

    r_e = b_e.T A(c)^-1 b_e
    d r_e / d c_f = -(b_e.T A(c)^-1 b_f)^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sksparse.cholmod import cholesky


_INV_METHODS = ("L+J", "L+eI")


@dataclass(frozen=True)
class NeighborToEdgeMap:
    """Mapping between GEM ``(N, k)`` entries and unique undirected edges."""

    original_shape: Tuple[int, int]
    valid_flat_index: torch.Tensor
    entry_to_edge: torch.Tensor
    entry_alpha: torch.Tensor
    edge_index: torch.Tensor


def _as_long_tensor(x: torch.Tensor | np.ndarray | Iterable[Iterable[int]]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.long, device="cpu")
    return torch.as_tensor(x, dtype=torch.long, device="cpu")


def _validate_inv_method(inv_method: str, epsilon: float) -> None:
    if inv_method not in _INV_METHODS:
        raise ValueError("inv_method must be 'L+J' or 'L+eI'.")
    if inv_method == "L+eI" and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when inv_method='L+eI'.")


def _solve_rows(num_nodes: int, inv_method: str) -> int:
    return num_nodes - 1 if inv_method == "L+J" else num_nodes


def build_neighbor_to_edge_map(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    *,
    average_duplicates: bool = True,
) -> NeighborToEdgeMap:
    """Build the fixed mapping from GEM ``neighbor_list`` to an edge list.

    Repeated entries of the same undirected edge, such as ``u -> v`` and
    ``v -> u``, are mapped to the same edge id.  With ``average_duplicates=True``
    their weights are averaged into one conductance; this matches the common
    symmetric storage where both directions contain the same value.
    """
    neighbor_cpu = _as_long_tensor(neighbor_list)
    if neighbor_cpu.dim() != 2:
        raise ValueError("neighbor_list must have shape (num_nodes, num_neighbors).")

    num_nodes, width = neighbor_cpu.shape
    flat_neighbor = neighbor_cpu.reshape(-1)
    flat_rows = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(width)
    valid = flat_neighbor >= 0
    if valid.any() and bool((flat_neighbor[valid] >= num_nodes).any()):
        bad_index = int(flat_neighbor[valid][flat_neighbor[valid] >= num_nodes][0])
        raise ValueError(
            f"neighbor_list contains node index {bad_index}, but num_nodes={num_nodes}."
        )

    valid &= flat_neighbor != flat_rows
    valid_flat_index = valid.nonzero(as_tuple=False).flatten()
    if valid_flat_index.numel() == 0:
        return NeighborToEdgeMap(
            original_shape=(int(num_nodes), int(width)),
            valid_flat_index=valid_flat_index,
            entry_to_edge=torch.empty(0, dtype=torch.long),
            entry_alpha=torch.empty(0, dtype=torch.float64),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    rows = flat_rows[valid]
    cols = flat_neighbor[valid]
    edges = torch.stack((torch.minimum(rows, cols), torch.maximum(rows, cols)), dim=1)
    unique_edges, entry_to_edge = torch.unique(
        edges,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    if average_duplicates:
        counts = torch.bincount(entry_to_edge, minlength=unique_edges.size(0))
        entry_alpha = counts.to(dtype=torch.float64).reciprocal()[entry_to_edge]
    else:
        entry_alpha = torch.ones(entry_to_edge.numel(), dtype=torch.float64)

    edge_index = unique_edges.t().contiguous()

    return NeighborToEdgeMap(
        original_shape=(int(num_nodes), int(width)),
        valid_flat_index=valid_flat_index,
        entry_to_edge=entry_to_edge,
        entry_alpha=entry_alpha,
        edge_index=edge_index,
    )


def neighbor_weights_to_edge_list(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    weight: torch.Tensor,
    *,
    average_duplicates: bool = True,
    return_mapping: bool = False,
):
    """Convert GEM ``(N, k)`` weights to ``edge_weight`` and ``edge_index``.

    Returns:
        By default, ``(edge_weight, edge_index)`` where ``edge_weight`` has
        shape ``(m,)`` and ``edge_index`` has shape ``(2, m)``.

        If ``return_mapping=True``, returns
        ``(edge_weight, edge_index, mapping)``.  The mapping can be passed to
        :func:`edge_values_to_neighbor_shape` to scatter edge values back to
        the original ``(N, k)`` layout.
    """
    mapping = build_neighbor_to_edge_map(
        neighbor_list, average_duplicates=average_duplicates
    )
    edge_weight = neighbor_weights_to_edge_weight(weight, mapping)
    edge_index = mapping.edge_index.to(device=weight.device)

    if return_mapping:
        return edge_weight, edge_index, mapping
    return edge_weight, edge_index


def neighbor_weights_to_edge_weight(
    weight: torch.Tensor,
    mapping: NeighborToEdgeMap,
) -> torch.Tensor:
    """Aggregate ``weight`` with shape ``(N, k)`` to edge weights ``(m,)``."""
    return _aggregate_edge_weight(
        weight,
        original_shape=mapping.original_shape,
        valid_flat_index=mapping.valid_flat_index,
        entry_to_edge=mapping.entry_to_edge,
        entry_alpha=mapping.entry_alpha,
        num_edges=int(mapping.edge_index.size(1)),
    )


def _aggregate_edge_weight(
    weight: torch.Tensor,
    *,
    original_shape: Tuple[int, int],
    valid_flat_index: torch.Tensor,
    entry_to_edge: torch.Tensor,
    entry_alpha: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    if tuple(weight.shape) != original_shape:
        raise ValueError("weight shape does not match the neighbor mapping.")

    device = weight.device
    dtype = weight.dtype
    valid_flat_index = valid_flat_index.to(device=device)
    entry_to_edge = entry_to_edge.to(device=device)
    entry_alpha = entry_alpha.to(device=device, dtype=dtype)

    edge_weight = weight.new_zeros(num_edges)
    if num_edges == 0:
        return edge_weight

    valid_weight = weight.reshape(-1).index_select(0, valid_flat_index)
    edge_weight.scatter_add_(0, entry_to_edge, valid_weight * entry_alpha)
    return edge_weight


def _aggregate_multi_edge_weight(
    weight: torch.Tensor,
    *,
    original_shape: Tuple[int, int],
    valid_flat_index: torch.Tensor,
    entry_to_edge: torch.Tensor,
    entry_alpha: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    if weight.dim() != 3 or tuple(weight.shape[1:]) != original_shape:
        raise ValueError("multi-head weight must have shape (num_heads, N, k).")

    device = weight.device
    dtype = weight.dtype
    num_heads = int(weight.size(0))
    valid_flat_index = valid_flat_index.to(device=device)
    entry_to_edge = entry_to_edge.to(device=device)
    entry_alpha = entry_alpha.to(device=device, dtype=dtype)

    edge_weight = weight.new_zeros((num_heads, num_edges))
    if num_edges == 0:
        return edge_weight

    valid_weight = weight.reshape(num_heads, -1).index_select(1, valid_flat_index)
    weighted_entries = valid_weight * entry_alpha.unsqueeze(0)
    edge_ids = entry_to_edge.unsqueeze(0).expand(num_heads, -1)
    edge_weight.scatter_add_(1, edge_ids, weighted_entries)
    return edge_weight


def edge_values_to_neighbor_shape(
    edge_values: torch.Tensor,
    mapping: NeighborToEdgeMap,
) -> torch.Tensor:
    """Scatter edge-list values ``(m,)`` back to the GEM ``(N, k)`` layout."""
    return _scatter_edge_values(
        edge_values,
        original_shape=mapping.original_shape,
        valid_flat_index=mapping.valid_flat_index,
        entry_to_edge=mapping.entry_to_edge,
        num_edges=int(mapping.edge_index.size(1)),
    )


def _scatter_edge_values(
    edge_values: torch.Tensor,
    *,
    original_shape: Tuple[int, int],
    valid_flat_index: torch.Tensor,
    entry_to_edge: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    if edge_values.dim() != 1:
        raise ValueError("edge_values must have shape (num_edges,).")
    if edge_values.numel() != num_edges:
        raise ValueError("edge_values length does not match the neighbor mapping.")

    output = edge_values.new_zeros(original_shape)
    if edge_values.numel() == 0:
        return output

    valid_flat_index = valid_flat_index.to(device=edge_values.device)
    entry_to_edge = entry_to_edge.to(device=edge_values.device)
    output.reshape(-1).index_copy_(
        0,
        valid_flat_index,
        edge_values.index_select(0, entry_to_edge),
    )
    return output


def _scatter_multi_edge_values(
    edge_values: torch.Tensor,
    *,
    original_shape: Tuple[int, int],
    valid_flat_index: torch.Tensor,
    entry_to_edge: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    if edge_values.dim() != 2:
        raise ValueError("multi-head edge_values must have shape (num_heads, num_edges).")
    if edge_values.size(1) != num_edges:
        raise ValueError("edge_values length does not match the neighbor mapping.")

    num_heads = int(edge_values.size(0))
    output = edge_values.new_zeros((num_heads, *original_shape))
    if num_edges == 0:
        return output

    valid_flat_index = valid_flat_index.to(device=edge_values.device)
    entry_to_edge = entry_to_edge.to(device=edge_values.device)
    gathered = edge_values.index_select(1, entry_to_edge)
    output.reshape(num_heads, -1).index_copy_(1, valid_flat_index, gathered)
    return output


def neighbor_weights_to_dense_adjacency(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    weight: torch.Tensor,
) -> torch.Tensor:
    """Scatter GEM ``(N, k)`` weights into a dense ``(N, N)`` adjacency matrix."""
    neighbor_cpu = _as_long_tensor(neighbor_list)
    if tuple(neighbor_cpu.shape) != tuple(weight.shape):
        raise ValueError("neighbor_list and weight must have the same shape.")

    num_nodes = int(neighbor_cpu.size(0))
    neighbor = neighbor_cpu.to(device=weight.device)
    rows = torch.arange(num_nodes, device=weight.device).unsqueeze(1).expand_as(neighbor)
    mask = neighbor >= 0

    adjacency = weight.new_zeros((num_nodes, num_nodes))
    adjacency[rows[mask], neighbor[mask]] = weight[mask]
    return adjacency


def dense_matrix_to_neighbor_shape(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Gather dense ``(N, N)`` matrix entries back to GEM ``(N, k)`` layout."""
    neighbor_cpu = _as_long_tensor(neighbor_list)
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must have shape (N, N).")
    if matrix.size(0) != neighbor_cpu.size(0):
        raise ValueError("matrix and neighbor_list disagree on num_nodes.")

    neighbor = neighbor_cpu.to(device=matrix.device)
    rows = torch.arange(matrix.size(0), device=matrix.device).unsqueeze(1).expand_as(
        neighbor
    )
    mask = neighbor >= 0

    output = matrix.new_zeros(neighbor.shape)
    output[mask] = matrix[rows[mask], neighbor[mask]]
    return output


def _laplacian_from_edges(
    num_nodes: int,
    edge_index: np.ndarray,
    conductance: np.ndarray,
) -> sp.csc_matrix:
    u = edge_index[0].astype(np.int64, copy=False)
    v = edge_index[1].astype(np.int64, copy=False)
    num_edges = conductance.size
    rows = np.empty(4 * num_edges, dtype=np.int64)
    cols = np.empty(4 * num_edges, dtype=np.int64)
    data = np.empty(4 * num_edges, dtype=np.float64)

    rows[0:num_edges] = u
    rows[num_edges : 2 * num_edges] = v
    rows[2 * num_edges : 3 * num_edges] = u
    rows[3 * num_edges :] = v

    cols[0:num_edges] = u
    cols[num_edges : 2 * num_edges] = v
    cols[2 * num_edges : 3 * num_edges] = v
    cols[3 * num_edges :] = u

    data[0:num_edges] = conductance
    data[num_edges : 2 * num_edges] = conductance
    data[2 * num_edges : 3 * num_edges] = -conductance
    data[3 * num_edges :] = -conductance

    shape = (num_nodes, num_nodes)
    return sp.coo_matrix((data, (rows, cols)), shape=shape).tocsc()


def _remove_root(matrix: sp.csc_matrix, root: int) -> sp.csc_matrix:
    if root == 0:
        return matrix[1:, 1:].tocsc()
    if root == matrix.shape[0] - 1:
        return matrix[:-1, :-1].tocsc()
    keep = np.ones(matrix.shape[0], dtype=bool)
    keep[root] = False
    return matrix[keep][:, keep].tocsc()


def _edge_solver_rows(
    num_nodes: int,
    edge_index: np.ndarray,
    root: int,
    inv_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if inv_method == "L+eI":
        return (
            edge_index[0].astype(np.int64, copy=False),
            edge_index[1].astype(np.int64, copy=False),
        )
    if inv_method != "L+J":
        raise ValueError(f"Unsupported inv_method: {inv_method}.")

    u = edge_index[0].astype(np.int64, copy=True)
    v = edge_index[1].astype(np.int64, copy=True)
    u[u == root] = -1
    v[v == root] = -1
    u[u > root] -= 1
    v[v > root] -= 1
    return u, v


def _edge_rhs_chunk(
    solve_rows: int,
    edge_u_rows: np.ndarray,
    edge_v_rows: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    rhs = np.zeros((solve_rows, end - start), dtype=np.float64)
    if end == start:
        return rhs

    columns = np.arange(end - start)
    u = edge_u_rows[start:end]
    v = edge_v_rows[start:end]
    u_mask = u >= 0
    v_mask = v >= 0
    rhs[u[u_mask], columns[u_mask]] += 1.0
    rhs[v[v_mask], columns[v_mask]] -= 1.0
    return rhs


def _gather_solution_rows(
    potentials: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    if bool((rows >= 0).all()):
        return potentials.index_select(0, rows)

    values = potentials.new_zeros((rows.numel(), potentials.size(1)))
    mask = rows >= 0
    if mask.any():
        values[mask] = potentials.index_select(0, rows[mask])
    return values


def _gather_solution_rows_batched(
    potentials: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    if bool((rows >= 0).all()):
        return potentials.index_select(1, rows)

    values = potentials.new_zeros((potentials.size(0), rows.numel(), potentials.size(2)))
    mask = rows >= 0
    if mask.any():
        values[:, mask] = potentials.index_select(1, rows[mask])
    return values


class _EdgeEffectiveResistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        edge_weight: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        root: int,
        inv_method: str,
        epsilon: float,
        backward_chunk_size: int,
    ) -> torch.Tensor:
        if edge_weight.dim() != 1:
            raise ValueError("edge_weight must have shape (num_edges,).")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")
        if edge_index.size(1) != edge_weight.numel():
            raise ValueError("edge_weight and edge_index disagree on num_edges.")
        if not edge_weight.is_floating_point():
            raise TypeError("edge_weight must be a floating point tensor.")
        if root < 0 or root >= num_nodes:
            raise ValueError(f"root must be in [0, {num_nodes}), got {root}.")
        _validate_inv_method(inv_method, epsilon)

        device = edge_weight.device
        dtype = edge_weight.dtype
        num_edges = int(edge_weight.numel())
        solve_rows = _solve_rows(num_nodes, inv_method)

        if num_edges == 0:
            ctx.save_for_backward(
                torch.empty((solve_rows, 0), device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
            ctx.backward_chunk_size = max(1, int(backward_chunk_size))
            return edge_weight.new_zeros((0,))

        conductance_np = edge_weight.detach().cpu().to(torch.float64).numpy()
        if np.any(~np.isfinite(conductance_np)) or np.any(conductance_np <= 0.0):
            raise ValueError("All edge conductances must be positive.")

        edge_index_np = edge_index.detach().cpu().to(torch.long).numpy()
        edge_u_rows_np, edge_v_rows_np = _edge_solver_rows(
            num_nodes, edge_index_np, root, inv_method
        )
        laplacian = _laplacian_from_edges(num_nodes, edge_index_np, conductance_np)
        if inv_method == "L+J":
            matrix = _remove_root(laplacian, root)
        else:
            matrix = laplacian + float(epsilon) * sp.eye(num_nodes, format="csc")

        try:
            factor = cholesky(matrix)
        except Exception as exc:  # pragma: no cover - CHOLMOD exception type is environment-specific.
            raise RuntimeError(
                "CHOLMOD failed to factor the graph matrix. "
                "Check that the graph is connected and edge weights are positive."
            ) from exc

        solve_chunk_size = max(1, min(int(backward_chunk_size), num_edges))
        needs_backward = edge_weight.requires_grad
        potentials_np = (
            np.empty((solve_rows, num_edges), dtype=np.float64)
            if needs_backward
            else None
        )
        resistance_np = np.empty(num_edges, dtype=np.float64)
        for start in range(0, num_edges, solve_chunk_size):
            end = min(start + solve_chunk_size, num_edges)
            rhs = _edge_rhs_chunk(solve_rows, edge_u_rows_np, edge_v_rows_np, start, end)
            solved = factor(rhs)
            if potentials_np is not None:
                potentials_np[:, start:end] = solved
            resistance_np[start:end] = np.sum(rhs * solved, axis=0)

        if needs_backward:
            edge_u_rows = torch.as_tensor(edge_u_rows_np, device=device, dtype=torch.long)
            edge_v_rows = torch.as_tensor(edge_v_rows_np, device=device, dtype=torch.long)
            potentials = torch.as_tensor(potentials_np, device=device, dtype=dtype)
            ctx.save_for_backward(potentials, edge_u_rows, edge_v_rows)
            ctx.backward_chunk_size = max(1, int(backward_chunk_size))

        return torch.as_tensor(resistance_np, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        potentials, edge_u_rows, edge_v_rows = ctx.saved_tensors
        chunk_size = ctx.backward_chunk_size

        num_edges = int(edge_u_rows.numel())
        grad_edge_weight = grad_output.new_zeros(num_edges)
        if num_edges == 0 or not ctx.needs_input_grad[0]:
            return grad_edge_weight, None, None, None, None, None, None

        grad_output = grad_output.contiguous()
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            potential_u = _gather_solution_rows(potentials, edge_u_rows[start:end])
            potential_v = _gather_solution_rows(potentials, edge_v_rows[start:end])
            transfer = potential_u - potential_v
            grad_edge_weight -= (
                transfer.square() * grad_output[start:end].unsqueeze(1)
            ).sum(dim=0)

        return grad_edge_weight, None, None, None, None, None, None


class _MultiEdgeEffectiveResistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        edge_weight: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        root: int,
        inv_method: str,
        epsilon: float,
        backward_chunk_size: int,
    ) -> torch.Tensor:
        if edge_weight.dim() != 2:
            raise ValueError("edge_weight must have shape (num_heads, num_edges).")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape (2, num_edges).")
        if edge_index.size(1) != edge_weight.size(1):
            raise ValueError("edge_weight and edge_index disagree on num_edges.")
        if not edge_weight.is_floating_point():
            raise TypeError("edge_weight must be a floating point tensor.")
        if root < 0 or root >= num_nodes:
            raise ValueError(f"root must be in [0, {num_nodes}), got {root}.")
        _validate_inv_method(inv_method, epsilon)

        device = edge_weight.device
        dtype = edge_weight.dtype
        num_heads = int(edge_weight.size(0))
        num_edges = int(edge_weight.size(1))
        solve_rows = _solve_rows(num_nodes, inv_method)

        if num_edges == 0:
            ctx.save_for_backward(
                torch.empty((num_heads, solve_rows, 0), device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
            ctx.backward_chunk_size = max(1, int(backward_chunk_size))
            return edge_weight.new_zeros((num_heads, 0))

        conductance_np = edge_weight.detach().cpu().to(torch.float64).numpy()
        if np.any(~np.isfinite(conductance_np)) or np.any(conductance_np <= 0.0):
            raise ValueError("All edge conductances must be positive.")

        edge_index_np = edge_index.detach().cpu().to(torch.long).numpy()
        edge_u_rows_np, edge_v_rows_np = _edge_solver_rows(
            num_nodes, edge_index_np, root, inv_method
        )

        solve_chunk_size = max(1, min(int(backward_chunk_size), num_edges))
        needs_backward = edge_weight.requires_grad
        potentials_np = (
            np.empty((num_heads, solve_rows, num_edges), dtype=np.float64)
            if needs_backward
            else None
        )
        resistance_np = np.empty((num_heads, num_edges), dtype=np.float64)

        for head in range(num_heads):
            laplacian = _laplacian_from_edges(
                num_nodes,
                edge_index_np,
                conductance_np[head],
            )
            if inv_method == "L+J":
                matrix = _remove_root(laplacian, root)
            else:
                matrix = laplacian + float(epsilon) * sp.eye(num_nodes, format="csc")

            try:
                factor = cholesky(matrix)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "CHOLMOD failed to factor a head graph matrix. "
                    "Check that every head graph is connected and weights are positive."
                ) from exc

            for start in range(0, num_edges, solve_chunk_size):
                end = min(start + solve_chunk_size, num_edges)
                rhs = _edge_rhs_chunk(
                    solve_rows,
                    edge_u_rows_np,
                    edge_v_rows_np,
                    start,
                    end,
                )
                solved = factor(rhs)
                if potentials_np is not None:
                    potentials_np[head, :, start:end] = solved
                resistance_np[head, start:end] = np.sum(rhs * solved, axis=0)

        if needs_backward:
            edge_u_rows = torch.as_tensor(edge_u_rows_np, device=device, dtype=torch.long)
            edge_v_rows = torch.as_tensor(edge_v_rows_np, device=device, dtype=torch.long)
            potentials = torch.as_tensor(potentials_np, device=device, dtype=dtype)
            ctx.save_for_backward(potentials, edge_u_rows, edge_v_rows)
            ctx.backward_chunk_size = max(1, int(backward_chunk_size))

        return torch.as_tensor(resistance_np, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        potentials, edge_u_rows, edge_v_rows = ctx.saved_tensors
        chunk_size = ctx.backward_chunk_size

        num_heads = int(potentials.size(0))
        num_edges = int(edge_u_rows.numel())
        grad_edge_weight = grad_output.new_zeros((num_heads, num_edges))
        if num_edges == 0 or not ctx.needs_input_grad[0]:
            return grad_edge_weight, None, None, None, None, None, None

        grad_output = grad_output.contiguous()
        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            potential_u = _gather_solution_rows_batched(
                potentials,
                edge_u_rows[start:end],
            )
            potential_v = _gather_solution_rows_batched(
                potentials,
                edge_v_rows[start:end],
            )
            transfer = potential_u - potential_v
            grad_edge_weight -= (
                transfer.square() * grad_output[:, start:end].unsqueeze(2)
            ).sum(dim=1)

        return grad_edge_weight, None, None, None, None, None, None


def edge_effective_resistance(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute effective resistance for unique edges.

    Args:
        edge_index: Long tensor with shape ``(2, m)``.
        edge_weight: Conductance tensor with shape ``(m,)``.
        num_nodes: Number of graph nodes.
        inv_method: ``"L+J"`` uses the reduced Laplacian; ``"L+eI"`` uses
            ``L + epsilon * I``.
        epsilon: Diagonal regularization for ``inv_method="L+eI"``.
    """
    return _EdgeEffectiveResistanceFunction.apply(
        edge_weight,
        edge_index,
        int(num_nodes),
        int(root),
        inv_method,
        float(epsilon),
        int(backward_chunk_size),
    )


def multi_edge_effective_resistance(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute effective resistance for multiple head graphs.

    Args:
        edge_index: Shared long tensor with shape ``(2, m)``.
        edge_weight: Conductance tensor with shape ``(num_heads, m)``.
        num_nodes: Number of graph nodes.

    Returns:
        Tensor with shape ``(num_heads, m)``.
    """
    return _MultiEdgeEffectiveResistanceFunction.apply(
        edge_weight,
        edge_index,
        int(num_nodes),
        int(root),
        inv_method,
        float(epsilon),
        int(backward_chunk_size),
    )


class DenseEffectiveResistance(torch.nn.Module):
    """Dense autograd reference that takes an ``(N, N)`` adjacency matrix.

    The output has shape ``(N, N)``.  By default only nonzero off-diagonal
    adjacency positions are evaluated and other positions are zero.  The
    Laplacian is built from ``0.5 * (adjacency + adjacency.T)`` by default, so
    symmetric dense adjacency matrices match the undirected ``(N, k)`` path.
    """

    def __init__(
        self,
        *,
        root: int = 0,
        inv_method: str = "L+J",
        epsilon: float = 0.2,
        edge_mask: torch.Tensor | None = None,
        symmetrize: bool = True,
    ) -> None:
        super().__init__()
        _validate_inv_method(inv_method, epsilon)

        self.root = int(root)
        self.inv_method = inv_method
        self.epsilon = float(epsilon)
        self.symmetrize = bool(symmetrize)

        if edge_mask is not None:
            if edge_mask.dim() != 2 or edge_mask.size(0) != edge_mask.size(1):
                raise ValueError("edge_mask must have shape (N, N).")
            self.register_buffer(
                "edge_mask",
                edge_mask.detach().to(dtype=torch.bool),
                persistent=False,
            )
        else:
            self.edge_mask = None

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must have shape (N, N).")
        if not adjacency.is_floating_point():
            raise TypeError("adjacency must be a floating point tensor.")

        num_nodes = int(adjacency.size(0))
        if self.root < 0 or self.root >= num_nodes:
            raise ValueError(f"root must be in [0, {num_nodes}), got {self.root}.")

        off_diag = ~torch.eye(num_nodes, device=adjacency.device, dtype=torch.bool)
        rows, cols = self._edge_positions(adjacency, off_diag)
        output = adjacency.new_zeros((num_nodes, num_nodes))
        if rows.numel() == 0:
            return output

        laplacian = self._laplacian(adjacency, off_diag)
        matrix, rhs = self._linear_system(laplacian, rows, cols, adjacency)

        solved = torch.linalg.solve(matrix, rhs)
        output[rows, cols] = (rhs * solved).sum(dim=0)
        return output

    def _edge_positions(
        self,
        adjacency: torch.Tensor,
        off_diag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.edge_mask is None:
            edge_mask = (adjacency.detach() != 0) & off_diag
        else:
            edge_mask = self.edge_mask.to(device=adjacency.device) & off_diag
        return edge_mask.nonzero(as_tuple=True)

    def _laplacian(self, adjacency: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
        conductance = 0.5 * (adjacency + adjacency.t()) if self.symmetrize else adjacency
        conductance = conductance * off_diag.to(dtype=adjacency.dtype)
        degree = conductance.sum(dim=1)
        return torch.diag(degree) - conductance

    def _linear_system(
        self,
        laplacian: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inv_method == "L+J":
            return self._reduced_linear_system(laplacian, rows, cols, adjacency)
        return self._regularized_linear_system(laplacian, rows, cols, adjacency)

    def _reduced_linear_system(
        self,
        laplacian: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = int(adjacency.size(0))
        keep = torch.ones(num_nodes, device=adjacency.device, dtype=torch.bool)
        keep[self.root] = False
        matrix = laplacian[keep][:, keep]

        rhs = adjacency.new_zeros((num_nodes - 1, rows.numel()))
        col_idx = torch.arange(rows.numel(), device=adjacency.device)
        row_mask = rows != self.root
        col_mask = cols != self.root
        reduced_rows = rows[row_mask] - (rows[row_mask] > self.root).to(rows.dtype)
        reduced_cols = cols[col_mask] - (cols[col_mask] > self.root).to(cols.dtype)
        rhs[reduced_rows, col_idx[row_mask]] += 1.0
        rhs[reduced_cols, col_idx[col_mask]] -= 1.0
        return matrix, rhs

    def _regularized_linear_system(
        self,
        laplacian: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = int(adjacency.size(0))
        matrix = laplacian + self.epsilon * torch.eye(
            num_nodes,
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        rhs = adjacency.new_zeros((num_nodes, rows.numel()))
        col_idx = torch.arange(rows.numel(), device=adjacency.device)
        rhs[rows, col_idx] += 1.0
        rhs[cols, col_idx] -= 1.0
        return matrix, rhs


def effective_resistance(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    weight: torch.Tensor,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    average_duplicates: bool = True,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return effective resistance in the same ``(N, k)`` shape as ``weight``.

    ``inv_method`` follows ``GEM_module_sparse_new.py``: ``"L+J"`` factors a
    reduced Laplacian, while ``"L+eI"`` factors ``L + epsilon * I``.
    """
    edge_weight, edge_index, mapping = neighbor_weights_to_edge_list(
        neighbor_list,
        weight,
        average_duplicates=average_duplicates,
        return_mapping=True,
    )
    edge_resistance = edge_effective_resistance(
        edge_index,
        edge_weight,
        int(mapping.original_shape[0]),
        root=root,
        inv_method=inv_method,
        epsilon=epsilon,
        backward_chunk_size=backward_chunk_size,
    )
    return edge_values_to_neighbor_shape(edge_resistance, mapping)


def multi_graph_effective_resistance(
    neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
    weight: torch.Tensor,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    average_duplicates: bool = True,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return multi-head ER in the same ``(num_heads, N, k)`` shape as ``weight``."""
    mapping = build_neighbor_to_edge_map(
        neighbor_list,
        average_duplicates=average_duplicates,
    )
    edge_weight = _aggregate_multi_edge_weight(
        weight,
        original_shape=mapping.original_shape,
        valid_flat_index=mapping.valid_flat_index,
        entry_to_edge=mapping.entry_to_edge,
        entry_alpha=mapping.entry_alpha,
        num_edges=int(mapping.edge_index.size(1)),
    )
    edge_resistance = multi_edge_effective_resistance(
        mapping.edge_index.to(device=weight.device),
        edge_weight,
        int(mapping.original_shape[0]),
        root=root,
        inv_method=inv_method,
        epsilon=epsilon,
        backward_chunk_size=backward_chunk_size,
    )
    return _scatter_multi_edge_values(
        edge_resistance,
        original_shape=mapping.original_shape,
        valid_flat_index=mapping.valid_flat_index,
        entry_to_edge=mapping.entry_to_edge,
        num_edges=int(mapping.edge_index.size(1)),
    )


class EffectiveResistance(torch.nn.Module):
    """Reusable module for a fixed GEM ``neighbor_list``."""

    def __init__(
        self,
        neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
        *,
        root: int = 0,
        inv_method: str = "L+J",
        epsilon: float = 0.2,
        average_duplicates: bool = True,
        backward_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        mapping = build_neighbor_to_edge_map(
            neighbor_list,
            average_duplicates=average_duplicates,
        )

        self.num_nodes = int(mapping.original_shape[0])
        self.root = int(root)
        self.inv_method = inv_method
        self.epsilon = float(epsilon)
        self.backward_chunk_size = int(backward_chunk_size)
        self.average_duplicates = bool(average_duplicates)
        self.original_shape = mapping.original_shape

        _validate_inv_method(self.inv_method, self.epsilon)

        self.register_buffer("valid_flat_index", mapping.valid_flat_index, persistent=False)
        self.register_buffer("entry_to_edge", mapping.entry_to_edge, persistent=False)
        self.register_buffer("entry_alpha", mapping.entry_alpha, persistent=False)
        self.register_buffer("edge_index", mapping.edge_index, persistent=False)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        edge_weight = _aggregate_edge_weight(
            weight,
            original_shape=self.original_shape,
            valid_flat_index=self.valid_flat_index,
            entry_to_edge=self.entry_to_edge,
            entry_alpha=self.entry_alpha,
            num_edges=int(self.edge_index.size(1)),
        )
        edge_resistance = edge_effective_resistance(
            self.edge_index.to(device=weight.device),
            edge_weight,
            self.num_nodes,
            root=self.root,
            inv_method=self.inv_method,
            epsilon=self.epsilon,
            backward_chunk_size=self.backward_chunk_size,
        )
        return _scatter_edge_values(
            edge_resistance,
            original_shape=self.original_shape,
            valid_flat_index=self.valid_flat_index,
            entry_to_edge=self.entry_to_edge,
            num_edges=int(self.edge_index.size(1)),
        )


class MultiGraphEffectiveResistance(torch.nn.Module):
    """Reusable ER module for ``num_heads`` graphs sharing one ``neighbor_list``.

    ``forward`` expects weights with shape ``(num_heads, N, k)`` and returns
    effective resistances with the same shape.  Each head has its own edge
    weights and CHOLMOD factorization; the neighbor-to-edge mapping is shared.
    """

    def __init__(
        self,
        neighbor_list: torch.Tensor | np.ndarray | Iterable[Iterable[int]],
        *,
        root: int = 0,
        inv_method: str = "L+J",
        epsilon: float = 0.2,
        average_duplicates: bool = True,
        backward_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        mapping = build_neighbor_to_edge_map(
            neighbor_list,
            average_duplicates=average_duplicates,
        )

        self.num_nodes = int(mapping.original_shape[0])
        self.root = int(root)
        self.inv_method = inv_method
        self.epsilon = float(epsilon)
        self.backward_chunk_size = int(backward_chunk_size)
        self.average_duplicates = bool(average_duplicates)
        self.original_shape = mapping.original_shape

        _validate_inv_method(self.inv_method, self.epsilon)

        self.register_buffer("valid_flat_index", mapping.valid_flat_index, persistent=False)
        self.register_buffer("entry_to_edge", mapping.entry_to_edge, persistent=False)
        self.register_buffer("entry_alpha", mapping.entry_alpha, persistent=False)
        self.register_buffer("edge_index", mapping.edge_index, persistent=False)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        edge_weight = _aggregate_multi_edge_weight(
            weight,
            original_shape=self.original_shape,
            valid_flat_index=self.valid_flat_index,
            entry_to_edge=self.entry_to_edge,
            entry_alpha=self.entry_alpha,
            num_edges=int(self.edge_index.size(1)),
        )
        edge_resistance = multi_edge_effective_resistance(
            self.edge_index.to(device=weight.device),
            edge_weight,
            self.num_nodes,
            root=self.root,
            inv_method=self.inv_method,
            epsilon=self.epsilon,
            backward_chunk_size=self.backward_chunk_size,
        )
        return _scatter_multi_edge_values(
            edge_resistance,
            original_shape=self.original_shape,
            valid_flat_index=self.valid_flat_index,
            entry_to_edge=self.entry_to_edge,
            num_edges=int(self.edge_index.size(1)),
        )


def _demo(device: torch.device) -> None:
    neighbor_list = torch.tensor(
        [
            [1, 2, -1, -1],
            [0, 2, 3, -1],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [2, 3, 5, -1],
            [3, 4, -1, -1],
        ],
        dtype=torch.long,
    )
    weight = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 1.5, 2.5, 0.0],
            [2.0, 1.5, 1.2, 0.8],
            [2.5, 1.2, 1.7, 2.2],
            [0.8, 1.7, 1.1, 0.0],
            [2.2, 1.1, 0.0, 0.0],
        ],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )

    print(f"device: {device}")
    for inv_method in ("L+J", "L+eI"):
        sparse_weight = weight.detach().clone().requires_grad_(True)
        dense_adjacency = neighbor_weights_to_dense_adjacency(
            neighbor_list,
            weight.detach(),
        )
        dense_adjacency.requires_grad_(True)

        edge_weight, edge_index, mapping = neighbor_weights_to_edge_list(
            neighbor_list,
            sparse_weight,
            return_mapping=True,
        )
        edge_resistance = edge_effective_resistance(
            edge_index,
            edge_weight,
            num_nodes=neighbor_list.size(0),
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )
        sparse_resistance = edge_values_to_neighbor_shape(edge_resistance, mapping)
        sparse_resistance.sum().backward()

        dense_module = DenseEffectiveResistance(
            inv_method=inv_method,
            epsilon=0.2,
        )
        dense_resistance_matrix = dense_module(dense_adjacency)
        dense_resistance_matrix.sum().backward()

        dense_resistance = dense_matrix_to_neighbor_shape(
            neighbor_list,
            dense_resistance_matrix,
        )
        dense_grad = dense_matrix_to_neighbor_shape(neighbor_list, dense_adjacency.grad)

        forward_diff = (sparse_resistance - dense_resistance).abs().max()
        backward_diff = (sparse_weight.grad - dense_grad).abs().max()

        print(f"\ninv_method: {inv_method}")
        print("edge_index:")
        print(edge_index.detach().cpu())
        print("edge_weight:")
        print(edge_weight.detach().cpu())
        print("dense adjacency:")
        print(dense_adjacency.detach().cpu())
        print("sparse effective resistance in (N, k):")
        print(sparse_resistance.detach().cpu())
        print("dense adjacency effective resistance gathered to (N, k):")
        print(dense_resistance.detach().cpu())
        print("sparse weight.grad in (N, k):")
        print(sparse_weight.grad.detach().cpu())
        print("dense adjacency.grad gathered to (N, k):")
        print(dense_grad.detach().cpu())
        print(f"max forward diff: {forward_diff.item():.3e}")
        print(f"max backward diff: {backward_diff.item():.3e}")

    multi_weight = torch.stack(
        (
            weight.detach(),
            weight.detach() * 1.25,
            weight.detach() * 0.75 + (weight.detach() > 0).to(weight.dtype) * 0.1,
        ),
        dim=0,
    ).requires_grad_(True)

    for inv_method in ("L+J", "L+eI"):
        multi_module = MultiGraphEffectiveResistance(
            neighbor_list,
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )
        single_module = EffectiveResistance(
            neighbor_list,
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )

        multi_input = multi_weight.detach().clone().requires_grad_(True)
        single_input = multi_weight.detach().clone().requires_grad_(True)

        multi_resistance = multi_module(multi_input)
        multi_resistance.sum().backward()

        single_resistance = torch.stack(
            [single_module(single_input[head]) for head in range(single_input.size(0))],
            dim=0,
        )
        single_resistance.sum().backward()

        print(f"\nmulti-head inv_method: {inv_method}")
        print("multi effective resistance shape:", tuple(multi_resistance.shape))
        print(
            "max multi-vs-single forward diff:",
            f"{(multi_resistance - single_resistance).abs().max().item():.3e}",
        )
        print(
            "max multi-vs-single backward diff:",
            f"{(multi_input.grad - single_input.grad).abs().max().item():.3e}",
        )


if __name__ == "__main__":
    _demo(torch.device("cpu"))
    if torch.cuda.is_available():
        _demo(torch.device("cuda:1"))
    else:
        print("CUDA is not available; skipped CUDA demo.")
