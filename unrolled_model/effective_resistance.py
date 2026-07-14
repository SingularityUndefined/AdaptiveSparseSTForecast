"""Exact effective resistance for GEM-style sparse neighbor weights.

``GEM_sparse/GEM_module_sparse_new.py`` stores sparse graph structure as
``neighbor_list`` and ``neighbor_mask`` tensors with shape ``(N, k)`` or
``(num_graphs, N, k)`` and multi-graph ``weight`` tensors with shape
``(num_graphs, N, k)``:

* ``neighbor_list[i, j]`` is the neighbor node id of node ``i``.
* ``neighbor_list[g, i, j]`` is the graph-specific neighbor node id when the
  neighbor list is not shared.
* ``neighbor_list[..., i, j] == -1`` can be used for structurally impossible
  neighbor slots.
* ``neighbor_mask[i, j]`` marks whether the entry is a real edge.
* ``weight[g, i, j]`` is the corresponding edge conductance for graph ``g`` and
  is masked to zero where ``neighbor_mask`` is false.

This module first converts that representation to a unique undirected edge
list:

    edge_weight: (num_graphs, m)
    edge_index:  (2, m)

When ``shared_neighbor_list=False``, each graph is converted with its own edge
count ``m_g``.

The effective resistance is computed on the unique edge list with CHOLMOD in
the forward pass, then scattered back to the original ``(num_graphs, N, k)``
shape.  The manual backward is implemented at the edge-list level; PyTorch
handles the chain rule from ``edge_weight`` back to the original ``weight``
tensor.

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
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
from sksparse.cholmod import cholesky

try:
    from utils_modules import _validate_undirected_edge_pairs
except ModuleNotFoundError:  # pragma: no cover
    from .utils_modules import _validate_undirected_edge_pairs


_INV_METHODS = ("L+J", "L+eI")


@dataclass(frozen=True)
class NeighborToEdgeMap:
    """Mapping between GEM ``(N, k)`` entries and unique undirected edges."""

    original_shape: Tuple[int, int]
    valid_flat_index: torch.Tensor
    entry_to_edge: torch.Tensor
    entry_alpha: torch.Tensor
    edge_index: torch.Tensor


def _mapping_from_buffers(module: torch.nn.Module, prefix: str) -> NeighborToEdgeMap:
    return NeighborToEdgeMap(
        original_shape=getattr(module, f"{prefix}_original_shape"),
        valid_flat_index=getattr(module, f"{prefix}_valid_flat_index"),
        entry_to_edge=getattr(module, f"{prefix}_entry_to_edge"),
        entry_alpha=getattr(module, f"{prefix}_entry_alpha"),
        edge_index=getattr(module, f"{prefix}_edge_index"),
    )


def _register_mapping_buffers(
    module: torch.nn.Module,
    prefix: str,
    mapping: NeighborToEdgeMap,
) -> None:
    setattr(module, f"{prefix}_original_shape", mapping.original_shape)
    module.register_buffer(
        f"{prefix}_valid_flat_index",
        mapping.valid_flat_index,
        persistent=False,
    )
    module.register_buffer(
        f"{prefix}_entry_to_edge",
        mapping.entry_to_edge,
        persistent=False,
    )
    module.register_buffer(
        f"{prefix}_entry_alpha",
        mapping.entry_alpha,
        persistent=False,
    )
    module.register_buffer(
        f"{prefix}_edge_index",
        mapping.edge_index,
        persistent=False,
    )


NeighborInput = Union[torch.Tensor, np.ndarray, Iterable[Iterable[int]]]
MultiNeighborInput = Union[torch.Tensor, np.ndarray, Iterable[Iterable[Iterable[int]]]]
MaskInput = Union[torch.Tensor, np.ndarray, Iterable[Iterable[bool]]]
MultiMaskInput = Union[torch.Tensor, np.ndarray, Iterable[Iterable[Iterable[bool]]]]


def _as_long_tensor(x: NeighborInput) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.long, device="cpu")
    return torch.as_tensor(x, dtype=torch.long, device="cpu")


def _as_bool_tensor(x: MaskInput) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.bool, device="cpu")
    return torch.as_tensor(x, dtype=torch.bool, device="cpu")


def _validate_inv_method(inv_method: str, epsilon: float) -> None:
    if inv_method not in _INV_METHODS:
        raise ValueError("inv_method must be 'L+J' or 'L+eI'.")
    if inv_method == "L+eI" and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when inv_method='L+eI'.")


def _solve_rows(num_nodes: int, inv_method: str) -> int:
    return num_nodes - 1 if inv_method == "L+J" else num_nodes


def build_neighbor_to_edge_map(
    neighbor_list: NeighborInput,
    neighbor_mask: MaskInput,
    *,
    average_duplicates: bool = True,
) -> NeighborToEdgeMap:
    """Build the fixed mapping from GEM neighbor entries to an edge list.

    Repeated entries of the same undirected edge, such as ``u -> v`` and
    ``v -> u``, are mapped to the same edge id.  With ``average_duplicates=True``
    their weights are averaged into one conductance; this matches the common
    symmetric storage where both directions contain the same value.

    ``neighbor_mask`` is the only source of edge validity.  ``neighbor_list`` may
    contain ``-1`` in inactive slots, but every active slot must contain a valid
    node id.
    """
    neighbor_cpu = _as_long_tensor(neighbor_list)
    mask_cpu = _as_bool_tensor(neighbor_mask)
    if neighbor_cpu.dim() != 2:
        raise ValueError("neighbor_list must have shape (num_nodes, num_neighbors).")
    if mask_cpu.shape != neighbor_cpu.shape:
        raise ValueError("neighbor_mask must have the same shape as neighbor_list.")

    num_nodes, width = neighbor_cpu.shape
    if bool((neighbor_cpu < -1).any()):
        raise ValueError("neighbor_list entries must be -1 or valid node indices.")
    if bool((neighbor_cpu >= num_nodes).any()):
        bad_index = int(neighbor_cpu[neighbor_cpu >= num_nodes][0])
        raise ValueError(
            f"neighbor_list contains node index {bad_index}, but num_nodes={num_nodes}."
        )
    if bool((mask_cpu & (neighbor_cpu < 0)).any()):
        raise ValueError("neighbor_mask=True entries must have valid node indices.")

    flat_neighbor = neighbor_cpu.reshape(-1)
    flat_rows = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(width)
    valid = mask_cpu.reshape(-1) & (flat_neighbor != flat_rows)
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
        raise ValueError("multi-graph weight must have shape (num_graphs, N, k).")

    device = weight.device
    dtype = weight.dtype
    num_graphs = int(weight.size(0))
    valid_flat_index = valid_flat_index.to(device=device)
    entry_to_edge = entry_to_edge.to(device=device)
    entry_alpha = entry_alpha.to(device=device, dtype=dtype)

    edge_weight = weight.new_zeros((num_graphs, num_edges))
    if num_edges == 0:
        return edge_weight

    valid_weight = weight.reshape(num_graphs, -1).index_select(1, valid_flat_index)
    weighted_entries = valid_weight * entry_alpha.unsqueeze(0)
    edge_ids = entry_to_edge.unsqueeze(0).expand(num_graphs, -1)
    edge_weight.scatter_add_(1, edge_ids, weighted_entries)
    return edge_weight


def _scatter_multi_edge_values(
    edge_values: torch.Tensor,
    *,
    original_shape: Tuple[int, int],
    valid_flat_index: torch.Tensor,
    entry_to_edge: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    if edge_values.dim() != 2:
        raise ValueError(
            "multi-graph edge_values must have shape (num_graphs, num_edges)."
        )
    if edge_values.size(1) != num_edges:
        raise ValueError("edge_values length does not match the neighbor mapping.")

    num_graphs = int(edge_values.size(0))
    output = edge_values.new_zeros((num_graphs, *original_shape))
    if num_edges == 0:
        return output

    valid_flat_index = valid_flat_index.to(device=edge_values.device)
    entry_to_edge = entry_to_edge.to(device=edge_values.device)
    gathered = edge_values.index_select(1, entry_to_edge)
    output.reshape(num_graphs, -1).index_copy_(1, valid_flat_index, gathered)
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
            raise ValueError("edge_weight must have shape (num_graphs, num_edges).")
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
        num_graphs = int(edge_weight.size(0))
        num_edges = int(edge_weight.size(1))
        solve_rows = _solve_rows(num_nodes, inv_method)

        if num_edges == 0:
            ctx.save_for_backward(
                torch.empty((num_graphs, solve_rows, 0), device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
            )
            ctx.backward_chunk_size = max(1, int(backward_chunk_size))
            return edge_weight.new_zeros((num_graphs, 0))

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
            np.empty((num_graphs, solve_rows, num_edges), dtype=np.float64)
            if needs_backward
            else None
        )
        resistance_np = np.empty((num_graphs, num_edges), dtype=np.float64)

        for graph in range(num_graphs):
            laplacian = _laplacian_from_edges(
                num_nodes,
                edge_index_np,
                conductance_np[graph],
            )
            if inv_method == "L+J":
                matrix = _remove_root(laplacian, root)
            else:
                matrix = laplacian + float(epsilon) * sp.eye(num_nodes, format="csc")

            try:
                factor = cholesky(matrix)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "CHOLMOD failed to factor a graph matrix. "
                    "Check that every graph is connected and weights are positive."
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
                    potentials_np[graph, :, start:end] = solved
                resistance_np[graph, start:end] = np.sum(rhs * solved, axis=0)

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

        num_graphs = int(potentials.size(0))
        num_edges = int(edge_u_rows.numel())
        grad_edge_weight = grad_output.new_zeros((num_graphs, num_edges))
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
    """Compute effective resistance for multiple graphs.

    Args:
        edge_index: Shared long tensor with shape ``(2, m)``.
        edge_weight: Conductance tensor with shape ``(num_graphs, m)``.
        num_nodes: Number of graph nodes.

    Returns:
        Tensor with shape ``(num_graphs, m)``.
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


def multi_graph_effective_resistance(
    neighbor_list: NeighborInput,
    neighbor_mask: MaskInput,
    weight: torch.Tensor,
    *,
    shared_neighbor_list: bool = True,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    average_duplicates: bool = True,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    """Return multi-graph ER in the same ``(num_graphs, N, k)`` shape as ``weight``.

    With ``shared_neighbor_list=True``, ``neighbor_list`` has shape ``(N, k)``
    and ``neighbor_mask`` has shape ``(N, k)``.  They are reused by every graph.
    With ``shared_neighbor_list=False``, both tensors must have shape
    ``(num_graphs, N, k)`` and each graph uses its own sparse structure.
    """
    if not shared_neighbor_list:
        return _multi_graph_effective_resistance_unshared(
            neighbor_list,
            neighbor_mask,
            weight,
            root=root,
            inv_method=inv_method,
            epsilon=epsilon,
            average_duplicates=average_duplicates,
            backward_chunk_size=backward_chunk_size,
        )

    mapping = build_neighbor_to_edge_map(
        neighbor_list,
        neighbor_mask,
        average_duplicates=average_duplicates,
    )
    return _multi_graph_effective_resistance_with_mapping(
        mapping,
        weight,
        root=root,
        inv_method=inv_method,
        epsilon=epsilon,
        backward_chunk_size=backward_chunk_size,
    )


def _multi_graph_effective_resistance_with_mapping(
    mapping: NeighborToEdgeMap,
    weight: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
    backward_chunk_size: int,
) -> torch.Tensor:
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


def _multi_graph_effective_resistance_unshared(
    neighbor_list: MultiNeighborInput,
    neighbor_mask: MultiMaskInput,
    weight: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
    average_duplicates: bool,
    backward_chunk_size: int,
) -> torch.Tensor:
    neighbor_cpu = _as_long_tensor(neighbor_list)
    mask_cpu = _as_bool_tensor(neighbor_mask)
    if neighbor_cpu.dim() != 3:
        raise ValueError(
            "neighbor_list must have shape (num_graphs, N, k) when "
            "shared_neighbor_list=False."
        )
    if tuple(mask_cpu.shape) != tuple(neighbor_cpu.shape):
        raise ValueError(
            "neighbor_mask must have the same shape as neighbor_list when "
            "shared_neighbor_list=False."
        )
    if weight.dim() != 3 or tuple(weight.shape) != tuple(neighbor_cpu.shape):
        raise ValueError(
            "weight and neighbor_list must both have shape (num_graphs, N, k) "
            "when shared_neighbor_list=False."
        )

    num_graphs = int(weight.size(0))
    if num_graphs == 0:
        return weight.new_zeros(weight.shape)

    outputs = []
    for graph in range(num_graphs):
        mapping = build_neighbor_to_edge_map(
            neighbor_cpu[graph],
            mask_cpu[graph],
            average_duplicates=average_duplicates,
        )
        graph_resistance = _multi_graph_effective_resistance_with_mapping(
            mapping,
            weight[graph : graph + 1],
            root=root,
            inv_method=inv_method,
            epsilon=epsilon,
            backward_chunk_size=backward_chunk_size,
        )
        outputs.append(graph_resistance.squeeze(0))

    return torch.stack(outputs, dim=0)


def _validate_multi_graph_undirected_edge_pairs(
    neighbor_list: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> None:
    if neighbor_list.dim() == 2:
        _validate_undirected_edge_pairs(neighbor_list, neighbor_mask)
        return

    for graph in range(int(neighbor_mask.size(0))):
        _validate_undirected_edge_pairs(
            neighbor_list[graph],
            neighbor_mask[graph : graph + 1],
        )


def _multi_graph_effective_resistance_dynamic_mask(
    neighbor_list: Union[NeighborInput, MultiNeighborInput],
    neighbor_mask: MultiMaskInput,
    weight: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
    average_duplicates: bool,
    backward_chunk_size: int,
) -> torch.Tensor:
    neighbor_cpu = _as_long_tensor(neighbor_list)
    mask_cpu = _as_bool_tensor(neighbor_mask)
    if mask_cpu.dim() != 3:
        raise ValueError("neighbor_mask must have shape (num_graphs, N, k).")
    if weight.dim() != 3 or tuple(weight.shape) != tuple(mask_cpu.shape):
        raise ValueError(
            "weight and neighbor_mask must both have shape (num_graphs, N, k)."
        )

    if neighbor_cpu.dim() == 2:
        if tuple(mask_cpu.shape[1:]) != tuple(neighbor_cpu.shape):
            raise ValueError(
                "neighbor_mask must have shape (num_graphs, N, k), where "
                "neighbor_list has shape (N, k)."
            )
    elif neighbor_cpu.dim() == 3:
        if tuple(neighbor_cpu.shape) != tuple(mask_cpu.shape):
            raise ValueError(
                "neighbor_list and neighbor_mask must have the same shape when "
                "neighbor_list is graph-specific."
            )
    else:
        raise ValueError(
            "neighbor_list must have shape (N, k) or (num_graphs, N, k)."
        )

    _validate_multi_graph_undirected_edge_pairs(neighbor_cpu, mask_cpu)

    num_graphs = int(weight.size(0))
    if num_graphs == 0:
        return weight.new_zeros(weight.shape)

    outputs = []
    for graph in range(num_graphs):
        graph_neighbor_list = (
            neighbor_cpu if neighbor_cpu.dim() == 2 else neighbor_cpu[graph]
        )
        mapping = build_neighbor_to_edge_map(
            graph_neighbor_list,
            mask_cpu[graph],
            average_duplicates=average_duplicates,
        )
        graph_resistance = _multi_graph_effective_resistance_with_mapping(
            mapping,
            weight[graph : graph + 1],
            root=root,
            inv_method=inv_method,
            epsilon=epsilon,
            backward_chunk_size=backward_chunk_size,
        )
        outputs.append(graph_resistance.squeeze(0))

    return torch.stack(outputs, dim=0)


class MultiGraphEffectiveResistance(torch.nn.Module):
    """Reusable ER module for multiple graphs.

    ``forward`` expects weights with shape ``(num_graphs, N, k)`` and returns
    effective resistances with the same shape.  Pass ``neighbor_mask`` to
    ``forward`` when the active sparse structure changes between calls.  Dynamic
    masks must have shape ``(num_graphs, N, k)``.  ``neighbor_list`` can be shared
    with shape ``(N, k)`` or graph-specific with shape ``(num_graphs, N, k)``.
    """

    def __init__(
        self,
        neighbor_list: Union[NeighborInput, MultiNeighborInput],
        neighbor_mask: Optional[Union[MaskInput, MultiMaskInput]] = None,
        *,
        shared_neighbor_list: bool = True,
        root: int = 0,
        inv_method: str = "L+J",
        epsilon: float = 0.2,
        average_duplicates: bool = True,
        backward_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.shared_neighbor_list = bool(shared_neighbor_list)
        self.root = int(root)
        self.inv_method = inv_method
        self.epsilon = float(epsilon)
        self.backward_chunk_size = int(backward_chunk_size)
        self.average_duplicates = bool(average_duplicates)

        _validate_inv_method(self.inv_method, self.epsilon)

        neighbor_cpu = _as_long_tensor(neighbor_list)
        self.register_buffer("neighbor_list", neighbor_cpu, persistent=False)
        self.register_buffer("neighbor_mask", None, persistent=False)
        self._has_static_mapping = False

        if neighbor_mask is None:
            if neighbor_cpu.dim() == 2:
                self.num_graphs = None
                self.num_nodes = int(neighbor_cpu.size(0))
                self.original_shape = (
                    int(neighbor_cpu.size(0)),
                    int(neighbor_cpu.size(1)),
                )
            elif neighbor_cpu.dim() == 3:
                self.num_graphs = int(neighbor_cpu.size(0))
                self.num_nodes = int(neighbor_cpu.size(1))
                self.original_shape = (
                    int(neighbor_cpu.size(1)),
                    int(neighbor_cpu.size(2)),
                )
            else:
                raise ValueError(
                    "neighbor_list must have shape (N, k) or (num_graphs, N, k)."
                )
            return

        mask_cpu = _as_bool_tensor(neighbor_mask)
        self.neighbor_mask = mask_cpu
        if mask_cpu.dim() == 3:
            if neighbor_cpu.dim() == 2:
                if tuple(mask_cpu.shape[1:]) != tuple(neighbor_cpu.shape):
                    raise ValueError(
                        "neighbor_mask must have shape (num_graphs, N, k), where "
                        "neighbor_list has shape (N, k)."
                    )
            elif neighbor_cpu.dim() == 3:
                if tuple(mask_cpu.shape) != tuple(neighbor_cpu.shape):
                    raise ValueError(
                        "neighbor_list and neighbor_mask must have the same shape when "
                        "neighbor_list is graph-specific."
                    )
            else:
                raise ValueError(
                    "neighbor_list must have shape (N, k) or (num_graphs, N, k)."
                )
            self.num_graphs = int(mask_cpu.size(0))
            self.num_nodes = int(mask_cpu.size(1))
            self.original_shape = (int(mask_cpu.size(1)), int(mask_cpu.size(2)))
            return

        if self.shared_neighbor_list:
            mapping = build_neighbor_to_edge_map(
                neighbor_cpu,
                mask_cpu,
                average_duplicates=self.average_duplicates,
            )
            self.num_graphs = None
            self.num_nodes = int(mapping.original_shape[0])
            self.original_shape = mapping.original_shape
            _register_mapping_buffers(self, "shared", mapping)
            self._has_static_mapping = True
        else:
            if neighbor_cpu.dim() != 3:
                raise ValueError(
                    "neighbor_list must have shape (num_graphs, N, k) when "
                    "shared_neighbor_list=False."
                )
            if tuple(mask_cpu.shape) != tuple(neighbor_cpu.shape):
                raise ValueError(
                    "neighbor_mask must have the same shape as neighbor_list when "
                    "shared_neighbor_list=False."
                )
            self.num_graphs = int(neighbor_cpu.size(0))
            self.num_nodes = int(neighbor_cpu.size(1))
            self.original_shape = (int(neighbor_cpu.size(1)), int(neighbor_cpu.size(2)))
            for graph in range(self.num_graphs):
                mapping = build_neighbor_to_edge_map(
                    neighbor_cpu[graph],
                    mask_cpu[graph],
                    average_duplicates=self.average_duplicates,
                )
                _register_mapping_buffers(self, f"graph_{graph}", mapping)
            self._has_static_mapping = True

    def forward(
        self,
        weight: torch.Tensor,
        neighbor_mask: Optional[MultiMaskInput] = None,
    ) -> torch.Tensor:
        if neighbor_mask is not None:
            return _multi_graph_effective_resistance_dynamic_mask(
                self.neighbor_list,
                neighbor_mask,
                weight,
                root=self.root,
                inv_method=self.inv_method,
                epsilon=self.epsilon,
                average_duplicates=self.average_duplicates,
                backward_chunk_size=self.backward_chunk_size,
            )

        if self.neighbor_mask is not None and self.neighbor_mask.dim() == 3:
            return _multi_graph_effective_resistance_dynamic_mask(
                self.neighbor_list,
                self.neighbor_mask,
                weight,
                root=self.root,
                inv_method=self.inv_method,
                epsilon=self.epsilon,
                average_duplicates=self.average_duplicates,
                backward_chunk_size=self.backward_chunk_size,
            )

        if not self._has_static_mapping:
            raise ValueError(
                "neighbor_mask must be passed to forward when no static mapping was "
                "created in __init__."
            )

        if self.shared_neighbor_list:
            mapping = _mapping_from_buffers(self, "shared")
            return _multi_graph_effective_resistance_with_mapping(
                mapping,
                weight,
                root=self.root,
                inv_method=self.inv_method,
                epsilon=self.epsilon,
                backward_chunk_size=self.backward_chunk_size,
            )

        if weight.dim() != 3 or tuple(weight.shape[1:]) != self.original_shape:
            raise ValueError("weight must have shape (num_graphs, N, k).")
        if self.num_graphs is not None and int(weight.size(0)) != self.num_graphs:
            raise ValueError(
                "weight and unshared neighbor_list disagree on num_graphs: "
                f"{int(weight.size(0))} != {self.num_graphs}."
            )
        if int(weight.size(0)) == 0:
            return weight.new_zeros(weight.shape)

        outputs = []
        for graph in range(int(weight.size(0))):
            mapping = _mapping_from_buffers(self, f"graph_{graph}")
            graph_resistance = _multi_graph_effective_resistance_with_mapping(
                mapping,
                weight[graph : graph + 1],
                root=self.root,
                inv_method=self.inv_method,
                epsilon=self.epsilon,
                backward_chunk_size=self.backward_chunk_size,
            )
            outputs.append(graph_resistance.squeeze(0))

        return torch.stack(outputs, dim=0)


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
    neighbor_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
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
    )
    multi_weight = torch.stack(
        (
            weight,
            weight * 1.25,
            weight * 0.75 + (weight > 0).to(weight.dtype) * 0.1,
        ),
        dim=0,
    )
    unshared_neighbor_list = neighbor_list.unsqueeze(0).expand(
        multi_weight.size(0),
        -1,
        -1,
    ).clone()
    unshared_neighbor_mask = neighbor_mask.unsqueeze(0).expand(
        multi_weight.size(0),
        -1,
        -1,
    ).clone()

    print(f"device: {device}")
    print("6-node neighbor_list with -1 impossible slots and explicit neighbor_mask:")
    print(neighbor_list)
    print(neighbor_mask)

    for inv_method in ("L+J", "L+eI"):
        shared_input = multi_weight.detach().clone().requires_grad_(True)
        shared_module = MultiGraphEffectiveResistance(
            neighbor_list,
            neighbor_mask,
            shared_neighbor_list=True,
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )
        shared_resistance = shared_module(shared_input)
        shared_resistance.sum().backward()

        function_resistance = multi_graph_effective_resistance(
            neighbor_list,
            neighbor_mask,
            shared_input.detach(),
            shared_neighbor_list=True,
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )

        unshared_input = multi_weight.detach().clone().requires_grad_(True)
        unshared_module = MultiGraphEffectiveResistance(
            unshared_neighbor_list,
            unshared_neighbor_mask,
            shared_neighbor_list=False,
            inv_method=inv_method,
            epsilon=0.2,
            backward_chunk_size=2,
        )
        unshared_resistance = unshared_module(unshared_input)
        unshared_resistance.sum().backward()

        single_input = weight.detach().clone().unsqueeze(0).requires_grad_(True)
        single_resistance = shared_module(single_input).squeeze(0)
        single_resistance.sum().backward()

        print(f"\ninv_method: {inv_method}")
        print("shared output shape:", tuple(shared_resistance.shape))
        print("unshared output shape:", tuple(unshared_resistance.shape))
        print("single graph through padded first dim:", tuple(single_resistance.shape))
        print(
            "max shared module-vs-function forward diff:",
            f"{(shared_resistance.detach() - function_resistance).abs().max().item():.3e}",
        )
        print(
            "max shared-vs-unshared forward diff:",
            f"{(shared_resistance.detach() - unshared_resistance.detach()).abs().max().item():.3e}",
        )
        print(
            "max shared-vs-unshared backward diff:",
            f"{(shared_input.grad - unshared_input.grad).abs().max().item():.3e}",
        )
        print("single graph grad shape:", tuple(single_input.grad.shape))


if __name__ == "__main__":
    _demo(torch.device("cpu"))
    if torch.cuda.is_available():
        _demo(torch.device("cuda:0"))
