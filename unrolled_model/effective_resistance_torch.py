"""Torch-native exact effective resistance backend.

This module keeps the effective-resistance solve on the current torch device.
It uses a sparse edge-list representation to build the graph Laplacian, then
uses dense batched Cholesky solves.  On CUDA this avoids the CPU NumPy/CHOLMOD
round trip used by ``effective_resistance.py``.

The implementation is exact for the selected edge set.  It is not a sparse
direct solver: the factorized matrix is dense, so this backend trades memory and
O(N^3) factorization cost for staying on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import torch


_INV_METHODS = ("L+J", "L+eI")


@dataclass(frozen=True)
class NeighborToEdgeMap:
    original_shape: Tuple[int, int]
    valid_flat_index: torch.Tensor
    entry_to_edge: torch.Tensor
    entry_alpha: torch.Tensor
    edge_index: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.original_shape[0])

    @property
    def width(self) -> int:
        return int(self.original_shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


NeighborInput = Union[torch.Tensor, Iterable[Iterable[int]]]
MultiNeighborInput = Union[torch.Tensor, Iterable[Iterable[Iterable[int]]]]
MaskInput = Union[torch.Tensor, Iterable[Iterable[bool]]]
MultiMaskInput = Union[torch.Tensor, Iterable[Iterable[Iterable[bool]]]]


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


def _neighbor_original_shape(neighbor_cpu: torch.Tensor) -> Tuple[int, int]:
    if neighbor_cpu.dim() == 2:
        return (int(neighbor_cpu.size(0)), int(neighbor_cpu.size(1)))
    if neighbor_cpu.dim() == 3:
        return (int(neighbor_cpu.size(1)), int(neighbor_cpu.size(2)))
    raise ValueError("neighbor_list must have shape (N, k) or (num_graphs, N, k).")


def _normalize_chunk_size(backward_chunk_size: int, num_edges: int) -> int:
    return max(1, min(int(backward_chunk_size), max(1, int(num_edges))))


def _validate_edge_problem(
    edge_weight: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    inv_method: str,
    epsilon: float,
) -> None:
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
    if not bool(torch.isfinite(edge_weight).all()):
        raise ValueError("All edge conductances must be finite.")


def _validate_dynamic_graph_inputs(
    neighbor_cpu: torch.Tensor,
    mask_cpu: torch.Tensor,
    weight: torch.Tensor,
) -> None:
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
        return

    if neighbor_cpu.dim() == 3:
        if tuple(neighbor_cpu.shape) != tuple(mask_cpu.shape):
            raise ValueError(
                "neighbor_list and neighbor_mask must have the same shape when "
                "neighbor_list is graph-specific."
            )
        return

    raise ValueError("neighbor_list must have shape (N, k) or (num_graphs, N, k).")


def build_neighbor_to_edge_map(
    neighbor_list: NeighborInput,
    neighbor_mask: MaskInput,
    *,
    average_duplicates: bool = True,
) -> NeighborToEdgeMap:
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

    return NeighborToEdgeMap(
        original_shape=(int(num_nodes), int(width)),
        valid_flat_index=valid_flat_index,
        entry_to_edge=entry_to_edge,
        entry_alpha=entry_alpha,
        edge_index=unique_edges.t().contiguous(),
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


def _mapping_from_buffers(module: torch.nn.Module, prefix: str) -> NeighborToEdgeMap:
    return NeighborToEdgeMap(
        original_shape=getattr(module, f"{prefix}_original_shape"),
        valid_flat_index=getattr(module, f"{prefix}_valid_flat_index"),
        entry_to_edge=getattr(module, f"{prefix}_entry_to_edge"),
        entry_alpha=getattr(module, f"{prefix}_entry_alpha"),
        edge_index=getattr(module, f"{prefix}_edge_index"),
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
        raise ValueError("edge_values must have shape (num_graphs, num_edges).")
    if int(edge_values.size(1)) != int(num_edges):
        raise ValueError("edge_values length does not match the neighbor mapping.")

    device = edge_values.device
    num_graphs = int(edge_values.size(0))
    output = edge_values.new_zeros((num_graphs, original_shape[0] * original_shape[1]))
    if num_edges:
        valid_flat_index = valid_flat_index.to(device=device)
        entry_to_edge = entry_to_edge.to(device=device)
        gathered = edge_values.index_select(1, entry_to_edge)
        output[:, valid_flat_index] = gathered
    return output.reshape(num_graphs, *original_shape)


def _empty_edge_state(
    edge_weight: torch.Tensor,
    num_graphs: int,
    solve_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        edge_weight.new_empty((num_graphs, solve_rows, 0)),
        edge_weight.new_empty(0, dtype=torch.long),
        edge_weight.new_empty(0, dtype=torch.long),
    )


def _edge_solver_rows_torch(
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


def _dense_laplacian_from_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    device = edge_weight.device
    num_graphs = int(edge_weight.size(0))
    u = edge_index[0].to(device=device)
    v = edge_index[1].to(device=device)
    laplacian = edge_weight.new_zeros((num_graphs, num_nodes * num_nodes))

    def scatter(flat_index: torch.Tensor, values: torch.Tensor) -> None:
        expanded = flat_index.unsqueeze(0).expand(num_graphs, -1)
        laplacian.scatter_add_(1, expanded, values)

    scatter(u * num_nodes + u, edge_weight)
    scatter(v * num_nodes + v, edge_weight)
    scatter(u * num_nodes + v, -edge_weight)
    scatter(v * num_nodes + u, -edge_weight)
    return laplacian.reshape(num_graphs, num_nodes, num_nodes)


def _system_matrix_from_laplacian(
    laplacian: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
) -> torch.Tensor:
    num_nodes = int(laplacian.size(1))
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
            "Torch dense Cholesky failed for graph "
            f"{bad_graph}. Check connectivity and non-negative weights."
        )
    return chol


def _edge_rhs(
    solve_rows: int,
    edge_u_rows: torch.Tensor,
    edge_v_rows: torch.Tensor,
    start: int,
    end: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rhs = torch.zeros((solve_rows, end - start), dtype=dtype, device=device)
    if end == start:
        return rhs

    columns = torch.arange(end - start, device=device)
    u = edge_u_rows[start:end]
    v = edge_v_rows[start:end]
    u_mask = u >= 0
    v_mask = v >= 0
    if bool(u_mask.any()):
        rhs[u[u_mask], columns[u_mask]] += 1.0
    if bool(v_mask.any()):
        rhs[v[v_mask], columns[v_mask]] -= 1.0
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


def _solve_edge_resistance_forward(
    edge_weight: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    root: int,
    inv_method: str,
    epsilon: float,
    chunk_size: int,
    keep_potentials: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = edge_weight.device
    dtype = edge_weight.dtype
    num_graphs = int(edge_weight.size(0))
    num_edges = int(edge_weight.size(1))
    solve_rows = _solve_rows(num_nodes, inv_method)

    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_u_rows, edge_v_rows = _edge_solver_rows_torch(
        num_nodes,
        edge_index,
        root,
        inv_method,
    )

    if num_edges == 0:
        potentials, edge_u_rows, edge_v_rows = _empty_edge_state(
            edge_weight,
            num_graphs,
            solve_rows,
        )
        return (
            edge_weight.new_zeros((num_graphs, 0)),
            potentials,
            edge_u_rows,
            edge_v_rows,
        )

    laplacian = _dense_laplacian_from_edges(edge_index, edge_weight, num_nodes)
    matrix = _system_matrix_from_laplacian(
        laplacian,
        root=root,
        inv_method=inv_method,
        epsilon=epsilon,
    )
    chol = _cholesky_factor(matrix)

    potentials = (
        edge_weight.new_empty((num_graphs, solve_rows, num_edges))
        if keep_potentials
        else None
    )
    resistance = edge_weight.new_empty((num_graphs, num_edges))

    for start in range(0, num_edges, chunk_size):
        end = min(start + chunk_size, num_edges)
        rhs = _edge_rhs(
            solve_rows,
            edge_u_rows,
            edge_v_rows,
            start,
            end,
            dtype=dtype,
            device=device,
        )
        solved = torch.cholesky_solve(
            rhs.unsqueeze(0).expand(num_graphs, -1, -1),
            chol,
            upper=False,
        )
        if potentials is not None:
            potentials[:, :, start:end] = solved
        resistance[:, start:end] = (rhs.unsqueeze(0) * solved).sum(dim=1)

    if potentials is None:
        potentials, edge_u_rows, edge_v_rows = _empty_edge_state(
            edge_weight,
            num_graphs,
            solve_rows,
        )

    return resistance, potentials, edge_u_rows, edge_v_rows


class _TorchDenseMultiEdgeEffectiveResistanceFunction(torch.autograd.Function):
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
        _validate_edge_problem(
            edge_weight,
            edge_index,
            num_nodes,
            root,
            inv_method,
            epsilon,
        )
        num_edges = int(edge_weight.size(1))
        chunk_size = _normalize_chunk_size(backward_chunk_size, num_edges)
        resistance, potentials, edge_u_rows, edge_v_rows = (
            _solve_edge_resistance_forward(
                edge_weight,
                edge_index,
                num_nodes=num_nodes,
                root=root,
                inv_method=inv_method,
                epsilon=epsilon,
                chunk_size=chunk_size,
                keep_potentials=edge_weight.requires_grad,
            )
        )

        ctx.save_for_backward(potentials, edge_u_rows, edge_v_rows)
        ctx.backward_chunk_size = chunk_size
        return resistance

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


def torch_dense_multi_edge_effective_resistance(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    *,
    root: int = 0,
    inv_method: str = "L+J",
    epsilon: float = 0.2,
    backward_chunk_size: int = 1024,
) -> torch.Tensor:
    return _TorchDenseMultiEdgeEffectiveResistanceFunction.apply(
        edge_weight,
        edge_index,
        int(num_nodes),
        int(root),
        inv_method,
        float(epsilon),
        int(backward_chunk_size),
    )


def _torch_dense_multi_graph_effective_resistance_with_mapping(
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
        num_edges=mapping.num_edges,
    )
    edge_resistance = torch_dense_multi_edge_effective_resistance(
        mapping.edge_index.to(device=weight.device),
        edge_weight,
        mapping.num_nodes,
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
        num_edges=mapping.num_edges,
    )


def _torch_dense_multi_graph_effective_resistance_unshared(
    mappings: Iterable[NeighborToEdgeMap],
    weight: torch.Tensor,
    *,
    root: int,
    inv_method: str,
    epsilon: float,
    backward_chunk_size: int,
) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError("weight must have shape (num_graphs, N, k).")

    outputs = []
    for graph, mapping in enumerate(mappings):
        graph_resistance = _torch_dense_multi_graph_effective_resistance_with_mapping(
            mapping,
            weight[graph : graph + 1],
            root=root,
            inv_method=inv_method,
            epsilon=epsilon,
            backward_chunk_size=backward_chunk_size,
        )
        outputs.append(graph_resistance.squeeze(0))

    if not outputs:
        return weight.new_zeros(weight.shape)
    return torch.stack(outputs, dim=0)


def _torch_dense_multi_graph_effective_resistance_dynamic_mask(
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
    _validate_dynamic_graph_inputs(neighbor_cpu, mask_cpu, weight)

    mappings = []
    for graph in range(int(mask_cpu.size(0))):
        graph_neighbor_list = (
            neighbor_cpu if neighbor_cpu.dim() == 2 else neighbor_cpu[graph]
        )
        mappings.append(
            build_neighbor_to_edge_map(
                graph_neighbor_list,
                mask_cpu[graph],
                average_duplicates=average_duplicates,
            )
        )
    return _torch_dense_multi_graph_effective_resistance_unshared(
        mappings,
        weight,
        root=root,
        inv_method=inv_method,
        epsilon=epsilon,
        backward_chunk_size=backward_chunk_size,
    )


class TorchDenseMultiGraphEffectiveResistance(torch.nn.Module):
    """Exact ER module that solves on the current torch device.

    If a static ``neighbor_mask`` is provided in ``__init__``, calls to
    ``forward`` ignore dynamic masks and reuse that superset mapping.  This is
    useful in GEM because later masks only zero out candidate edges; avoiding
    dynamic remapping removes repeated GPU-to-CPU mask transfers.
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
        self.average_duplicates = bool(average_duplicates)
        self.backward_chunk_size = int(backward_chunk_size)
        _validate_inv_method(self.inv_method, self.epsilon)

        neighbor_cpu = _as_long_tensor(neighbor_list)
        self.register_buffer("neighbor_list", neighbor_cpu, persistent=False)
        self._has_static_mapping = False
        self.num_graphs: Optional[int] = None

        if neighbor_mask is None:
            self._init_without_static_mapping(neighbor_cpu)
            return

        mask_cpu = _as_bool_tensor(neighbor_mask)
        if self.shared_neighbor_list:
            self._init_shared_static_mapping(neighbor_cpu, mask_cpu)
            return

        self._init_unshared_static_mappings(neighbor_cpu, mask_cpu)

    def _init_without_static_mapping(self, neighbor_cpu: torch.Tensor) -> None:
        self.original_shape = _neighbor_original_shape(neighbor_cpu)
        if neighbor_cpu.dim() == 3:
            self.num_graphs = int(neighbor_cpu.size(0))

    def _init_shared_static_mapping(
        self,
        neighbor_cpu: torch.Tensor,
        mask_cpu: torch.Tensor,
    ) -> None:
        if neighbor_cpu.dim() != 2:
            raise ValueError(
                "shared static neighbor_list must have shape (N, k)."
            )
        if mask_cpu.dim() == 3:
            if tuple(mask_cpu.shape[1:]) != tuple(neighbor_cpu.shape):
                raise ValueError(
                    "neighbor_mask must have shape (num_graphs, N, k), where "
                    "neighbor_list has shape (N, k)."
                )
            mask_cpu = mask_cpu.any(dim=0)
        mapping = build_neighbor_to_edge_map(
            neighbor_cpu,
            mask_cpu,
            average_duplicates=self.average_duplicates,
        )
        self.original_shape = mapping.original_shape
        _register_mapping_buffers(self, "shared", mapping)
        self._has_static_mapping = True

    def _init_unshared_static_mappings(
        self,
        neighbor_cpu: torch.Tensor,
        mask_cpu: torch.Tensor,
    ) -> None:
        if neighbor_cpu.dim() != 3 or mask_cpu.dim() != 3:
            raise ValueError(
                "unshared static neighbor_list and neighbor_mask must have shape "
                "(num_graphs, N, k)."
            )
        if tuple(neighbor_cpu.shape) != tuple(mask_cpu.shape):
            raise ValueError("unshared neighbor_list and neighbor_mask shapes differ.")

        self.num_graphs = int(neighbor_cpu.size(0))
        self.original_shape = _neighbor_original_shape(neighbor_cpu)
        for graph in range(self.num_graphs):
            mapping = build_neighbor_to_edge_map(
                neighbor_cpu[graph],
                mask_cpu[graph],
                average_duplicates=self.average_duplicates,
            )
            _register_mapping_buffers(self, f"graph_{graph}", mapping)
        self._has_static_mapping = True

    def _solver_kwargs(self) -> dict:
        return {
            "root": self.root,
            "inv_method": self.inv_method,
            "epsilon": self.epsilon,
            "backward_chunk_size": self.backward_chunk_size,
        }

    def _static_graph_mappings(self) -> Iterable[NeighborToEdgeMap]:
        if self.num_graphs is None:
            raise ValueError("num_graphs is unknown for unshared static mapping.")
        for graph in range(self.num_graphs):
            yield _mapping_from_buffers(self, f"graph_{graph}")

    def forward(
        self,
        weight: torch.Tensor,
        neighbor_mask: Optional[MultiMaskInput] = None,
    ) -> torch.Tensor:
        if self._has_static_mapping:
            solver_kwargs = self._solver_kwargs()
            if self.shared_neighbor_list:
                mapping = _mapping_from_buffers(self, "shared")
                return _torch_dense_multi_graph_effective_resistance_with_mapping(
                    mapping,
                    weight,
                    **solver_kwargs,
                )

            if weight.dim() != 3 or tuple(weight.shape[1:]) != self.original_shape:
                raise ValueError("weight must have shape (num_graphs, N, k).")
            if self.num_graphs is not None and int(weight.size(0)) != self.num_graphs:
                raise ValueError(
                    "weight and unshared neighbor_list disagree on num_graphs: "
                    f"{int(weight.size(0))} != {self.num_graphs}."
                )
            return _torch_dense_multi_graph_effective_resistance_unshared(
                self._static_graph_mappings(),
                weight,
                **solver_kwargs,
            )

        if neighbor_mask is None:
            raise ValueError(
                "neighbor_mask must be passed to forward when no static mapping was "
                "created in __init__."
            )

        return _torch_dense_multi_graph_effective_resistance_dynamic_mask(
            self.neighbor_list,
            neighbor_mask,
            weight,
            root=self.root,
            inv_method=self.inv_method,
            epsilon=self.epsilon,
            average_duplicates=self.average_duplicates,
            backward_chunk_size=self.backward_chunk_size,
        )
