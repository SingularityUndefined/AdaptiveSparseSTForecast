import time

import torch
import torch.nn as nn
from unrolled_GEM_block import (
    UnrolledGEMBlock,
    _build_grid_window_neighbors,
    _make_grid_signal,
)
from utils_modules import GraphLearningModule

class UnrolledGEM(nn.Module):
    '''
    Unrolled Graph Embedding Module (GEM) for multi-graph learning.
    
    E-step: given signal y and current graph W^o, S, solve the smoothness problem $(I+\mu L)^{-1} x = y$ by unrolled CG.

    M-step: given signal x and current graph W^o, S, update the graph by unrolling the gradient descent steps for the learned graph W, and its connectivity mask S.

    Solve with multi-head graph learning.
    
    Shape of public input y: (batch_size, num_data, num_nodes)
    Internal y/x shape after head expansion: (batch_size, num_data, num_heads, num_nodes)
    Shape of input W^o: (batch_size, num_heads, num_nodes, k)
    Shape of input S: (batch_size, num_heads, num_nodes, k)
    Shape of neighbor list: (num_nodes, k) (possible neighbors are pre-defined)
    '''
    def __init__(self, num_nodes, neighbor_list, input_neighbor_mask, num_heads, num_blocks, E_iters, M_iters, GD_step_init=0.1, mu_init=0.2, gamma_init=0.4, c=20, scale=True, epsilon=0.2, alpha_init=0.5, er_backend="sksparse", er_backward_chunk_size=1024):
        super(UnrolledGEM, self).__init__()
        self.num_nodes = num_nodes
        self.neighbor_list = neighbor_list
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.E_iters = E_iters
        self.M_iters = M_iters
        self.er_backend = er_backend

        if input_neighbor_mask.ndim == 2:
            input_neighbor_mask = input_neighbor_mask.unsqueeze(0).repeat(num_heads, 1, 1) # (1, num_nodes, k) -> (num_heads, num_nodes, k)
        elif input_neighbor_mask.ndim == 3 and input_neighbor_mask.size(0) == 1:
            input_neighbor_mask = input_neighbor_mask.repeat(num_heads, 1, 1)

        # input mask in (G, N, k)
        self.register_buffer("input_S", input_neighbor_mask.bool())

        # Unrolled GEM Block
        self.GLM_list = nn.ModuleList([
            GraphLearningModule(num_nodes, num_heads, neighbor_list)
            for _ in range(num_blocks)
        ])

        self.GEM_block_list = nn.ModuleList([
            UnrolledGEMBlock(
                num_nodes,
                neighbor_list,
                num_heads,
                E_iters=E_iters,
                M_iters=M_iters,
                GD_step_init=GD_step_init,
                mu_init=mu_init,
                gamma_init=gamma_init,
                c=c,
                scale=scale,
                epsilon=epsilon,
                input_neighbor_mask=self.input_S,
                er_backend=er_backend,
                er_backward_chunk_size=er_backward_chunk_size,
            )
            for _ in range(num_blocks)
        ])

        # learnable parameters for skip connections
        self.alpha_list = nn.Parameter(torch.ones(num_blocks) * alpha_init, requires_grad=True)  # learnable skip connection weights for each block

    def forward(self, y, W_o):
        '''
        Forward pass of the Unrolled GEM module.
        
        Args:
            y: input signal, public shape (B, n_data, n_nodes). If a
                pre-expanded internal tensor is passed, (B, n_data, n_head, n_nodes)
                is also accepted.
            W_o: initial graph weights, shape (B, n_head, n_nodes, k)
        
        Returns:
            x: output signal after E-step and M-step, shape (B, n_data, n_head, n_nodes)
            W: updated graph weights after M-step, shape (B, n_head, n_nodes, k)
            S: updated connectivity mask after M-step, shape (B, n_head, n_nodes, k)
        '''
        # TODO: other initial recovery of x from y, e.g., x = y, or x = (I + mu L)^-1 y, or parameterized, etc.
        y = self._ensure_signal_shape(y)
        W_o = self._ensure_graph_shape(W_o, y.size(0), "W_o")
        x = y
        S = self._expand_input_S(y.size(0))
        W_old = None
        for i in range(self.num_blocks):
            if W_old is not None:
                W = self.GLM_list[i](x, S) * self.alpha_list[i] + W_old * (1 - self.alpha_list[i])  # Weighted combination of learned graph and previous graph
            else:
                W = self.GLM_list[i](x, S)
            
            x, W, S = self.GEM_block_list[i](y, W, S)
            W_old = W

        return x, W, S

    def _ensure_signal_shape(self, y):
        if y.ndim == 2:
            y = y.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_heads, -1)
        elif y.ndim == 3:
            y = y.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        elif y.ndim != 4:
            raise ValueError(
                "public input y must have shape (B, n_data, n_nodes) "
                "or legacy shape (B, n_nodes); pre-expanded internal shape "
                "(B, n_data, n_head, n_nodes) is also accepted."
            )
        if y.size(2) != self.num_heads or y.size(3) != self.num_nodes:
            raise ValueError(
                "expanded y must have shape (B, n_data, n_head, n_nodes) with "
                f"n_head={self.num_heads} and n_nodes={self.num_nodes}; "
                "public input y should normally be (B, n_data, n_nodes)."
            )
        return y

    def _ensure_graph_shape(self, graph, batch_size, name):
        if graph.ndim == 3:
            graph = graph.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if (
            graph.ndim != 4
            or graph.size(0) != batch_size
            or graph.size(1) != self.num_heads
            or graph.size(2) != self.num_nodes
            or graph.size(3) != self.neighbor_list.size(1)
        ):
            raise ValueError(
                f"{name} must have shape (B, n_head, n_nodes, k) with "
                f"B={batch_size}, n_head={self.num_heads}, "
                f"n_nodes={self.num_nodes}, k={self.neighbor_list.size(1)}; "
                "legacy (n_head, n_nodes, k) inputs are expanded across B."
            )
        return graph

    def _expand_input_S(self, batch_size):
        return self.input_S.unsqueeze(0).expand(batch_size, -1, -1, -1)


def _as_batched_graph_tensor(graph, batch_size=None):
    if graph.dim() == 3:
        if batch_size is None:
            batch_size = 1
        graph = graph.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if graph.dim() != 4:
        raise ValueError("graph tensor must have shape (H, N, k) or (B, H, N, k).")
    if batch_size is not None and graph.size(0) != batch_size:
        raise ValueError("graph tensor batch dimension does not match.")
    return graph


def _summarize_block_sparsity(prefix, initial_S, previous_S, current_S):
    batch_size = current_S.size(0) if current_S.dim() == 4 else None
    current_S = _as_batched_graph_tensor(current_S, batch_size)
    batch_size = current_S.size(0)
    initial_S = _as_batched_graph_tensor(initial_S, batch_size)
    previous_S = _as_batched_graph_tensor(previous_S, batch_size)
    initial = initial_S.bool()
    previous = previous_S.bool() & initial
    current = current_S.bool() & initial
    initial_counts = initial.flatten(2).sum(dim=2)
    previous_counts = previous.flatten(2).sum(dim=2)
    current_counts = current.flatten(2).sum(dim=2)

    print(prefix)
    for batch_idx in range(batch_size):
        for head_idx in range(current.size(1)):
            initial_count = int(initial_counts[batch_idx, head_idx].item())
            previous_count = int(previous_counts[batch_idx, head_idx].item())
            current_count = int(current_counts[batch_idx, head_idx].item())
            cumulative_removed_count = initial_count - current_count
            incremental_removed_count = previous_count - current_count
            cumulative_sparsity = (
                cumulative_removed_count / initial_count if initial_count else 0.0
            )
            incremental_sparsity = (
                incremental_removed_count / previous_count if previous_count else 0.0
            )
            print(
                f"  batch {batch_idx}, head {head_idx}: "
                f"prev_edges={previous_count // 2}, "
                f"remaining_edges={current_count // 2}, "
                f"newly_removed_edges={incremental_removed_count // 2}, "
                f"layer_sparsity={incremental_sparsity:.2%}, "
                f"cumulative_sparsity={cumulative_sparsity:.2%}"
            )


def _infer_square_grid_size(num_nodes):
    grid_size = int(num_nodes ** 0.5)
    if grid_size * grid_size == num_nodes:
        return grid_size
    return None


def _print_node_degree_changes(prefix, previous_S, current_S, grid_size=None):
    batch_size = current_S.size(0) if current_S.dim() == 4 else None
    current_S = _as_batched_graph_tensor(current_S, batch_size)
    previous_S = _as_batched_graph_tensor(previous_S, current_S.size(0))
    current_degree = current_S.bool().sum(dim=-1).detach().cpu()
    previous_degree = previous_S.bool().sum(dim=-1).detach().cpu()
    degree_delta = current_degree - previous_degree

    print(f"{prefix} node topo degree changes:")
    for batch_idx in range(degree_delta.size(0)):
        for head_idx in range(degree_delta.size(1)):
            head_delta = degree_delta[batch_idx, head_idx]
            print(
                f"  batch {batch_idx}, head {head_idx}: "
                f"delta min={int(head_delta.min().item())}, "
                f"mean={head_delta.float().mean().item():.2f}, "
                f"max={int(head_delta.max().item())}"
            )

            if grid_size is not None and grid_size * grid_size == head_delta.numel():
                print("  degree delta grid:")
                print(head_delta.reshape(grid_size, grid_size))
            else:
                print("  degree delta per node:", head_delta)


def _format_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.3f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _resolve_demo_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda":
        device_index = 0 if device.index is None else device.index
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but this demo requires cuda:0.")
        if torch.cuda.device_count() <= device_index:
            raise RuntimeError(f"CUDA device cuda:{device_index} is not available.")
        torch.cuda.set_device(device_index)
    return device


def _sync_demo_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_demo_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _read_demo_peak_memory(device):
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return peak_allocated_mb, peak_reserved_mb


def _profile_one_backward_step(model, noisy_signal, W_o, device, reset_peak=True):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        if reset_peak:
            _reset_demo_peak_memory(device)

    model.train()
    model.zero_grad(set_to_none=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        _sync_demo_device(device)
        forward_start = time.perf_counter()
        x_forward, W_forward, S_forward = model(noisy_signal, W_o)
        _sync_demo_device(device)
        forward_seconds = time.perf_counter() - forward_start

        loss = x_forward.square().mean() + 1e-4 * W_forward.square().mean()

        _sync_demo_device(device)
        backward_start = time.perf_counter()
        loss.backward()
        _sync_demo_device(device)
        backward_seconds = time.perf_counter() - backward_start

    peak_allocated_mb, peak_reserved_mb = _read_demo_peak_memory(device)

    total_flops = sum(
        event.flops for event in prof.key_averages() if event.flops is not None
    )

    return {
        "er_backend": model.er_backend,
        "loss": loss.detach(),
        "x_shape": tuple(x_forward.shape),
        "W_shape": tuple(W_forward.shape),
        "S_shape": tuple(S_forward.shape),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "total_flops": total_flops,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_backward_seconds": forward_seconds + backward_seconds,
    }


def _build_demo_model(
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    er_backend,
):
    return UnrolledGEM(
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        E_iters=6,
        M_iters=10,
        GD_step_init=0.2,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
        alpha_init=0.5,
        er_backend=er_backend,
    )


def _print_profile_stats(profile_stats, device):
    print(f"one backward step ({profile_stats['er_backend']}):")
    print("  loss:", f"{profile_stats['loss'].item():.6f}")
    print("  x:", profile_stats["x_shape"])
    print("  W:", profile_stats["W_shape"])
    print("  S:", profile_stats["S_shape"])
    print("  forward time:", f"{profile_stats['forward_seconds']:.6f} s")
    print("  backward time:", f"{profile_stats['backward_seconds']:.6f} s")
    print(
        "  forward + backward time:",
        f"{profile_stats['forward_backward_seconds']:.6f} s",
    )
    if device.type == "cuda" and "full_run_seconds" not in profile_stats:
        print(
            "  peak GPU memory allocated:",
            f"{profile_stats['peak_allocated_mb']:.2f} MiB",
        )
        print(
            "  peak GPU memory reserved:",
            f"{profile_stats['peak_reserved_mb']:.2f} MiB",
        )
    print(
        "  profiler FLOPs:",
        _format_flops(profile_stats["total_flops"]),
    )


def _load_reference_state(model, reference_state, er_backend):
    missing, unexpected = model.load_state_dict(reference_state, strict=False)
    unexpected = [
        key for key in unexpected
        if not key.startswith("GEM_block_list.") or ".ER_solver." not in key
    ]
    if unexpected:
        raise RuntimeError(
            f"Unexpected state_dict keys for {er_backend}: {unexpected}"
        )
    return missing


def _run_sparsity_trace(model, noisy_signal):
    model.train()
    noisy_signal = model._ensure_signal_shape(noisy_signal)
    grid_size = _infer_square_grid_size(model.num_nodes)
    input_S = model._expand_input_S(noisy_signal.size(0))
    _summarize_block_sparsity(
        "input candidate graph:",
        input_S,
        input_S,
        input_S,
    )

    with torch.no_grad():
        x = noisy_signal
        S = input_S
        W_old = None
        for block_idx in range(model.num_blocks):
            W_learned = model.GLM_list[block_idx](x, S)
            if W_old is None:
                W = W_learned
            else:
                alpha = model.alpha_list[block_idx]
                W = W_learned * alpha + W_old * (1 - alpha)

            previous_S = S
            x, W, S = model.GEM_block_list[block_idx](noisy_signal, W, S)
            _summarize_block_sparsity(
                f"after block {block_idx + 1:02d}:",
                input_S,
                previous_S,
                S,
            )
            _print_node_degree_changes(
                f"after block {block_idx + 1:02d}:",
                previous_S,
                S,
                grid_size=grid_size,
            )
            W_old = W

    return x, W, S


def _print_trace_outputs(x, W, S):
    active = S.bool()
    print("trace final outputs:")
    print("  x:", tuple(x.shape))
    print("  W:", tuple(W.shape))
    print("  S:", tuple(S.shape))
    print("  x mean/std:", f"{x.mean().item():.6f}/{x.std().item():.6f}")
    if active.any():
        values = W[active]
        print(
            "  active W min/mean/max:",
            f"{values.min().item():.6f}/"
            f"{values.mean().item():.6f}/"
            f"{values.max().item():.6f}",
        )
    else:
        print("  active W min/mean/max: no active edges")


def _build_loaded_demo_model(
    er_backend,
    reference_state,
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    device,
):
    model = _build_demo_model(
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        er_backend,
    ).to(device)
    _load_reference_state(model, reference_state, er_backend)
    return model.to(device)


def _run_backend_trace(
    er_backend,
    reference_state,
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    noisy_signal,
    device,
):
    print(f"sparsity trace with ER backend: {er_backend}")
    model = _build_loaded_demo_model(
        er_backend,
        reference_state,
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        device,
    )
    x, W, S = _run_sparsity_trace(model, noisy_signal)
    _print_trace_outputs(x, W, S)
    print()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_trace_backends(
    backends,
    reference_state,
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    noisy_signal,
    device,
):
    print("Sparsity traces:")
    for er_backend in backends:
        _run_backend_trace(
            er_backend,
            reference_state,
            num_nodes,
            neighbor_list,
            input_S,
            num_heads,
            num_blocks,
            noisy_signal,
            device,
        )


def _run_training_profile_backend(
    er_backend,
    reference_state,
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    noisy_signal,
    W_o,
    device,
):
    print(f"training profile with ER backend: {er_backend}")
    model = _build_loaded_demo_model(
        er_backend,
        reference_state,
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        device,
    )
    _reset_demo_peak_memory(device)
    profile_stats = _profile_one_backward_step(
        model,
        noisy_signal,
        W_o,
        device,
        reset_peak=False,
    )
    _print_profile_stats(profile_stats, device)
    print()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile_stats


def _profile_demo_backends(
    backends,
    reference_state,
    num_nodes,
    neighbor_list,
    input_S,
    num_heads,
    num_blocks,
    noisy_signal,
    W_o,
    device,
):
    stats_by_backend = []
    for er_backend in backends:
        stats_by_backend.append(
            _run_training_profile_backend(
                er_backend,
                reference_state,
                num_nodes,
                neighbor_list,
                input_S,
                num_heads,
                num_blocks,
                noisy_signal,
                W_o,
                device,
            )
        )

    return stats_by_backend


def _print_backend_comparison(stats_by_backend, device):
    if len(stats_by_backend) < 2:
        return
    print("ER backend comparison:")
    for stats in stats_by_backend:
        parts = [
            f"{stats['er_backend']}:",
            f"forward={stats['forward_seconds']:.6f}s",
            f"backward={stats['backward_seconds']:.6f}s",
            f"fwd_bwd={stats['forward_backward_seconds']:.6f}s",
        ]
        if device.type == "cuda":
            parts.append(f"peak_alloc={stats['peak_allocated_mb']:.2f}MiB")
            parts.append(f"peak_reserved={stats['peak_reserved_mb']:.2f}MiB")
        print("  " + ", ".join(parts))


def _demo_block_sparsity(device_name="cuda:0") -> None:
    torch.manual_seed(7)
    device = _resolve_demo_device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)

    grid_size = 10
    batch_size = 4
    num_data = 3
    num_heads = 2
    num_blocks = 4
    num_nodes = grid_size * grid_size
    neighbor_list, input_S = _build_grid_window_neighbors(grid_size, window_size=5)
    neighbor_list = neighbor_list.to(device)
    input_S = input_S.to(device)

    clean_signal = _make_grid_signal(grid_size).to(device).view(1, 1, -1).expand(
        batch_size,
        num_data,
        -1,
    )
    noisy_signal = clean_signal + 0.25 * torch.randn_like(clean_signal)
    W_o = input_S.repeat(num_heads, 1, 1).unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
        -1,
    ).float()

    model = _build_demo_model(
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        er_backend="sksparse",
    ).to(device)
    reference_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    print(f"{grid_size}x{grid_size} grid UnrolledGEM sparsity demo")
    print("device:", device)
    print("batch size:", batch_size)
    print("n_data:", num_data)
    print("neighbor_list shape:", tuple(neighbor_list.shape))
    print("input_S shape:", tuple(input_S.shape))
    print("clean signal shape:", tuple(clean_signal.shape))
    print("noisy signal shape:", tuple(noisy_signal.shape))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    backends = ("torch_dense", "sksparse")
    print()
    _run_trace_backends(
        backends,
        reference_state,
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        noisy_signal,
        device,
    )

    print("Training forward/backward profiles from fresh models:")
    stats_by_backend = _profile_demo_backends(
        backends,
        reference_state,
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        noisy_signal,
        W_o,
        device,
    )
    _print_backend_comparison(stats_by_backend, device)


if __name__ == "__main__":
    _demo_block_sparsity("cuda:0")
