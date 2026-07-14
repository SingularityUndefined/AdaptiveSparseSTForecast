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
    
    Shape of input y: (batch_size, num_nodes)
    Shape of input W^o: (num_graphs, num_nodes, k)
    Shape of input S: (num_graphs, num_nodes, k)
    Shape of neighbor list: (num_nodes, k) (possible neighbors are pre-defined)
    '''
    def __init__(self, num_nodes, neighbor_list, input_neighbor_mask, num_heads, num_blocks, E_iters, M_iters, GD_step_init=0.1, mu_init=0.2, gamma_init=0.4, c=20, scale=True, epsilon=0.2, alpha_init=0.5):
        super(UnrolledGEM, self).__init__()
        self.num_nodes = num_nodes
        self.neighbor_list = neighbor_list
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.E_iters = E_iters
        self.M_iters = M_iters

        if input_neighbor_mask.ndim == 2:
            input_neighbor_mask = input_neighbor_mask.unsqueeze(0).repeat(num_heads, 1, 1) # (1, num_nodes, k) -> (num_heads, num_nodes, k)

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
            )
            for _ in range(num_blocks)
        ])

        # learnable parameters for skip connections
        self.alpha_list = nn.Parameter(torch.ones(num_blocks) * alpha_init, requires_grad=True)  # learnable skip connection weights for each block

    def forward(self, y, W_o):
        '''
        Forward pass of the Unrolled GEM module.
        
        Args:
            y: input signal, shape (batch_size, num_nodes)
            W_o: initial graph weights, shape (num_graphs, num_nodes, k)
        
        Returns:
            x: output signal after E-step and M-step, shape (batch_size, num_nodes)
            W: updated graph weights after M-step, shape (num_graphs, num_nodes, k)
            S: updated connectivity mask after M-step, shape (num_graphs, num_nodes, k)
        '''
        # TODO: other initial recovery of x from y, e.g., x = y, or x = (I + mu L)^-1 y, or parameterized, etc.
        x = y
        S = self.input_S
        W_old = None
        for i in range(self.num_blocks):
            if x.ndim == 2:
                x_for_glm = x.unsqueeze(1).repeat(1, self.num_heads, 1)
            else:
                x_for_glm = x

            if W_old is not None:
                W = self.GLM_list[i](x_for_glm, S) * self.alpha_list[i] + W_old * (1 - self.alpha_list[i])  # Weighted combination of learned graph and previous graph
            else:
                W = self.GLM_list[i](x_for_glm, S)
            
            x, W, S = self.GEM_block_list[i](y, W, S)
            W_old = W

        return x, W, S


def _summarize_block_sparsity(prefix, initial_S, previous_S, current_S):
    initial = initial_S.bool()
    previous = previous_S.bool() & initial
    current = current_S.bool() & initial
    initial_counts = initial.flatten(1).sum(dim=1)
    previous_counts = previous.flatten(1).sum(dim=1)
    current_counts = current.flatten(1).sum(dim=1)

    print(prefix)
    for head_idx, (initial_count, previous_count, current_count) in enumerate(
        zip(
            initial_counts.tolist(),
            previous_counts.tolist(),
            current_counts.tolist(),
        )
    ):
        cumulative_removed_count = initial_count - current_count
        incremental_removed_count = previous_count - current_count
        cumulative_sparsity = (
            cumulative_removed_count / initial_count if initial_count else 0.0
        )
        incremental_sparsity = (
            incremental_removed_count / previous_count if previous_count else 0.0
        )
        print(
            f"  head {head_idx}: "
            f"prev_edges={previous_count // 2}, "
            f"remaining_edges={current_count // 2}, "
            f"newly_removed_edges={incremental_removed_count // 2}, "
            f"layer_sparsity={incremental_sparsity:.2%}, "
            f"cumulative_sparsity={cumulative_sparsity:.2%}"
        )


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


def _profile_one_backward_step(model, noisy_signal, W_o, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    model.train()
    model.zero_grad(set_to_none=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        x_forward, W_forward, S_forward = model(noisy_signal, W_o)
        loss = x_forward.square().mean() + 1e-4 * W_forward.square().mean()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        peak_allocated_mb = None
        peak_reserved_mb = None

    total_flops = sum(
        event.flops for event in prof.key_averages() if event.flops is not None
    )

    return {
        "loss": loss.detach(),
        "x_shape": tuple(x_forward.shape),
        "W_shape": tuple(W_forward.shape),
        "S_shape": tuple(S_forward.shape),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "total_flops": total_flops,
    }


def _demo_block_sparsity(device_name="cuda:0") -> None:
    torch.manual_seed(7)
    device = _resolve_demo_device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)

    grid_size = 7
    num_heads = 1
    num_blocks = 4
    num_nodes = grid_size * grid_size
    neighbor_list, input_S = _build_grid_window_neighbors(grid_size, window_size=5)
    neighbor_list = neighbor_list.to(device)
    input_S = input_S.to(device)

    clean_signal = _make_grid_signal(grid_size).to(device)
    noisy_signal = clean_signal + 0.25 * torch.randn_like(clean_signal)
    W_o = input_S.float()

    model = UnrolledGEM(
        num_nodes,
        neighbor_list,
        input_S,
        num_heads,
        num_blocks,
        E_iters=6,
        M_iters=10,
        GD_step_init=0.1,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
        alpha_init=0.5,
    ).to(device)

    print("7x7 grid UnrolledGEM sparsity demo")
    print("device:", device)
    print("neighbor_list shape:", tuple(neighbor_list.shape))
    print("input_S shape:", tuple(input_S.shape))
    print("clean signal shape:", tuple(clean_signal.shape))
    print("noisy signal shape:", tuple(noisy_signal.shape))
    _summarize_block_sparsity(
        "input candidate graph:",
        model.input_S,
        model.input_S,
        model.input_S,
    )

    with torch.no_grad():
        x = noisy_signal
        S = model.input_S
        W_old = None
        for block_idx in range(model.num_blocks):
            if x.ndim == 2:
                x_for_glm = x.unsqueeze(1).repeat(1, model.num_heads, 1)
            else:
                x_for_glm = x

            W_learned = model.GLM_list[block_idx](x_for_glm, S)
            if W_old is None:
                W = W_learned
            else:
                alpha = model.alpha_list[block_idx]
                W = W_learned * alpha + W_old * (1 - alpha)

            previous_S = S
            x, W, S = model.GEM_block_list[block_idx](noisy_signal, W, S)
            _summarize_block_sparsity(
                f"after block {block_idx + 1:02d}:",
                model.input_S,
                previous_S,
                S,
            )
            W_old = W

    profile_stats = _profile_one_backward_step(model, noisy_signal, W_o, device)

    print("one backward step:")
    print("  loss:", f"{profile_stats['loss'].item():.6f}")
    print("  x:", profile_stats["x_shape"])
    print("  W:", profile_stats["W_shape"])
    print("  S:", profile_stats["S_shape"])
    if device.type == "cuda":
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


if __name__ == "__main__":
    _demo_block_sparsity("cuda:0")
