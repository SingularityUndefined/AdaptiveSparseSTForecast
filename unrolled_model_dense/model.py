import torch
import torch.nn as nn

try:
    from .unrolled_GEM_block import (
        UnrolledDenseGEMBlock,
        build_grid_window_upper_mask,
        make_grid_signal,
    )
    from .utils_modules import DenseGraphLearningModule, normalize_upper_mask
except ImportError:  # pragma: no cover
    from unrolled_GEM_block import (
        UnrolledDenseGEMBlock,
        build_grid_window_upper_mask,
        make_grid_signal,
    )
    from utils_modules import DenseGraphLearningModule, normalize_upper_mask


class UnrolledDenseGEM(nn.Module):
    """Unrolled GEM using dense upper-triangular graph matrices.

    Args:
        y: ``(batch_size, num_nodes)`` or ``(batch_size, num_heads, num_nodes)``.
        W_o: upper-triangular initial weights,
            ``(num_heads, num_nodes, num_nodes)``.

    Returns:
        ``x, W, S`` where ``W`` and ``S`` are upper-triangular dense matrices.
    """

    def __init__(
        self,
        num_nodes,
        input_upper_mask,
        num_heads,
        num_blocks,
        E_iters,
        M_iters,
        GD_step_init=0.1,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
        alpha_init=0.5,
        emb_dim=8,
        feature_dim=8,
        theta=0.5,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.E_iters = int(E_iters)
        self.M_iters = int(M_iters)

        input_upper_mask = normalize_upper_mask(input_upper_mask, self.num_heads)
        if tuple(input_upper_mask.shape[1:]) != (self.num_nodes, self.num_nodes):
            raise ValueError(
                "input_upper_mask must have shape "
                f"({self.num_heads}, {self.num_nodes}, {self.num_nodes})."
            )
        self.register_buffer("input_S", input_upper_mask.bool())

        self.GLM_list = nn.ModuleList(
            [
                DenseGraphLearningModule(
                    self.num_nodes,
                    self.num_heads,
                    emb_dim=emb_dim,
                    feature_dim=feature_dim,
                    theta=theta,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.GEM_block_list = nn.ModuleList(
            [
                UnrolledDenseGEMBlock(
                    self.num_nodes,
                    self.num_heads,
                    E_iters=E_iters,
                    M_iters=M_iters,
                    GD_step_init=GD_step_init,
                    mu_init=mu_init,
                    gamma_init=gamma_init,
                    c=c,
                    scale=scale,
                    epsilon=epsilon,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.alpha_list = nn.Parameter(
            torch.ones(self.num_blocks) * alpha_init,
            requires_grad=True,
        )

    def forward(self, y, W_o):
        x = y
        S = self.input_S
        W_old = None
        for idx in range(self.num_blocks):
            if x.ndim == 2:
                x_for_glm = x.unsqueeze(1).repeat(1, self.num_heads, 1)
            else:
                x_for_glm = x

            W_learned = self.GLM_list[idx](x_for_glm, S)
            if W_old is None:
                W = W_learned
            else:
                alpha = self.alpha_list[idx]
                W = W_learned * alpha + W_old * (1 - alpha)
            W = torch.triu(W, diagonal=1) * S.to(device=W.device, dtype=W.dtype)

            x, W, S = self.GEM_block_list[idx](y, W, S)
            W_old = W

        return x, W, S


def summarize_upper_sparsity(prefix, initial_S, previous_S, current_S):
    initial = initial_S.bool()
    previous = previous_S.bool() & initial
    current = current_S.bool() & initial
    initial_counts = initial.flatten(1).sum(dim=1)
    previous_counts = previous.flatten(1).sum(dim=1)
    current_counts = current.flatten(1).sum(dim=1)

    print(prefix)
    for head_idx, (initial_count, previous_count, current_count) in enumerate(
        zip(initial_counts.tolist(), previous_counts.tolist(), current_counts.tolist())
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
            f"prev_edges={previous_count}, "
            f"remaining_edges={current_count}, "
            f"newly_removed_edges={incremental_removed_count}, "
            f"layer_sparsity={incremental_sparsity:.2%}, "
            f"cumulative_sparsity={cumulative_sparsity:.2%}"
        )


def _demo(device_name="cpu"):
    torch.manual_seed(7)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(7)

    grid_size = 5
    batch_size = 2
    num_heads = 1
    num_blocks = 2
    num_nodes = grid_size * grid_size
    input_S = build_grid_window_upper_mask(
        grid_size,
        window_size=3,
        num_heads=num_heads,
    ).to(device)
    clean_signal = make_grid_signal(grid_size).to(device).repeat(batch_size, 1)
    noisy_signal = clean_signal + 0.25 * torch.randn_like(clean_signal)
    W_o = input_S.to(dtype=noisy_signal.dtype)

    model = UnrolledDenseGEM(
        num_nodes,
        input_S,
        num_heads,
        num_blocks,
        E_iters=3,
        M_iters=3,
        GD_step_init=0.1,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
    ).to(device)

    print("Dense upper-triangular UnrolledGEM demo")
    print("device:", device)
    print("input_S shape:", tuple(input_S.shape))
    print("noisy signal shape:", tuple(noisy_signal.shape))
    summarize_upper_sparsity("input candidate graph:", input_S, input_S, input_S)

    x = noisy_signal
    S = model.input_S
    W_old = None
    with torch.no_grad():
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
            summarize_upper_sparsity(
                f"after block {block_idx + 1:02d}:",
                model.input_S,
                previous_S,
                S,
            )
            W_old = W

    model.train()
    model.zero_grad(set_to_none=True)
    x, W, S = model(noisy_signal, W_o)
    loss = x.square().mean() + 1e-4 * W.square().mean()
    loss.backward()
    print("one backward step:")
    print("  loss:", f"{loss.detach().item():.6f}")
    print("  x:", tuple(x.shape))
    print("  W:", tuple(W.shape))
    print("  S:", tuple(S.shape))


if __name__ == "__main__":
    _demo("cpu")
