import torch
import torch.nn as nn

try:
    from .effective_resistance_dense import DenseUpperEffectiveResistance
    from .utils_modules import (
        DenseGraphLearningModule,
        UnrolledCG,
        normalize_upper_mask,
        upper_to_symmetric,
    )
except ImportError:  # pragma: no cover
    from effective_resistance_dense import DenseUpperEffectiveResistance
    from utils_modules import (
        DenseGraphLearningModule,
        UnrolledCG,
        normalize_upper_mask,
        upper_to_symmetric,
    )


class UnrolledDenseGEMBlock(nn.Module):
    """Dense upper-triangular GEM block.

    ``W`` and ``S`` use shape ``(num_heads, num_nodes, num_nodes)``.  Only
    entries ``i < j`` are meaningful; the lower triangle and diagonal are
    forced to zero.
    """

    def __init__(
        self,
        num_nodes,
        num_heads,
        E_iters=5,
        M_iters=5,
        GD_step_init=0.1,
        mu_init=0.2,
        gamma_init=0.4,
        c=20,
        scale=True,
        epsilon=0.2,
        xi=1,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.E_iters = int(E_iters)
        self.M_iters = int(M_iters)
        self.c = c
        self.scale = scale
        self.epsilon = epsilon
        self.xi = xi

        self.CG_solver = UnrolledCG(
            self.E_iters,
            0.1,
            0.0,
            self.num_heads,
            init_scale=0.02,
        )
        self.mu = nn.Parameter(torch.ones(1) * mu_init, requires_grad=True)
        self.gamma_list = nn.Parameter(
            torch.ones(self.M_iters) * gamma_init,
            requires_grad=True,
        )
        self.step_size_list = nn.Parameter(
            torch.ones(self.M_iters) * GD_step_init,
            requires_grad=True,
        )
        self.ER_solver = DenseUpperEffectiveResistance(
            inv_method="L+J",
            epsilon=self.epsilon,
        )

    def _normalize_upper(self, matrix):
        return torch.triu(matrix, diagonal=1)

    def edge_diff_square(self, x, upper_mask):
        """Return mean squared node differences in upper-triangular layout."""
        batch_size, num_heads, num_nodes = x.size()
        if num_heads != self.num_heads or num_nodes != self.num_nodes:
            raise ValueError("x has incompatible shape.")
        upper_mask = normalize_upper_mask(upper_mask, self.num_heads).to(
            device=x.device
        )
        diff = x.unsqueeze(-1) - x.unsqueeze(-2)
        diff_square = diff.square().mean(dim=0)
        return torch.triu(diff_square, diagonal=1) * upper_mask.to(dtype=x.dtype)

    def apply_L(self, x, W, S):
        """Apply graph Laplacian from dense upper-triangular ``W`` and ``S``."""
        upper_weight = self._normalize_upper(W) * normalize_upper_mask(
            S,
            self.num_heads,
        ).to(device=W.device, dtype=W.dtype)
        adjacency = upper_to_symmetric(upper_weight)
        degree = adjacency.sum(dim=-1)
        Ax = torch.einsum("gij,bgj->bgi", adjacency, x)
        return degree.unsqueeze(0) * x - Ax

    def scale_graph_weights(self, W, S):
        active_weight = W * S.to(device=W.device, dtype=W.dtype)
        fro_norm = torch.sqrt(active_weight.pow(2).sum(dim=(1, 2)))
        scale_factors = self.c / (fro_norm + 1e-8)
        return W * scale_factors.view(-1, 1, 1)

    def E_step(self, y, W, S):
        if y.ndim == 2:
            y = y.unsqueeze(1).repeat(1, self.num_heads, 1)

        def lhs(x):
            return x + self.mu * self.apply_L(x, W, S)

        return self.CG_solver(lhs, y)

    def M_step(self, x, W, input_S):
        candidate_mask = normalize_upper_mask(input_S, self.num_heads).to(
            device=W.device
        )
        neighbor_mask = candidate_mask
        W = self._normalize_upper(W) * candidate_mask.to(dtype=W.dtype)
        x_diff_square = self.edge_diff_square(x, candidate_mask)

        for idx in range(self.M_iters):
            if W[~candidate_mask].abs().sum() != 0:
                raise ValueError("W has non-zero entries outside candidate mask.")
            resistance = self.ER_solver(W, neighbor_mask)
            sparse_penalty = self.gamma_list[idx] * (W + self.xi)
            grad = x_diff_square + sparse_penalty - resistance
            grad = grad * neighbor_mask.to(device=grad.device, dtype=grad.dtype)

            W = torch.clamp(W - self.step_size_list[idx] * grad, min=0)
            W = self._normalize_upper(W) * candidate_mask.to(dtype=W.dtype)
            neighbor_mask = (W > 0) & candidate_mask

        return W

    def forward(self, y, W_o, input_S, default_threshold=1e-4):
        candidate_mask = normalize_upper_mask(input_S, self.num_heads).to(
            device=W_o.device
        )
        W_o = self._normalize_upper(W_o) * candidate_mask.to(dtype=W_o.dtype)
        x = self.E_step(y, W_o, candidate_mask)
        W = self.M_step(x, W_o, candidate_mask)

        active_values = W[candidate_mask]
        if active_values.numel() == 0:
            threshold = W.new_tensor(default_threshold)
        else:
            threshold = torch.clamp(
                torch.quantile(active_values, 0.05),
                min=default_threshold,
            )
        W_new = torch.where(W < threshold, torch.zeros_like(W), W)
        W_new = self._normalize_upper(W_new)
        S = (W_new > 0) & candidate_mask
        S_float = S.to(dtype=W_new.dtype)

        if self.scale:
            W_new = self.scale_graph_weights(W_new, S_float)
            W_new = self._normalize_upper(W_new) * S_float

        return x, W_new, S_float


def build_grid_window_upper_mask(grid_size=7, window_size=5, num_heads=1):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    radius = window_size // 2
    num_nodes = grid_size * grid_size
    mask = torch.zeros(num_heads, num_nodes, num_nodes, dtype=torch.bool)
    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col
            for drow in range(-radius, radius + 1):
                for dcol in range(-radius, radius + 1):
                    if drow == 0 and dcol == 0:
                        continue
                    nrow = row + drow
                    ncol = col + dcol
                    if 0 <= nrow < grid_size and 0 <= ncol < grid_size:
                        neighbor = nrow * grid_size + ncol
                        i = min(node, neighbor)
                        j = max(node, neighbor)
                        if i != j:
                            mask[:, i, j] = True
    return mask


def make_grid_signal(grid_size=7):
    coords = torch.linspace(-1.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    signal = (
        torch.sin(torch.pi * xx)
        + 0.6 * torch.cos(torch.pi * yy)
        + torch.exp(-5.0 * (xx.square() + yy.square()))
    )
    signal = (signal - signal.mean()) / signal.std()
    return signal.reshape(1, -1)

