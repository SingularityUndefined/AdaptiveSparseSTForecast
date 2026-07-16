"""Dense upper-triangular utilities for GEM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def normalize_upper_mask(mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).repeat(num_heads, 1, 1)
    elif mask.dim() == 3 and mask.size(0) == 1:
        mask = mask.repeat(num_heads, 1, 1)
    elif mask.dim() != 3:
        raise ValueError("mask must have shape (N, N) or (num_heads, N, N).")
    if int(mask.size(0)) != int(num_heads):
        raise ValueError(f"mask first dimension must be num_heads={num_heads}.")
    if mask.size(1) != mask.size(2):
        raise ValueError("mask must be square in its last two dimensions.")

    upper = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
    return mask & upper


def upper_to_symmetric(upper_matrix: torch.Tensor) -> torch.Tensor:
    upper_matrix = torch.triu(upper_matrix, diagonal=1)
    return upper_matrix + upper_matrix.transpose(-1, -2)


class UnrolledCG(nn.Module):
    """Learned fixed-depth conjugate-gradient style solver."""

    def __init__(
        self,
        num_iters: int,
        alpha_init: float,
        beta_init: float,
        num_heads: int,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if num_iters <= 0:
            raise ValueError("num_iters must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        self.num_iters = int(num_iters)
        self.num_heads = int(num_heads)
        self.alpha = nn.Parameter(torch.empty(self.num_iters, self.num_heads))
        self.beta = nn.Parameter(torch.empty(self.num_iters, self.num_heads))
        nn.init.uniform_(self.alpha, alpha_init - init_scale, alpha_init + init_scale)
        nn.init.uniform_(self.beta, beta_init - init_scale, beta_init + init_scale)

    def forward(self, A_func, B_vecs: torch.Tensor) -> torch.Tensor:
        if B_vecs.dim() != 3 or B_vecs.size(1) != self.num_heads:
            raise ValueError(
                "B_vecs must have shape (batch_size, num_heads, num_nodes)."
            )
        x = torch.zeros_like(B_vecs)
        residual = B_vecs - A_func(x)
        direction = residual.clone()
        for alpha, beta in zip(self.alpha, self.beta):
            alpha = alpha.view(1, -1, 1)
            beta = beta.view(1, -1, 1)
            A_direction = A_func(direction)
            x = x + alpha * direction
            residual = residual - alpha * A_direction
            direction = residual + beta * direction
        return x


class DenseGraphLearningModule(nn.Module):
    """Learn dense upper-triangular graph weights from node signals.

    Args:
        x: ``(batch_size, num_heads, num_nodes)``.
        upper_mask: ``(num_heads, num_nodes, num_nodes)`` upper-triangular mask.

    Returns:
        Upper-triangular weights with shape ``(num_heads, num_nodes, num_nodes)``.
    """

    def __init__(
        self,
        num_nodes: int,
        num_heads: int,
        emb_dim: int = 8,
        feature_dim: int = 8,
        theta: float = 0.5,
        theta_min: float = 1e-4,
        embedding_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if theta <= theta_min:
            raise ValueError("theta must be greater than theta_min.")

        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.emb_dim = int(emb_dim)
        self.feature_dim = int(feature_dim)
        self.theta_min = float(theta_min)

        self.node_embeddings = nn.Parameter(
            torch.empty(self.num_heads, self.num_nodes, self.emb_dim)
        )
        self.fc = nn.Linear(self.emb_dim + 1, self.feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        raw_theta = _inverse_softplus(
            torch.full((self.num_heads,), float(theta - self.theta_min))
        )
        self.raw_theta = nn.Parameter(raw_theta)
        self.reset_parameters(embedding_std)

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + self.theta_min

    def reset_parameters(self, embedding_std: float = 0.02) -> None:
        nn.init.normal_(self.node_embeddings, mean=0.0, std=embedding_std)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, upper_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.num_heads or x.size(2) != self.num_nodes:
            raise ValueError(
                "x must have shape (batch_size, num_heads, num_nodes)."
            )
        upper_mask = normalize_upper_mask(upper_mask, self.num_heads).to(
            device=x.device
        )

        batch_size = int(x.size(0))
        embeddings = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        node_input = torch.cat([x.unsqueeze(-1), embeddings], dim=-1)
        features = self.leakyrelu(self.fc(node_input))
        features_by_head = features.permute(1, 0, 2, 3).contiguous()

        diff = features_by_head.unsqueeze(3) - features_by_head.unsqueeze(2)
        theta = self.theta.to(device=x.device, dtype=features.dtype).view(
            self.num_heads,
            1,
            1,
            1,
        )
        weights = torch.exp(-diff.square().sum(dim=-1) / (2.0 * theta)).mean(dim=1)
        weights = torch.triu(weights, diagonal=1)
        weights = weights * upper_mask.to(device=x.device, dtype=weights.dtype)
        if not torch.isfinite(weights).all():
            raise ValueError("Non-finite values detected in learned graph weights.")
        return weights

