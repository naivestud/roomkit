"""Pre-LayerNorm transformer block."""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Attention + MLP block with pre-normalisation residuals.

    Matches the block layout used by the whisper encoder: LayerNorm before
    each sub-layer, residual connections around both, GELU in the MLP.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(d_model)
        width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, width),
            nn.GELU(),
            nn.Linear(width, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``(batch, frames, d_model)`` embeddings."""
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x))
