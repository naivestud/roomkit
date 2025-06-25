"""Fixed sinusoidal positional embeddings."""

from __future__ import annotations

import math

import torch


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """Sinusoidal position embeddings of shape ``(length, dim)``.

    The same fixed embedding family used by the original transformer and by
    whisper; no parameters are learned. ``dim`` must be even.

    Args:
        length: Number of positions.
        dim: Embedding dimension.

    Returns:
        Float tensor of shape ``(length, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    positions = torch.arange(length, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
