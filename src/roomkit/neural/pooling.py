"""Utterance-level pooling."""

from __future__ import annotations

import torch


def statistics_pool(
    embeddings: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Statistics pooling: concatenate the mean and std over time.

    Args:
        embeddings: Frame embeddings of shape ``(batch, frames, dim)``.
        mask: Optional boolean mask of shape ``(batch, frames)``; padded
            frames are excluded from the statistics.

    Returns:
        Utterance embeddings of shape ``(batch, 2 * dim)``.
    """
    if embeddings.ndim != 3:
        raise ValueError(f"expected 3-D embeddings, got shape {tuple(embeddings.shape)}")
    if mask is None:
        mean = embeddings.mean(dim=1)
        std = embeddings.std(dim=1)
        return torch.cat([mean, std], dim=-1)
    mask = mask.to(embeddings.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    masked = embeddings * mask
    mean = masked.sum(dim=1) / counts
    second = (masked**2).sum(dim=1) / counts
    # unbiased variance to match torch.std used in the unmasked branch
    unbiased = counts / (counts - 1.0).clamp(min=1.0)
    std = torch.sqrt(((second - mean**2) * unbiased).clamp(min=0.0) + 1e-8)
    return torch.cat([mean, std], dim=-1)
