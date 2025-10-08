"""Residual vector quantization."""

from __future__ import annotations

import numpy as np

from roomkit.codec.kmeans import _nearest, kmeans


def fit_rvq(
    data: np.ndarray,
    n_codebooks: int,
    codebook_size: int,
    iterations: int = 25,
    seed: int = 0,
) -> list[np.ndarray]:
    """Train residual vector quantizers stage by stage.

    Each stage is fitted on the residual left by the previous stages, in the
    style of neural audio codecs.

    Args:
        data: Training vectors ``(num_vectors, dim)``.
        n_codebooks: Number of residual stages.
        codebook_size: Entries per codebook.
        iterations: Lloyd iterations per stage.
        seed: Base seed; each stage gets a derived seed.

    Returns:
        List of codebooks, one per stage.
    """
    if n_codebooks <= 0:
        raise ValueError("n_codebooks must be positive")
    data = np.asarray(data, dtype=np.float64)
    residual = data.copy()
    codebooks = []
    for stage in range(n_codebooks):
        codebook, assignments = kmeans(
            residual, codebook_size, iterations=iterations, seed=seed + stage
        )
        codebooks.append(codebook)
        residual = residual - codebook[assignments]
    return codebooks


def rvq_encode(data: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Quantize ``data`` against each residual stage.

    Returns:
        Integer token array of shape ``(num_vectors, n_codebooks)``.
    """
    data = np.asarray(data, dtype=np.float64)
    residual = data.copy()
    tokens = np.empty((data.shape[0], len(codebooks)), dtype=np.int32)
    for stage, codebook in enumerate(codebooks):
        indices = _nearest(residual, codebook)
        tokens[:, stage] = indices
        residual = residual - codebook[indices]
    return tokens


def rvq_decode(tokens: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Reconstruct vectors from tokens by summing the selected entries."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"expected 2-D tokens, got shape {tokens.shape}")
    if tokens.shape[1] != len(codebooks):
        raise ValueError("token width does not match the number of codebooks")
    reconstruction = np.zeros((tokens.shape[0], codebooks[0].shape[1]))
    for stage, codebook in enumerate(codebooks):
        reconstruction += codebook[tokens[:, stage]]
    return reconstruction
