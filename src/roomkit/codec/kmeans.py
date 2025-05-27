"""K-means codebook training."""

from __future__ import annotations

import numpy as np


def _nearest(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Index of the nearest codebook entry for each row of ``data``.

    Uses ``argmax(2 x.c - ||c||^2)`` instead of materialising the full
    distance matrix, which is cheaper for large codebooks.
    """
    scores = 2.0 * data @ codebook.T - np.sum(codebook**2, axis=1)[None, :]
    return np.argmax(scores, axis=1)


def kmeans(
    data: np.ndarray,
    num_clusters: int,
    iterations: int = 25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit k-means with Lloyd's algorithm.

    Args:
        data: Training vectors of shape ``(num_vectors, dim)``.
        num_clusters: Number of clusters.
        iterations: Number of Lloyd updates.
        seed: Seed for the centroid initialisation.

    Returns:
        Tuple of ``(codebook, assignments)`` where the codebook has shape
        ``(num_clusters, dim)`` and assignments label each input vector.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected 2-D data, got shape {data.shape}")
    if not 0 < num_clusters <= data.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of vectors")

    rng = np.random.default_rng(seed)
    initial = rng.choice(data.shape[0], size=num_clusters, replace=False)
    codebook = data[initial].copy()
    assignments = np.zeros(data.shape[0], dtype=np.int64)
    for _ in range(iterations):
        assignments = _nearest(data, codebook)
        for cluster in range(num_clusters):
            members = data[assignments == cluster]
            if members.shape[0] > 0:
                codebook[cluster] = members.mean(axis=0)
    return codebook, assignments
