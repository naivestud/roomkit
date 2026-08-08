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


def _kmeanspp_init(data: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """Seed centroids with k-means++ so bad random starts are rare."""
    num_vectors = data.shape[0]
    centers = [data[rng.integers(num_vectors)]]
    for _ in range(1, num_clusters):
        stacked = np.stack(centers)
        squared = np.min(np.sum((data[:, None, :] - stacked[None, :, :]) ** 2, axis=-1), axis=1)
        total = squared.sum()
        probs = squared / total if total > 0 else None
        centers.append(data[rng.choice(num_vectors, p=probs)])
    return np.stack(centers)


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
    codebook = _kmeanspp_init(data, num_clusters, rng)
    assignments = np.zeros(data.shape[0], dtype=np.int64)
    for _ in range(iterations):
        assignments = _nearest(data, codebook)
        for cluster in range(num_clusters):
            members = data[assignments == cluster]
            if members.shape[0] > 0:
                codebook[cluster] = members.mean(axis=0)
    return codebook, assignments
