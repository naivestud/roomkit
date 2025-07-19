"""Agglomerative speaker clustering."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def cluster_embeddings(
    embeddings: np.ndarray,
    num_speakers: int | None = None,
    distance_threshold: float = 0.5,
) -> np.ndarray:
    """Cluster embeddings with average-linkage agglomerative clustering.

    Cosine distance is used, which pairs naturally with L2-normalised
    embeddings. Either a fixed number of clusters or a distance threshold
    determines the partition.

    Args:
        embeddings: One embedding per row.
        num_speakers: If given, cut the dendrogram into exactly this many
            clusters.
        distance_threshold: Used when ``num_speakers`` is None; embeddings
            closer than this end up in the same cluster.

    Returns:
        Zero-based cluster labels, one per embedding.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {embeddings.shape}")
    num_rows = embeddings.shape[0]
    if num_rows == 0:
        return np.empty(0, dtype=np.int64)
    if num_rows == 1:
        return np.zeros(1, dtype=np.int64)
    if num_speakers is not None and not 0 < num_speakers <= num_rows:
        raise ValueError("num_speakers must be between 1 and the number of embeddings")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    distances = np.clip(pdist(normed, metric="cosine"), 0.0, None)
    tree = linkage(distances, method="average")
    if num_speakers is not None:
        labels = fcluster(tree, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(tree, t=distance_threshold, criterion="distance")
    return labels - 1
