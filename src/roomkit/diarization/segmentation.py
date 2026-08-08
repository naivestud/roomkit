"""Speaker change point segmentation."""

from __future__ import annotations

import numpy as np


def _cosine_steps(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between each embedding and its successor."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    similarity = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def segment_embeddings(
    embeddings: np.ndarray,
    threshold: float = 0.06,
    min_segment: int = 3,
) -> list[tuple[int, int]]:
    """Split an embedding sequence at cosine-distance jumps.

    Consecutive embeddings with a distance above ``threshold`` mark a
    speaker change; segments shorter than ``min_segment`` windows are merged
    into their predecessor.

    Args:
        embeddings: Window embeddings of shape ``(num_windows, dim)``.
        threshold: Cosine distance that marks a change point.
        min_segment: Minimum segment length in windows.

    Returns:
        List of half-open ``[start, end)`` window ranges covering the input.
    """
    num_windows = embeddings.shape[0]
    if num_windows == 0:
        return []
    steps = _cosine_steps(embeddings)
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(steps > threshold)] + [num_windows]
    segments = list(zip(cuts[:-1], cuts[1:], strict=True))
    merged: list[list[int]] = []
    for start, end in segments:
        if merged and end - start < min_segment:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
