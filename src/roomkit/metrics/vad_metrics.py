"""Metrics for voice activity detection."""

from __future__ import annotations

import numpy as np


def vad_frame_metrics(
    reference: np.ndarray, estimate: np.ndarray
) -> dict[str, float]:
    """Frame-level precision, recall and F1 for VAD decisions.

    Args:
        reference: Ground-truth boolean decisions, one per frame.
        estimate: Predicted boolean decisions of the same length.

    Returns:
        Dictionary with ``precision``, ``recall``, ``f1`` and ``accuracy``.
    """
    reference = np.asarray(reference, dtype=bool)
    estimate = np.asarray(estimate, dtype=bool)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same length")
    true_positive = int(np.sum(reference & estimate))
    false_positive = int(np.sum(~reference & estimate))
    false_negative = int(np.sum(reference & ~estimate))
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive
    ) else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative
    ) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = float(np.mean(reference == estimate))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": accuracy,
    }
