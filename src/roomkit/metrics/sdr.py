"""Signal-to-distortion metrics."""

from __future__ import annotations

import numpy as np


def _validate_pair(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("reference and estimate must be 1-D")
    if reference.shape != estimate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs estimate {estimate.shape}"
        )
    return reference, estimate


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-distortion ratio in dB.

    The estimate is projected onto the reference and everything orthogonal
    counts as distortion; no compensation for scaling or delay.

    Args:
        reference: Clean target signal.
        estimate: Enhanced estimate of the same length.

    Returns:
        SDR in dB.
    """
    reference, estimate = _validate_pair(reference, estimate)
    reference_energy = float(np.dot(reference, reference))
    if reference_energy == 0:
        raise ValueError("reference must have non-zero energy")
    projection = float(np.dot(estimate, reference)) / reference_energy * reference
    distortion = estimate - projection
    distortion_energy = float(np.dot(distortion, distortion))
    if distortion_energy == 0:
        return float("inf")
    return float(10.0 * np.log10(reference_energy / distortion_energy))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB.

    Both signals are zero-mean before the projection, making the metric
    insensitive to global gain errors (Le Roux et al., 2019).

    Args:
        reference: Clean target signal.
        estimate: Enhanced estimate of the same length.

    Returns:
        SI-SDR in dB.
    """
    reference = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    estimate = np.asarray(estimate, dtype=np.float64) - np.mean(estimate)
    return sdr(reference, estimate)
