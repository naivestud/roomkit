"""Wiener filtering."""

from __future__ import annotations

import numpy as np


def wiener_gain(
    magnitudes: np.ndarray,
    noise_power: np.ndarray,
    gain_floor: float = 0.0,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Posterior Wiener gain from noisy magnitudes and a noise power estimate.

    ``G = max(1 - noise / noisy, gain_floor)`` applied to power spectra.

    Args:
        magnitudes: Noisy magnitude spectrogram ``(num_frames, num_bins)``.
        noise_power: Noise power estimate with the same shape.
        gain_floor: Lower bound for the gain, useful to avoid musical noise.
        smoothing: Reserved for future temporal smoothing of the gain.

    Returns:
        Real gain array in ``[gain_floor, 1]``.
    """
    if not 0 <= gain_floor < 1:
        raise ValueError("gain_floor must be in [0, 1)")
    power = np.asarray(magnitudes, dtype=np.float64) ** 2
    gain = 1.0 - noise_power / np.maximum(power, 1e-12)
    return np.clip(gain, gain_floor, 1.0)
