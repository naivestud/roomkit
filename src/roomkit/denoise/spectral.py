"""Spectral subtraction denoising."""

from __future__ import annotations

import numpy as np


def spectral_subtraction(
    noisy: np.ndarray,
    noise_power: np.ndarray,
    oversubtraction: float = 1.0,
    spectral_floor: float = 0.0,
) -> np.ndarray:
    """Enhance a noisy STFT by subtracting the noise power spectrum.

    Args:
        noisy: Complex noisy spectrogram of shape ``(num_frames, num_bins)``.
        noise_power: Noise power estimate with the same shape.
        oversubtraction: Factor applied to the noise estimate before
            subtraction. Values above 1 remove more noise at the cost of
            speech distortion; typical range is 1.0 to 2.0.
        spectral_floor: Fraction of the noisy power kept as a lower bound on
            the gain. A small floor (e.g. 0.01) suppresses musical noise.

    Returns:
        Enhanced complex spectrogram.
    """
    if oversubtraction <= 0:
        raise ValueError("oversubtraction must be positive")
    if not 0 <= spectral_floor < 1:
        raise ValueError("spectral_floor must be in [0, 1)")
    power = np.abs(noisy) ** 2
    subtracted = power - oversubtraction * noise_power
    floored = np.maximum(subtracted, spectral_floor * power)
    gain = np.sqrt(floored / np.maximum(power, 1e-12))
    return noisy * gain
