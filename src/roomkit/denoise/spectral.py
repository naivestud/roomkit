"""Spectral subtraction denoising."""

from __future__ import annotations

import numpy as np


def spectral_subtraction(noisy: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
    """Enhance a noisy STFT by subtracting the noise power spectrum.

    Args:
        noisy: Complex noisy spectrogram of shape ``(num_frames, num_bins)``.
        noise_power: Noise power estimate with the same shape.

    Returns:
        Enhanced complex spectrogram.
    """
    power = np.abs(noisy) ** 2
    gain = np.sqrt(np.maximum(power - noise_power, 0.0) / np.maximum(power, 1e-12))
    return noisy * gain
