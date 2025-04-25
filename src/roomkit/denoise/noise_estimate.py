"""Noise power estimation from magnitude spectra."""

from __future__ import annotations

import numpy as np


def estimate_noise(
    magnitudes: np.ndarray,
    initial_frames: int = 10,
    window: int = 30,
) -> np.ndarray:
    """Estimate the noise power spectrum with a windowed minimum tracker.

    The power of each frame is estimated as the minimum power seen in the
    ``window`` frames up to and including it, capped by the average of the
    first ``initial_frames`` frames (assumed noise-only). The cap keeps
    continuous speech from contaminating the noise floor.

    This is minimum statistics without bias compensation: the estimate is
    slightly low on average, which is a useful bias for denoising.

    Args:
        magnitudes: Magnitude spectrogram of shape ``(num_frames, num_bins)``.
        initial_frames: Number of frames averaged for the initial estimate.
        window: Number of frames in the minimum search window.

    Returns:
        Noise power estimate with the same shape as ``magnitudes``.
    """
    power = np.asarray(magnitudes, dtype=np.float64) ** 2
    if power.ndim != 2:
        raise ValueError(f"expected a 2-D spectrogram, got shape {power.shape}")
    if initial_frames <= 0 or window <= 0:
        raise ValueError("initial_frames and window must be positive")

    num_frames = power.shape[0]
    seed = power[: min(initial_frames, num_frames)].mean(axis=0)
    noise = np.empty_like(power)
    for t in range(num_frames):
        lo = max(0, t + 1 - window)
        noise[t] = power[lo : t + 1].min(axis=0)
    # Cap at the lead-in level: the tracker may dip below it inside speech
    # pauses but must never rise above the opening noise-only estimate,
    # otherwise continuous speech contaminates the noise floor.
    noise = np.minimum(noise, seed)
    return noise
