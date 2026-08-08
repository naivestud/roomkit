"""Sample-rate conversion utilities."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample ``samples`` from ``source_rate`` to ``target_rate``.

    Uses polyphase filtering (``scipy.signal.resample_poly``), which preserves
    speech waveforms better than FFT resampling.

    Args:
        samples: Mono ``(num_samples,)`` or multi-channel ``(num_samples, channels)``.
        source_rate: Original sampling rate in Hz.
        target_rate: Desired sampling rate in Hz.

    Returns:
        Resampled audio with the same number of channels.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate:
        return np.asarray(samples)
    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return signal.resample_poly(np.asarray(samples), up, down, axis=0)
