"""Dereverberation algorithms."""

from __future__ import annotations

import numpy as np

from roomkit.dereverb.wpe import wpe
from roomkit.signal.stft import istft, stft

__all__ = ["dereverb", "wpe"]


def dereverb(
    samples: np.ndarray,
    taps: int = 8,
    delay: int = 3,
    iterations: int = 3,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Dereverberate a mono waveform with WPE.

    Args:
        samples: One-dimensional reverberant signal.
        taps: Number of prediction filter taps.
        delay: Prediction delay in frames.
        iterations: Number of WPE iterations.
        n_fft: FFT size.
        hop_length: Hop between frames in samples.

    Returns:
        Dereverberated waveform with the same length as ``samples``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    spec = stft(samples, n_fft=n_fft, hop_length=hop_length)
    # wpe expects (freqs, channels, frames); mono adds a size-1 channel axis
    spectrum = np.transpose(spec)[:, None, :]
    enhanced = wpe(spectrum, taps=taps, delay=delay, iterations=iterations)
    enhanced_spec = np.transpose(enhanced[:, 0, :])
    return istft(enhanced_spec, hop_length=hop_length, length=samples.shape[0])
