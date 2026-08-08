"""Short-time Fourier transform."""

from __future__ import annotations

import numpy as np

from roomkit.audio.framing import frame_signal
from roomkit.signal.windows import get_window


def stft(
    samples: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
) -> np.ndarray:
    """Compute the short-time Fourier transform of a mono signal.

    Args:
        samples: One-dimensional input signal.
        n_fft: FFT size; also the analysis frame length.
        hop_length: Hop between frames in samples.
        window: Window name, see :func:`roomkit.signal.windows.get_window`.

    Returns:
        Complex array of shape ``(num_frames, n_fft // 2 + 1)``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(f"expected a mono signal, got shape {samples.shape}")
    win = get_window(window, n_fft)
    frames = frame_signal(samples, n_fft, hop_length)
    return np.fft.rfft(frames * win, n=n_fft, axis=-1)
