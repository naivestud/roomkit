"""Short-time Fourier transform."""

from __future__ import annotations

import numpy as np

from roomkit.audio.framing import frame_signal, overlap_add
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


def istft(
    frames: np.ndarray,
    hop_length: int = 160,
    window: str = "hann",
    length: int | None = None,
) -> np.ndarray:
    """Invert an STFT produced by :func:`stft` with overlap-add.

    The same window is used for analysis and synthesis; the output is
    normalised by the accumulated squared window, so reconstruction is exact
    in the interior of the signal for any window.

    Args:
        frames: Complex array of shape ``(num_frames, n_fft // 2 + 1)``.
        hop_length: Hop used during analysis.
        window: Window used during analysis.
        length: If given, trim or zero-pad the output to this many samples.

    Returns:
        Reconstructed time-domain signal (float64).
    """
    frames = np.asarray(frames, dtype=np.complex128)
    if frames.ndim != 2:
        raise ValueError(f"expected a 2-D spectrogram, got shape {frames.shape}")
    n_fft = (frames.shape[-1] - 1) * 2
    win = get_window(window, n_fft)
    time_frames = np.fft.irfft(frames, n=n_fft, axis=-1)
    out = overlap_add(time_frames, hop_length, window=win)
    if length is not None:
        pad = max(length - out.shape[0], 0)
        out = out[:length] if pad == 0 else np.pad(out, (0, pad))
    return out.astype(np.float64, copy=False)
