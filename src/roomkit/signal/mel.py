"""Mel spectrogram features."""

from __future__ import annotations

import numpy as np

from roomkit.signal.stft import stft


def hz_to_mel(freqs: np.ndarray | float) -> np.ndarray:
    """Convert frequencies in Hz to the mel scale (O'Shaughnessy formula)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freqs / 700.0)


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    """Convert mel-scale values back to Hz."""
    mels = np.asarray(mels, dtype=np.float64)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Build a mel filterbank matrix of shape ``(n_mels, n_fft // 2 + 1)``.

    Triangular filters spaced uniformly on the mel scale. Each filter is
    area-normalised (Slaney style) so that filter outputs are comparable
    across frequency.

    Args:
        n_mels: Number of mel bands.
        n_fft: FFT size used during analysis.
        sample_rate: Sampling rate in Hz.
        f_min: Lowest band edge in Hz.
        f_max: Highest band edge in Hz; defaults to the Nyquist frequency.

    Returns:
        Non-negative filterbank matrix.
    """
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if f_max is None:
        f_max = sample_rate / 2
    if not 0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError(f"invalid frequency range [{f_min}, {f_max}]")

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center > left:
            fb[m, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m, center:right] = (right - np.arange(center, right)) / (right - center)
        bandwidth = points[m + 2] - points[m]
        if bandwidth > 0:
            fb[m] *= 2.0 / bandwidth
    return fb


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
) -> np.ndarray:
    """Compute a log-mel spectrogram following the whisper scaling recipe.

    The magnitude spectrum is mapped through a mel filterbank, converted with
    ``log10`` (clamped at ``1e-10`` and to within 8 decades of the maximum),
    then rescaled to roughly ``[-1, 1]`` with ``(x + 4) / 4``.

    Args:
        samples: One-dimensional input signal sampled at ``sample_rate``.
        sample_rate: Sampling rate in Hz.
        n_mels: Number of mel bands.
        n_fft: FFT size.
        hop_length: Hop between frames in samples.
        window: Analysis window name.

    Returns:
        Array of shape ``(num_frames, n_mels)``.
    """
    spec = stft(samples, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitudes = np.abs(spec)
    fb = mel_filterbank(n_mels, n_fft, sample_rate)
    mel = magnitudes @ fb.T
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0
