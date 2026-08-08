"""Griffin-Lim phase reconstruction from magnitude spectra."""

from __future__ import annotations

import numpy as np

from roomkit.signal.stft import istft, stft


def griffin_lim(
    magnitude: np.ndarray,
    hop_length: int = 160,
    window: str = "hann",
    n_iter: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Reconstruct a waveform from STFT magnitudes with Griffin-Lim iteration.

    Alternates between the time domain (overlap-add) and the frequency domain,
    keeping the target magnitudes and discarding the estimated phase each
    round. Converges to a local optimum of the spectral error.

    Args:
        magnitude: Non-negative array of shape ``(num_frames, n_fft // 2 + 1)``.
        hop_length: Hop between frames in samples.
        window: Analysis window name.
        n_iter: Number of iterations.
        seed: Seed for the initial random phase.

    Returns:
        Time-domain signal consistent with the given magnitudes.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if np.any(magnitude < 0):
        raise ValueError("magnitude must be non-negative")
    n_fft = (magnitude.shape[-1] - 1) * 2

    rng = np.random.default_rng(seed)
    angles = np.exp(2j * np.pi * rng.random(magnitude.shape))
    for _ in range(n_iter):
        signal = istft(magnitude * angles, hop_length=hop_length, window=window)
        spec = stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
        spec = spec[: magnitude.shape[0]]
        angles = np.exp(1j * np.angle(spec))
    return istft(magnitude * angles, hop_length=hop_length, window=window)
