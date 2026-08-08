"""Single-channel denoising algorithms."""

from __future__ import annotations

import numpy as np

from roomkit.denoise.noise_estimate import estimate_noise
from roomkit.denoise.spectral import spectral_subtraction
from roomkit.denoise.wiener import wiener_gain
from roomkit.signal.stft import istft, stft

__all__ = ["denoise", "estimate_noise", "spectral_subtraction", "wiener_gain"]

_METHODS = ("wiener", "spectral")


def denoise(
    samples: np.ndarray,
    method: str = "wiener",
    n_fft: int = 400,
    hop_length: int = 160,
    noise_frames: int = 10,
) -> np.ndarray:
    """Denoise a waveform with STFT-domain filtering.

    The noise power is estimated from the first ``noise_frames`` frames, so
    the recording should start with a short noise-only lead-in (a few tens of
    milliseconds is enough).

    Args:
        samples: One-dimensional noisy signal.
        method: ``wiener`` (posterior Wiener gain) or ``spectral``
            (spectral subtraction).
        n_fft: FFT size.
        hop_length: Hop between frames in samples.
        noise_frames: Number of leading frames treated as noise only.

    Returns:
        Enhanced waveform with the same length as ``samples``.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {_METHODS}")
    samples = np.asarray(samples, dtype=np.float64)
    spec = stft(samples, n_fft=n_fft, hop_length=hop_length)
    # lead-in frames are noise-only by assumption, so their average power is
    # an unbiased noise floor; the minimum-statistics tracker in
    # estimate_noise is better suited to signals with speech pauses
    power = np.abs(spec) ** 2
    lead_in = power[: max(noise_frames, 1)].mean(axis=0)
    noise_power = np.broadcast_to(lead_in, power.shape)
    if method == "wiener":
        gain = wiener_gain(np.abs(spec), noise_power)
        enhanced = spec * gain
    else:
        enhanced = spectral_subtraction(spec, noise_power)
    return istft(enhanced, hop_length=hop_length, length=samples.shape[0])
