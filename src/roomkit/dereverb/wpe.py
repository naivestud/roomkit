"""Weighted prediction error (WPE) dereverberation.

WPE models the reverberant tail of each frequency bin as a multichannel
linear prediction of delayed past frames and subtracts the prediction. See
Yoshioka et al., "Generalization of multi-channel linear prediction methods
for blind MIMO impulse response shortening", 2012.
"""

from __future__ import annotations

import numpy as np


def _estimate_power(spectrum: np.ndarray, context: int = 0) -> np.ndarray:
    """Smoothed power spectral density of the current source estimate.

    Args:
        spectrum: Complex spectrogram of shape ``(freqs, channels, frames)``.
        context: Number of neighbouring frames averaged on each side of the
            current one (0 disables smoothing).

    Returns:
        Real power array of shape ``(freqs, frames)``.
    """
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    if context > 0:
        kernel = np.ones(2 * context + 1)
        power = np.array(
            [np.convolve(row, kernel, mode="same") / kernel.sum() for row in power]
        )
    return power


def _estimate_filter(
    spectrum: np.ndarray,
    power: np.ndarray,
    taps: int,
    delay: int,
    regularization: float,
) -> np.ndarray:
    """Estimate the prediction filter for each frequency bin.

    Solves the weighted normal equations ``R G = P`` where the regressor is
    the stack of delayed past frames and the weights are the inverse power of
    the current source estimate.

    Args:
        spectrum: Complex spectrogram ``(freqs, channels, frames)``.
        power: Source power estimate ``(freqs, frames)``.
        taps: Number of filter taps.
        delay: Prediction delay (guard interval) in frames.
        regularization: Diagonal loading added before solving.

    Returns:
        Filter tensor of shape ``(freqs, taps * channels, channels)``.
    """
    freqs, channels, frames = spectrum.shape
    inv_power = 1.0 / np.maximum(power, 1e-10)

    padded = np.pad(spectrum, ((0, 0), (0, 0), (delay + taps - 1, 0)))
    blocks = [padded[:, :, taps - 1 - k : taps - 1 - k + frames] for k in range(taps)]
    y_tilde = np.stack(blocks, axis=2).reshape(freqs, taps * channels, frames)

    correlation = np.einsum(
        "fkt,ft,flt->fkl", y_tilde.conj(), inv_power, y_tilde, optimize="optimal"
    )
    cross = np.einsum(
        "fkt,ft,fdt->fkd", y_tilde.conj(), inv_power, spectrum, optimize="optimal"
    )
    correlation += regularization * np.eye(taps * channels)[None, :, :]
    return np.linalg.solve(correlation, cross)
