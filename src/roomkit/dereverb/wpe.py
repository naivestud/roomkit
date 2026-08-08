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
