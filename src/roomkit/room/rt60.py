"""Reverberation time estimation via Schroeder backward integration."""

from __future__ import annotations

import numpy as np


def schroeder_integration(impulse_response: np.ndarray) -> np.ndarray:
    """Backward energy integration of an impulse response (Schroeder curve).

    The curve starts at the total signal energy and decays toward zero; it is
    returned in dB, normalised to 0 dB at the start.

    Args:
        impulse_response: One-dimensional impulse response.

    Returns:
        Decay curve in dB with shape ``(len(impulse_response),)``.
    """
    response = np.asarray(impulse_response, dtype=np.float64)
    if response.ndim != 1 or response.shape[0] == 0:
        raise ValueError("expected a non-empty 1-D impulse response")
    energy = response**2
    integrated = np.cumsum(energy[::-1])[::-1]
    curve = 10.0 * np.log10(np.maximum(integrated, 1e-12 * integrated[0]))
    return curve - curve[0]
