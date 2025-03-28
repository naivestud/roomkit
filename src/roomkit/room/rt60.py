"""Reverberation time estimation via Schroeder backward integration."""

from __future__ import annotations

import numpy as np


def schroeder_integration(
    impulse_response: np.ndarray, noise_floor: float | None = None
) -> np.ndarray:
    """Backward energy integration of an impulse response (Schroeder curve).

    The curve starts at the total signal energy and decays toward zero; it is
    returned in dB, normalised to 0 dB at the start.

    Args:
        impulse_response: One-dimensional impulse response.
        noise_floor: Optional noise energy level to subtract before
            integration. Measured responses flatten out once the decaying
            tail drops below the background noise; subtracting the floor
            keeps the curve decaying. ``None`` disables the correction.

    Returns:
        Decay curve in dB with shape ``(len(impulse_response),)``.
    """
    response = np.asarray(impulse_response, dtype=np.float64)
    if response.ndim != 1 or response.shape[0] == 0:
        raise ValueError("expected a non-empty 1-D impulse response")
    energy = response**2
    if noise_floor is not None:
        if noise_floor < 0:
            raise ValueError("noise_floor must be non-negative")
        energy = np.maximum(energy - noise_floor, 0.0)
    integrated = np.cumsum(energy[::-1])[::-1]
    if integrated[0] <= 0:
        raise ValueError("impulse response has no energy above the noise floor")
    curve = 10.0 * np.log10(np.maximum(integrated, 1e-12 * integrated[0]))
    return curve - curve[0]


def estimate_rt60(
    impulse_response: np.ndarray,
    sample_rate: int,
    lower_db: float = -5.0,
    upper_db: float = -35.0,
    noise_floor: float | None = None,
) -> float:
    """Estimate RT60 from a measured impulse response.

    Fits a line to the Schroeder curve between ``lower_db`` and ``upper_db``
    (a T30-style regression) and extrapolates to the -60 dB crossing.

    Args:
        impulse_response: One-dimensional impulse response.
        sample_rate: Sampling rate of the impulse response in Hz.
        lower_db: Start of the regression window in dB (relative to peak).
        upper_db: End of the regression window in dB.
        noise_floor: Optional noise energy level passed through to
            :func:`schroeder_integration`.

    Returns:
        Estimated RT60 in seconds.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not lower_db > upper_db:
        raise ValueError("lower_db must be greater than upper_db")
    curve = schroeder_integration(impulse_response, noise_floor=noise_floor)
    times = np.arange(curve.shape[0]) / sample_rate
    window = (curve <= lower_db) & (curve >= upper_db)
    if window.sum() < 2:
        raise ValueError("decay range too short for rt60 estimation")
    slope, intercept = np.polyfit(times[window], curve[window], 1)
    if slope >= 0:
        raise ValueError("impulse response does not decay")
    return float((-60.0 - intercept) / slope)
