"""Classical reverberation-time formulas."""

from __future__ import annotations

import math


def sabine_rt60(volume: float, surface_area: float, absorption: float) -> float:
    """Reverberation time from Sabine's formula.

    Args:
        volume: Room volume in m^3.
        surface_area: Total surface area in m^2.
        absorption: Average absorption coefficient in (0, 1].

    Returns:
        RT60 in seconds.
    """
    if volume <= 0 or surface_area <= 0:
        raise ValueError("volume and surface_area must be positive")
    if not 0 < absorption <= 1:
        raise ValueError("absorption must be in (0, 1]")
    return 0.161 * volume / (surface_area * absorption)


def eyring_rt60(volume: float, surface_area: float, absorption: float) -> float:
    """Reverberation time from Eyring's formula.

    Tends to be more accurate than Sabine for highly absorptive rooms.

    Args:
        volume: Room volume in m^3.
        surface_area: Total surface area in m^2.
        absorption: Average absorption coefficient in (0, 1).

    Returns:
        RT60 in seconds.
    """
    if volume <= 0 or surface_area <= 0:
        raise ValueError("volume and surface_area must be positive")
    if not 0 < absorption < 1:
        raise ValueError("absorption must be in (0, 1)")
    return 0.161 * volume / (-surface_area * math.log(1.0 - absorption))


def absorption_from_rt60(volume: float, surface_area: float, rt60: float) -> float:
    """Average absorption coefficient implied by a target RT60 (inverted Sabine).

    Args:
        volume: Room volume in m^3.
        surface_area: Total surface area in m^2.
        rt60: Desired reverberation time in seconds.

    Returns:
        The absorption coefficient in (0, 1].
    """
    if volume <= 0 or surface_area <= 0:
        raise ValueError("volume and surface_area must be positive")
    if rt60 <= 0:
        raise ValueError("rt60 must be positive")
    alpha = 0.161 * volume / (surface_area * rt60)
    if not 0 < alpha <= 1:
        raise ValueError(f"target rt60={rt60}s is not achievable in this room")
    return alpha
