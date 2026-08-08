"""Post-processing of voice activity decisions."""

from __future__ import annotations

import numpy as np


def hangover(decisions: np.ndarray, hangover_frames: int = 5) -> np.ndarray:
    """Extend each speech region by a hangover tail.

    Short energy dips inside speech are bridged, which reduces chopping of
    plosives and unvoiced endings.

    Args:
        decisions: Boolean frame decisions.
        hangover_frames: Number of frames to keep marking speech after the
            last speech frame.

    Returns:
        Smoothed boolean decisions.
    """
    if hangover_frames < 0:
        raise ValueError("hangover_frames must be non-negative")
    decisions = np.asarray(decisions, dtype=bool)
    smoothed = decisions.copy()
    counter = 0
    for index, active in enumerate(decisions):
        if active:
            counter = hangover_frames
        elif counter > 0:
            smoothed[index] = True
            counter -= 1
    return smoothed
