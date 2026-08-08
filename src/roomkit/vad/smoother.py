"""Post-processing of voice activity decisions."""

from __future__ import annotations

import numpy as np

from roomkit.types import Segment


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


def decisions_to_segments(
    decisions: np.ndarray,
    sample_rate: int,
    hop_length: int = 160,
    frame_length: int = 400,
    min_duration: float = 0.0,
    label: str = "speech",
) -> list[Segment]:
    """Convert boolean frame decisions into timed segments.

    Args:
        decisions: Boolean decisions, one per analysis frame.
        sample_rate: Sampling rate used during analysis.
        hop_length: Hop between frames in samples.
        frame_length: Analysis window length in samples.
        min_duration: Drop segments shorter than this many seconds.
        label: Label attached to every segment.

    Returns:
        Segments sorted by start time. The end time accounts for the last
        active frame's analysis window, so it marks where the speech energy
        actually stops rather than where the last frame starts.
    """
    decisions = np.asarray(decisions, dtype=bool)
    if decisions.size == 0:
        return []
    padded = np.concatenate([[False], decisions, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    frame_duration = hop_length / sample_rate
    window_duration = frame_length / sample_rate
    segments = []
    for start, end in zip(starts, ends, strict=True):
        duration = (end - start) * frame_duration
        if duration < min_duration:
            continue
        start_time = start * frame_duration
        # last active frame (end - 1) extends a full window past its start
        end_time = (end - 1) * frame_duration + window_duration
        segments.append(Segment(start_time, end_time, label))
    return segments
