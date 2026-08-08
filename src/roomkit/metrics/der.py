"""Diarization error rate."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from roomkit.types import Segment

_FRAME = 0.01  # scoring resolution in seconds


def _segments_to_frames(segments: list[Segment], duration: float) -> np.ndarray:
    """Rasterise labelled segments into per-frame labels (-1 = silence)."""
    num_frames = int(np.ceil(duration / _FRAME))
    labels = np.full(num_frames, -1, dtype=np.int64)
    speaker_ids: dict[str, int] = {}
    for segment in segments:
        if segment.label not in speaker_ids:
            speaker_ids[segment.label] = len(speaker_ids)
        lo = max(0, int(np.floor(segment.start / _FRAME)))
        hi = min(num_frames, int(np.ceil(segment.end / _FRAME)))
        labels[lo:hi] = speaker_ids[segment.label]
    return labels


def diarization_error_rate(
    reference: list[Segment],
    estimate: list[Segment],
    duration: float,
    collar: float = 0.25,
) -> float:
    """Diarization error rate (DER) between two sets of segments.

    Both hypotheses are rasterised to 10 ms frames, speakers are aligned
    with the Hungarian algorithm to minimise confusion, and frames within
    ``collar`` of a reference speaker boundary are excluded from scoring.

    DER = (missed speech + false alarms + speaker confusion) / reference
    speech, expressed as a fraction (multiply by 100 for percent).

    Args:
        reference: Ground-truth speaker segments.
        estimate: Predicted speaker segments.
        duration: Total scored duration in seconds.
        collar: No-score collar around reference boundaries, in seconds.

    Returns:
        DER as a fraction; can exceed 1.0 for very poor hypotheses.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    ref = _segments_to_frames(reference, duration)
    hyp = _segments_to_frames(estimate, duration)

    # no-score collar around reference boundaries (including speaker changes)
    scored = np.ones(ref.shape[0], dtype=bool)
    if collar > 0:
        collar_frames = int(np.round(collar / _FRAME))
        boundaries = np.flatnonzero(np.diff(ref) != 0)
        for boundary in boundaries:
            lo = max(0, boundary - collar_frames)
            hi = min(ref.shape[0], boundary + collar_frames)
            scored[lo:hi] = False

    ref_active = ref != -1
    hyp_active = hyp != -1
    total_reference = int(np.sum(ref_active & scored))
    if total_reference == 0:
        return 0.0 if not np.any(hyp_active & scored) else 1.0

    miss = int(np.sum(ref_active & ~hyp_active & scored))
    false_alarm = int(np.sum(~ref_active & hyp_active & scored))

    # confusion: align speakers to minimise disagreements on doubly-active frames
    both = ref_active & hyp_active & scored
    confusion = 0
    if both.any():
        ref_ids = np.unique(ref[both])
        hyp_ids = np.unique(hyp[both])
        cost = np.zeros((ref_ids.shape[0], hyp_ids.shape[0]))
        for i, ref_id in enumerate(ref_ids):
            for j, hyp_id in enumerate(hyp_ids):
                cost[i, j] = -int(np.sum((ref[both] == ref_id) & (hyp[both] == hyp_id)))
        row, col = linear_sum_assignment(cost)
        matched = int(-cost[row, col].sum())
        confusion = int(both.sum()) - matched

    return (miss + false_alarm + confusion) / total_reference
