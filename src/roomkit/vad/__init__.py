"""Voice activity detection."""

from __future__ import annotations

import numpy as np

from roomkit.types import Segment
from roomkit.vad.energy import energy_vad, frame_energy_db
from roomkit.vad.smoother import decisions_to_segments, hangover

__all__ = ["decisions_to_segments", "detect_voice", "energy_vad", "frame_energy_db", "hangover"]


def detect_voice(
    samples: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold_db: float = 6.0,
    noise_frames: int = 10,
    hangover_frames: int = 8,
    min_duration: float = 0.05,
) -> list[Segment]:
    """Detect voice activity in a waveform.

    Runs adaptive energy thresholding, bridges short dips with a hangover
    smoother and returns the surviving regions as timed segments.

    Args:
        samples: One-dimensional input signal.
        sample_rate: Sampling rate in Hz.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.
        threshold_db: Margin above the noise floor required to call speech.
        noise_frames: Leading frames used to estimate the noise floor.
        hangover_frames: Frames appended after each speech offset.
        min_duration: Drop segments shorter than this many seconds.

    Returns:
        Speech segments in seconds.
    """
    decisions = energy_vad(
        samples,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold_db=threshold_db,
        noise_frames=noise_frames,
    )
    smoothed = hangover(decisions, hangover_frames=hangover_frames)
    return decisions_to_segments(
        smoothed, sample_rate, hop_length=hop_length, min_duration=min_duration
    )
