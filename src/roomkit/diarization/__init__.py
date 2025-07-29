"""Lightweight speaker diarization."""

from __future__ import annotations

import numpy as np

from roomkit.diarization.clustering import cluster_embeddings
from roomkit.diarization.embeddings import spectral_embeddings
from roomkit.diarization.pipeline import DiarizationPipeline
from roomkit.diarization.segmentation import segment_embeddings
from roomkit.types import Segment

__all__ = [
    "DiarizationPipeline",
    "cluster_embeddings",
    "diarize",
    "segment_embeddings",
    "spectral_embeddings",
]


def diarize(
    samples: np.ndarray,
    num_speakers: int | None = None,
    sample_rate: int = 16000,
) -> list[Segment]:
    """Diarize a waveform: answer "who spoke when?".

    Args:
        samples: One-dimensional input signal.
        num_speakers: If known, the exact number of speakers; otherwise the
            cluster count is decided from a distance threshold.
        sample_rate: Sampling rate in Hz.

    Returns:
        Speaker segments in seconds.
    """
    pipeline = DiarizationPipeline(sample_rate=sample_rate)
    return pipeline(samples, num_speakers=num_speakers)
