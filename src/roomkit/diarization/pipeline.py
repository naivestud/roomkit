"""End-to-end speaker diarization."""

from __future__ import annotations

import numpy as np

from roomkit.diarization.clustering import cluster_embeddings
from roomkit.diarization.embeddings import spectral_embeddings
from roomkit.diarization.segmentation import segment_embeddings
from roomkit.types import Segment


class DiarizationPipeline:
    """Segment-embed-cluster diarization over spectral embeddings.

    The signal is embedded in sliding windows, change points split the
    sequence into homogeneous runs, each run is summarised by its mean
    embedding and agglomerative clustering assigns speaker labels.

    Args:
        sample_rate: Sampling rate of the input signal.
        frame_length: STFT frame length in samples.
        hop_length: STFT hop in samples.
        window_frames: Embedding window length in STFT frames.
        window_hop: Embedding window hop in STFT frames.
        threshold: Change-point cosine distance.
        min_segment: Minimum segment length in windows.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 400,
        hop_length: int = 160,
        window_frames: int = 25,
        window_hop: int = 10,
        threshold: float = 0.35,
        min_segment: int = 3,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window_frames = window_frames
        self.window_hop = window_hop
        self.threshold = threshold
        self.min_segment = min_segment

    def __call__(
        self, samples: np.ndarray, num_speakers: int | None = None
    ) -> list[Segment]:
        """Diarize a waveform and return speaker segments in seconds."""
        samples = np.asarray(samples, dtype=np.float64)
        embeddings, window_starts = spectral_embeddings(
            samples,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window_frames=self.window_frames,
            window_hop=self.window_hop,
        )
        if embeddings.shape[0] == 0:
            return []
        runs = segment_embeddings(embeddings, self.threshold, self.min_segment)
        means = np.stack([embeddings[start:end].mean(axis=0) for start, end in runs])
        labels = cluster_embeddings(means, num_speakers=num_speakers)

        frame_duration = self.hop_length / self.sample_rate
        window_duration = self.window_frames * frame_duration
        merged: list[Segment] = []
        for (start, end), label in zip(runs, labels, strict=True):
            start_time = float(window_starts[start]) * frame_duration
            end_time = float(window_starts[end - 1]) * frame_duration + window_duration
            speaker = f"speaker_{label}"
            if merged and merged[-1].label == speaker and start_time <= merged[-1].end:
                merged[-1] = Segment(merged[-1].start, end_time, speaker)
            else:
                merged.append(Segment(start_time, end_time, speaker))
        return merged
