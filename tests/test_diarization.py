import numpy as np

from roomkit.diarization.pipeline import DiarizationPipeline


def _two_speakers(fs: int = 16000):
    """Three one-second blocks: speaker A, speaker B, speaker A."""

    def harmonic_stack(f0: float, tilt: float):
        t = np.arange(fs) / fs
        signal = np.zeros(fs)
        for k in range(1, 9):
            signal += np.sin(2 * np.pi * f0 * k * t) / k**tilt
        return signal / np.max(np.abs(signal)) * 0.4

    speaker_a = harmonic_stack(160.0, tilt=0.8)
    speaker_b = harmonic_stack(420.0, tilt=1.6)
    return np.concatenate([speaker_a, speaker_b, speaker_a]), fs


def _dominant_label(segments, start, end):
    share: dict[str, float] = {}
    for segment in segments:
        overlap = min(segment.end, end) - max(segment.start, start)
        if overlap > 0:
            share[segment.label] = share.get(segment.label, 0.0) + overlap
    return max(share, key=share.get)


def test_two_speakers_separated():
    samples, fs = _two_speakers()
    pipeline = DiarizationPipeline(sample_rate=fs)
    segments = pipeline(samples, num_speakers=2)
    assert len(segments) >= 3
    first = _dominant_label(segments, 0.2, 0.9)
    second = _dominant_label(segments, 1.2, 1.9)
    third = _dominant_label(segments, 2.2, 2.9)
    # blocks one and three are the same speaker, block two is not
    assert first == third
    assert first != second


def test_segment_labels_are_speaker_ids():
    samples, fs = _two_speakers()
    segments = DiarizationPipeline(sample_rate=fs)(samples, num_speakers=2)
    labels = {segment.label for segment in segments}
    assert labels <= {"speaker_0", "speaker_1"}
    assert len(labels) == 2


def test_short_signal_returns_no_segments():
    pipeline = DiarizationPipeline()
    assert pipeline(np.zeros(400)) == []
