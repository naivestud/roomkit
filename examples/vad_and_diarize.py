"""Voice activity detection and speaker diarization on synthetic audio.

Builds a signal where two "speakers" (distinct harmonic stacks) alternate,
runs VAD and diarization, and scores the result against ground truth.

    python examples/vad_and_diarize.py
"""

from __future__ import annotations

import numpy as np

from roomkit.diarization import diarize
from roomkit.metrics.der import diarization_error_rate
from roomkit.types import Segment
from roomkit.vad import detect_voice


def make_conversation(sample_rate: int = 16000) -> tuple[np.ndarray, list[Segment]]:
    """Noise lead-in, then three one-second turns: speaker A, speaker B, speaker A."""

    def harmonic_stack(f0: float, tilt: float) -> np.ndarray:
        t = np.arange(sample_rate) / sample_rate
        signal = np.zeros(sample_rate)
        for k in range(1, 9):
            signal += np.sin(2 * np.pi * f0 * k * t) / k**tilt
        return signal / np.max(np.abs(signal)) * 0.4

    speaker_a = harmonic_stack(160.0, tilt=0.8)
    speaker_b = harmonic_stack(420.0, tilt=1.6)
    lead_in = 0.2  # seconds of low noise so the VAD floor has something to latch onto
    noise = np.random.default_rng(0).standard_normal(int(lead_in * sample_rate)) * 1e-3
    samples = np.concatenate([noise, speaker_a, speaker_b, speaker_a])
    truth = [
        Segment(lead_in, lead_in + 1.0, "speaker_a"),
        Segment(lead_in + 1.0, lead_in + 2.0, "speaker_b"),
        Segment(lead_in + 2.0, lead_in + 3.0, "speaker_a"),
    ]
    return samples, truth


def main() -> None:
    samples, truth = make_conversation()
    fs = 16000
    duration = float(len(samples)) / fs

    segments = detect_voice(samples, sample_rate=fs)
    coverage = sum(segment.duration for segment in segments)
    print(f"VAD: {len(segments)} segments, {coverage:.2f} s of {duration:.2f} s marked speech")
    for segment in segments:
        print(f"  {segment.start:5.2f} - {segment.end:5.2f} s")

    hypothesis = diarize(samples, num_speakers=2, sample_rate=fs)
    print(f"\ndiarization: {len(hypothesis)} segments")
    for segment in hypothesis:
        print(f"  {segment.label}: {segment.start:5.2f} - {segment.end:5.2f} s")

    der = diarization_error_rate(truth, hypothesis, duration=duration, collar=0.25)
    print(f"\nDER (250 ms collar): {der * 100:.1f}%")


if __name__ == "__main__":
    main()
