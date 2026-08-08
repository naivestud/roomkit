import numpy as np
import pytest

from roomkit.vad import detect_voice, hangover
from roomkit.vad.smoother import decisions_to_segments


def _alternating_signal(fs: int = 16000):
    """silence / tone / silence / tone / silence"""
    rng = np.random.default_rng(0)

    def silence(duration):
        return rng.standard_normal(int(duration * fs)) * 1e-3

    def tone(duration, freq=300.0):
        t = np.arange(int(duration * fs)) / fs
        return 0.3 * np.sin(2 * np.pi * freq * t)

    parts = [silence(0.3), tone(0.5), silence(0.3), tone(0.4, 500.0), silence(0.2)]
    return np.concatenate(parts), fs


def test_detects_two_bursts():
    samples, fs = _alternating_signal()
    segments = detect_voice(samples, sample_rate=fs)
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.3, abs=0.1)
    assert segments[0].end == pytest.approx(0.8, abs=0.15)
    assert segments[1].start == pytest.approx(1.1, abs=0.1)
    assert segments[1].end == pytest.approx(1.5, abs=0.15)


def test_silence_has_no_segments():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(16000) * 1e-3
    assert detect_voice(samples) == []


def test_hangover_bridges_gaps():
    decisions = np.array([True, True, False, False, False, True])
    out = hangover(decisions, hangover_frames=2)
    assert out.tolist() == [True, True, True, True, False, True]


def test_segments_from_decisions():
    decisions = np.array([False, True, True, True, False, False, True, False])
    segments = decisions_to_segments(decisions, sample_rate=16000, hop_length=160)
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.01)
    assert segments[0].end == pytest.approx(0.04)
    assert segments[1].label == "speech"
