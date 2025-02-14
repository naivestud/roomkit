import numpy as np
import pytest

from roomkit.audio.resample import resample


def test_same_rate_is_identity():
    samples = np.arange(100, dtype=np.float64)
    out = resample(samples, 16000, 16000)
    assert np.array_equal(out, samples)


def test_length_scales_with_rate():
    samples = np.random.default_rng(0).standard_normal(16000)
    out = resample(samples, 16000, 8000)
    assert out.shape[0] == 8000


def test_tone_frequency_preserved():
    rate = 16000
    freq = 440.0
    t = np.arange(rate) / rate
    samples = np.sin(2 * np.pi * freq * t)
    out = resample(samples, rate, 8000)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(out.shape[0])))
    freqs = np.fft.rfftfreq(out.shape[0], d=1 / 8000)
    peak = freqs[np.argmax(spectrum)]
    assert abs(peak - freq) < 10


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        resample(np.zeros(10), 0, 16000)
