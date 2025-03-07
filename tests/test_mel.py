import numpy as np
import pytest

from roomkit.signal.mel import hz_to_mel, mel_filterbank, mel_to_hz


def test_mel_hz_roundtrip():
    freqs = np.array([100.0, 440.0, 1000.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(freqs)), freqs)


def test_filterbank_shape():
    fb = mel_filterbank(n_mels=40, n_fft=400, sample_rate=16000)
    assert fb.shape == (40, 201)


def test_filterbank_non_negative():
    fb = mel_filterbank(n_mels=80, n_fft=400, sample_rate=16000)
    assert np.all(fb >= 0)


def test_filterbank_peaks_ordered():
    fb = mel_filterbank(n_mels=40, n_fft=512, sample_rate=16000)
    peaks = np.argmax(fb, axis=1)
    # band centres move monotonically up in frequency
    assert np.all(np.diff(peaks) >= 0)
    assert peaks[0] < peaks[-1]


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        mel_filterbank(n_mels=10, n_fft=400, sample_rate=16000, f_min=9000.0)
