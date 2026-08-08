import numpy as np
import pytest
from scipy.signal import fftconvolve

from roomkit.dereverb import dereverb
from roomkit.dereverb.wpe import wpe


def _reverb_signal(seed: int = 0, fs: int = 16000):
    """Noise burst followed by a silent tail, convolved with a decaying RIR."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(int(1.0 * fs)) * 0.3
    n = int(0.6 * fs)
    t = np.arange(n) / fs
    rir = rng.standard_normal(n) * 10.0 ** (-3.0 * t / 0.4)
    rir[0] = 1.0
    reverberant = fftconvolve(source, rir)[: len(source) + int(0.5 * fs)]
    return reverberant, fs


def test_tail_energy_reduced():
    reverberant, fs = _reverb_signal()
    enhanced = dereverb(reverberant, taps=10, delay=3, iterations=5)
    assert enhanced.shape == reverberant.shape
    assert np.all(np.isfinite(enhanced))
    tail = slice(int(1.1 * fs), None)  # well after the source stopped
    tail_in = float(np.sum(reverberant[tail] ** 2))
    tail_out = float(np.sum(enhanced[tail] ** 2))
    assert tail_out < 0.9 * tail_in


def test_multichannel_wpe_shape():
    rng = np.random.default_rng(2)
    spectrum = rng.standard_normal((17, 2, 40)) + 1j * rng.standard_normal((17, 2, 40))
    out = wpe(spectrum, taps=3, delay=2, iterations=1)
    assert out.shape == spectrum.shape
    assert np.all(np.isfinite(out))


def test_invalid_parameters():
    spectrum = np.ones((4, 1, 8), dtype=complex)
    with pytest.raises(ValueError):
        wpe(spectrum, taps=0)
    with pytest.raises(ValueError):
        wpe(spectrum, iterations=0)
