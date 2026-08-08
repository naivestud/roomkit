import numpy as np
import pytest

from roomkit.denoise.wiener import wiener_gain


def test_gain_bounded_zero_one():
    rng = np.random.default_rng(0)
    magnitudes = rng.random((16, 32))
    noise_power = rng.random((16, 32))
    gain = wiener_gain(magnitudes, noise_power)
    assert np.all(gain >= 0.0)
    assert np.all(gain <= 1.0)


def test_high_snr_gain_near_one():
    gain = wiener_gain(np.array([[10.0]]), np.array([[0.01]]))
    assert float(gain[0, 0]) == pytest.approx(0.9999, rel=1e-3)


def test_low_snr_gain_near_zero():
    gain = wiener_gain(np.array([[0.1]]), np.array([[1.0]]))
    assert float(gain[0, 0]) == 0.0


def test_gain_floor_applied():
    gain = wiener_gain(np.array([[0.1]]), np.array([[1.0]]), gain_floor=0.1)
    assert float(gain[0, 0]) == pytest.approx(0.1)


def test_invalid_floor():
    with pytest.raises(ValueError):
        wiener_gain(np.ones((2, 2)), np.ones((2, 2)), gain_floor=1.2)
