import numpy as np
import pytest

from roomkit.metrics.sdr import sdr, si_sdr


def test_identical_signals_high_sdr():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal(4000)
    assert sdr(reference, reference) == float("inf")
    assert si_sdr(reference, reference) == float("inf")


def test_scale_invariance():
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(4000)
    noisy = reference + 0.1 * rng.standard_normal(4000)
    base = si_sdr(reference, noisy)
    assert si_sdr(reference, 3.0 * noisy) == pytest.approx(base, abs=1e-9)


def test_sdr_ignores_gain_but_not_offset():
    t = np.arange(4000) / 16000
    tone = np.sin(2 * np.pi * 440 * t)
    reference = tone + 1.0  # DC offset
    estimate = tone  # same tone without the offset
    assert np.isfinite(sdr(reference, estimate))
    assert sdr(reference, estimate) < 10  # the offset is real distortion here
    # SI-SDR removes the mean of both signals, so the offset nearly disappears
    assert si_sdr(reference, estimate) > 100


def test_more_noise_lower_si_sdr():
    rng = np.random.default_rng(3)
    reference = rng.standard_normal(4000)
    mild = reference + 0.05 * rng.standard_normal(4000)
    heavy = reference + 0.5 * rng.standard_normal(4000)
    assert si_sdr(reference, mild) > si_sdr(reference, heavy)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sdr(np.zeros(10), np.zeros(11))
