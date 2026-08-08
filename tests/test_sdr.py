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


def test_sdr_not_scale_invariant():
    rng = np.random.default_rng(2)
    reference = rng.standard_normal(4000)
    base = sdr(reference, reference * 0.5)
    assert np.isfinite(base)
    assert base < 20  # a 2x gain error is a real distortion here


def test_more_noise_lower_si_sdr():
    rng = np.random.default_rng(3)
    reference = rng.standard_normal(4000)
    mild = reference + 0.05 * rng.standard_normal(4000)
    heavy = reference + 0.5 * rng.standard_normal(4000)
    assert si_sdr(reference, mild) > si_sdr(reference, heavy)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sdr(np.zeros(10), np.zeros(11))
