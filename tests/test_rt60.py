import numpy as np
import pytest

from roomkit.acoustics.rt60 import estimate_rt60, schroeder_integration


def _synthetic_ir(rt60: float, sample_rate: int = 16000, seed: int = 0) -> np.ndarray:
    """Exponentially decaying noise with the given RT60."""
    duration = 2.0 * rt60
    n = int(duration * sample_rate)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    t = np.arange(n) / sample_rate
    return noise * 10.0 ** (-3.0 * t / rt60)


def test_schroeder_curve_starts_at_zero_and_decays():
    response = _synthetic_ir(0.4)
    curve = schroeder_integration(response)
    assert curve[0] == pytest.approx(0.0)
    assert curve[-1] < -60.0
    # mostly non-increasing (cumulative energy only decreases)
    assert np.all(np.diff(curve) <= 1e-9)


def test_rt60_estimation_recovers_design_value():
    sample_rate = 16000
    for target in (0.3, 0.6):
        response = _synthetic_ir(target, sample_rate=sample_rate, seed=42)
        estimate = estimate_rt60(response, sample_rate)
        assert estimate == pytest.approx(target, rel=0.15)


def test_empty_response_raises():
    with pytest.raises(ValueError):
        schroeder_integration(np.array([]))


def test_regression_window_validation():
    response = _synthetic_ir(0.3)
    with pytest.raises(ValueError):
        estimate_rt60(response, 16000, lower_db=-35.0, upper_db=-5.0)
