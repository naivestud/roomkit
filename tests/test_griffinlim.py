import numpy as np
import pytest

from roomkit.signal.griffinlim import griffin_lim
from roomkit.signal.stft import stft


def _magnitude_error(samples, magnitude, hop_length):
    spec = stft(samples, n_fft=(magnitude.shape[-1] - 1) * 2, hop_length=hop_length)
    spec = spec[: magnitude.shape[0]]
    return float(np.linalg.norm(np.abs(spec) - magnitude))


def test_griffin_lim_reduces_error():
    rng = np.random.default_rng(7)
    samples = rng.standard_normal(4000)
    hop = 160
    magnitude = np.abs(stft(samples, n_fft=400, hop_length=hop))

    coarse = griffin_lim(magnitude, hop_length=hop, n_iter=2)
    fine = griffin_lim(magnitude, hop_length=hop, n_iter=32)

    target = float(np.linalg.norm(magnitude))
    assert _magnitude_error(fine, magnitude, hop) < _magnitude_error(
        coarse, magnitude, hop
    )
    # after 32 iterations most of the spectral energy should match
    assert _magnitude_error(fine, magnitude, hop) < 0.5 * target


def test_griffin_lim_deterministic_with_seed():
    magnitude = np.abs(np.random.default_rng(1).standard_normal((20, 201)))
    a = griffin_lim(magnitude, n_iter=4, seed=3)
    b = griffin_lim(magnitude, n_iter=4, seed=3)
    assert np.allclose(a, b)


def test_negative_magnitude_raises():
    with pytest.raises(ValueError):
        griffin_lim(np.ones((4, 201)) * -1.0)
