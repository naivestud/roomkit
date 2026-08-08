import numpy as np
import pytest

from roomkit.denoise.spectral import spectral_subtraction


def _noisy_stft():
    # noise floor of 0.1 power everywhere plus a strong tone in bin 5
    noisy = np.full((8, 32), 0.1 + 0.0j)
    noisy[:, 5] = 1.0
    noise_power = np.full((8, 32), 0.1)
    return noisy, noise_power


def test_noise_bins_are_attenuated():
    noisy, noise_power = _noisy_stft()
    enhanced = spectral_subtraction(noisy, noise_power)
    assert np.all(np.abs(enhanced[:, 0]) < np.abs(noisy[:, 0]))
    assert np.allclose(enhanced[:, 0], 0.0, atol=1e-9)


def test_strong_bin_preserved():
    noisy, noise_power = _noisy_stft()
    enhanced = spectral_subtraction(noisy, noise_power)
    # tone power 1.0 minus noise 0.1 -> gain sqrt(0.9)
    assert np.allclose(np.abs(enhanced[:, 5]), np.sqrt(0.9), atol=1e-9)


def test_oversubtraction_removes_more():
    noisy, noise_power = _noisy_stft()
    mild = spectral_subtraction(noisy, noise_power, oversubtraction=1.0)
    strong = spectral_subtraction(noisy, noise_power, oversubtraction=1.5)
    assert np.abs(strong[:, 5]).mean() < np.abs(mild[:, 5]).mean()


def test_spectral_floor_bounds_gain():
    noisy, noise_power = _noisy_stft()
    enhanced = spectral_subtraction(noisy, noise_power, spectral_floor=0.04)
    gains = np.abs(enhanced) / np.abs(noisy)
    assert np.all(gains >= np.sqrt(0.04) - 1e-9)


def test_invalid_parameters():
    noisy, noise_power = _noisy_stft()
    with pytest.raises(ValueError):
        spectral_subtraction(noisy, noise_power, oversubtraction=0.0)
    with pytest.raises(ValueError):
        spectral_subtraction(noisy, noise_power, spectral_floor=1.5)


def test_aggressive_subtraction_floor_avoids_zeroing():
    # with strong oversubtraction the floor keeps every bin audible instead
    # of creating spectral holes (the main cause of musical noise)
    noisy, noise_power = _noisy_stft()
    enhanced = spectral_subtraction(
        noisy, noise_power, oversubtraction=2.0, spectral_floor=0.01
    )
    gains = np.abs(enhanced) / np.abs(noisy)
    assert np.all(gains > 0)
