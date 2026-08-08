import numpy as np
import pytest

from roomkit.denoise.noise_estimate import estimate_noise


def test_tracks_stationary_noise():
    rng = np.random.default_rng(0)
    sigma = 0.1
    magnitudes = np.abs(rng.normal(0.0, sigma, size=(200, 64)))
    noise = estimate_noise(magnitudes, initial_frames=10, window=30)
    assert noise.shape == magnitudes.shape
    # windowed minimum of exponential noise sits below the true power
    assert np.mean(noise[50:]) < sigma**2
    assert np.all(noise > 0)


def test_does_not_follow_speech_up():
    rng = np.random.default_rng(1)
    sigma = 0.05
    magnitudes = np.abs(rng.normal(0.0, sigma, size=(240, 32)))
    # loud "speech" bursts shorter than the minimum window
    magnitudes[50:85] += 1.0
    magnitudes[150:185] += 1.0
    noise = estimate_noise(magnitudes, initial_frames=10, window=60)
    # during speech the estimate should stay near the noise power, not the speech
    assert np.mean(noise[60:85]) < 5 * sigma**2


def test_invalid_input():
    with pytest.raises(ValueError):
        estimate_noise(np.zeros(10))
    with pytest.raises(ValueError):
        estimate_noise(np.zeros((10, 5)), window=0)
