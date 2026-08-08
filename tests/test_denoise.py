import numpy as np
import pytest

from roomkit.denoise import denoise


def _si_sdr(target: np.ndarray, estimate: np.ndarray) -> float:
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, target) / np.dot(target, target)
    projection = alpha * target
    residual = estimate - projection
    return float(10 * np.log10(np.dot(projection, projection) / (np.dot(residual, residual) + 1e-12)))


def _noisy_signal(seed: int = 3):
    fs = 16000
    n = int(1.5 * fs)
    t = np.arange(n) / fs
    speech = (
        0.3 * np.sin(2 * np.pi * 180 * t)
        + 0.15 * np.sin(2 * np.pi * 360 * t)
        + 0.08 * np.sin(2 * np.pi * 540 * t)
    )
    speech *= 1.0 + 0.5 * np.sin(2 * np.pi * 3 * t)
    clean = np.zeros(n)
    lead_in = int(0.2 * fs)  # noise-only lead-in for the noise estimate
    clean[lead_in:] = speech[: n - lead_in]
    sigma = float(np.sqrt(np.mean(speech**2)))  # roughly 0 dB SNR
    noise = np.random.default_rng(seed).standard_normal(n) * sigma
    return clean, clean + noise, lead_in


def test_wiener_improves_si_sdr():
    clean, noisy, lead_in = _noisy_signal()
    enhanced = denoise(noisy, method="wiener")
    interior = slice(lead_in + 400, -400)
    before = _si_sdr(clean[interior], noisy[interior])
    after = _si_sdr(clean[interior], enhanced[interior])
    assert after > before + 1.5


def test_spectral_subtraction_improves_si_sdr():
    clean, noisy, lead_in = _noisy_signal(seed=5)
    enhanced = denoise(noisy, method="spectral")
    interior = slice(lead_in + 400, -400)
    before = _si_sdr(clean[interior], noisy[interior])
    after = _si_sdr(clean[interior], enhanced[interior])
    assert after > before


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        denoise(np.zeros(800), method="magic")
