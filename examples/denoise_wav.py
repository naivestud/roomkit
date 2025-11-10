"""Denoise a synthetic noisy recording end to end.

Generates a harmonic "voice" buried in white noise, denoises it with both
available methods and reports SI-SDR before/after. No input files needed:

    python examples/denoise_wav.py
"""

from __future__ import annotations

import numpy as np

from roomkit.denoise import denoise
from roomkit.metrics.sdr import si_sdr


def make_noisy_signal(sample_rate: int = 16000, seconds: float = 1.5, seed: int = 0):
    """Noise-only lead-in followed by a noisy harmonic stack."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    voice = np.zeros(n)
    for k in range(1, 7):
        voice += np.sin(2 * np.pi * 180.0 * k * t) / k
    voice *= 0.3 * (1.0 + 0.4 * np.sin(2 * np.pi * 3.0 * t))
    clean = np.zeros(n)
    lead_in = int(0.2 * sample_rate)
    clean[lead_in:] = voice[: n - lead_in]
    sigma = float(np.sqrt(np.mean(voice**2)))  # ~0 dB SNR
    noise = rng.standard_normal(n) * sigma
    return clean, clean + noise, lead_in


def main() -> None:
    clean, noisy, lead_in = make_noisy_signal()
    interior = slice(lead_in + 400, -400)
    print(f"noisy SI-SDR:    {si_sdr(clean[interior], noisy[interior]):6.2f} dB")
    for method in ("wiener", "spectral"):
        enhanced = denoise(noisy, method=method)
        gained = si_sdr(clean[interior], enhanced[interior])
        print(f"{method:8s} SI-SDR: {gained:6.2f} dB")


if __name__ == "__main__":
    main()
