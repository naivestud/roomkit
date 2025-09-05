"""Shared fixtures: synthetic speech-like signals for deterministic tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def sample_rate() -> int:
    return 16000


@pytest.fixture()
def harmonic_voice(sample_rate):
    """One second of a vowel-like harmonic stack with a noise-only lead-in."""

    def _make(seed: int = 0, f0: float = 180.0, lead_in: float = 0.2):
        rng = np.random.default_rng(seed)
        n = int(1.2 * sample_rate)
        t = np.arange(n) / sample_rate
        voice = np.zeros(n)
        for k in range(1, 7):
            voice += np.sin(2 * np.pi * f0 * k * t) / k
        voice *= 0.3 * (1.0 + 0.4 * np.sin(2 * np.pi * 3 * t))
        clean = np.zeros(n)
        clean[int(lead_in * sample_rate) :] = voice[: n - int(lead_in * sample_rate)]
        noise = rng.standard_normal(n) * 0.03
        return clean, clean + noise

    return _make
