"""Encode audio into discrete codec tokens and resynthesize it.

Fits the RVQ codebooks on a synthetic signal, encodes it, prints the token
budget, and resynthesizes a waveform from the tokens alone.

    python examples/codec_tokens.py
"""

from __future__ import annotations

import numpy as np

from roomkit.codec.frontend import CodecFrontend


def make_signal(sample_rate: int = 16000, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """A vowel-like harmonic stack with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    for k in range(1, 8):
        signal += np.sin(2 * np.pi * 220.0 * k * t) / k
    signal *= 0.3 * (1.0 + 0.4 * np.sin(2 * np.pi * 3.0 * t))
    return signal + rng.standard_normal(signal.shape) * 0.01


def main() -> None:
    samples = make_signal()
    for n_codebooks in (2, 4, 8):
        frontend = CodecFrontend(n_codebooks=n_codebooks, codebook_size=32).fit(samples)
        tokens = frontend.encode(samples)
        bits = tokens.size * int(np.log2(frontend.codebook_size))
        kbps = bits / (len(samples) / 16000) / 1000
        waveform = frontend.synthesize(tokens, n_iter=16)
        print(
            f"codebooks={n_codebooks}: tokens {tokens.shape}, "
            f"~{kbps:6.1f} kbps, resynthesized {len(waveform)} samples"
        )


if __name__ == "__main__":
    main()
