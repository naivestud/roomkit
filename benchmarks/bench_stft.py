"""Benchmark STFT / ISTFT throughput on a few representative sizes.

Run directly: python benchmarks/bench_stft.py
"""

from __future__ import annotations

import time

import numpy as np

from roomkit.signal.stft import istft, stft

SIZES = [
    (16000, 400, 160),   # 1 s at the toolkit defaults
    (160000, 400, 160),  # 10 s at the toolkit defaults
    (160000, 1024, 256),  # 10 s, larger frames
]


def bench(samples: np.ndarray, n_fft: int, hop_length: int, repeats: int = 5) -> float:
    """Median wall time in milliseconds for one STFT + ISTFT round trip."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        spec = stft(samples, n_fft=n_fft, hop_length=hop_length)
        istft(spec, hop_length=hop_length, length=samples.shape[0])
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)) * 1000.0


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'samples':>10} {'n_fft':>6} {'hop':>5} {'round trip (ms)':>16}")
    for length, n_fft, hop in SIZES:
        samples = rng.standard_normal(length)
        elapsed = bench(samples, n_fft, hop)
        print(f"{length:>10} {n_fft:>6} {hop:>5} {elapsed:>16.2f}")


if __name__ == "__main__":
    main()
