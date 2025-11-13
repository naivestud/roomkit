"""Simulate rooms, measure RT60 and dereverberate.

Part 1 builds an image-source room impulse response. Part 2 convolves a
noise burst with a dense exponentially decaying IR (a stand-in for a
diffuse late field), checks that RT60 estimation recovers the design value,
and shows WPE suppressing the tail.

    python examples/simulate_room.py
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from roomkit.acoustics.rir import ShoeboxRoom, simulate_rir
from roomkit.acoustics.rt60 import estimate_rt60
from roomkit.acoustics.sabine import sabine_rt60
from roomkit.dereverb import dereverb


def synthetic_ir(rt60: float, sample_rate: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Exponentially decaying noise with the given RT60."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    noise = np.random.default_rng(seed).standard_normal(n)
    ir = noise * 10.0 ** (-3.0 * t / rt60)
    ir[0] = 1.0
    return ir


def main() -> None:
    fs = 16000

    # --- image-source simulation -----------------------------------------
    room = ShoeboxRoom(dimensions=(5.0, 4.0, 3.0), absorption=0.2)
    rir = simulate_rir(
        room,
        source=(1.0, 1.0, 1.5),
        microphone=(3.5, 2.5, 1.0),
        sample_rate=fs,
        max_order=3,
    )
    design = sabine_rt60(room.volume, room.surface_area, room.absorption)
    print(f"shoebox {room.dimensions} m, absorption {room.absorption}")
    print(f"  Sabine RT60:        {design:5.2f} s")
    print(f"  image-source RIR:   {len(rir)} taps ({len(rir) / fs * 1000:.0f} ms)")

    # --- RT60 estimation on a dense tail ---------------------------------
    target = 0.5
    ir = synthetic_ir(target, fs, seconds=1.5)
    print(f"\nsynthetic IR designed for RT60 = {target:.2f} s")
    print(f"  estimated RT60:     {estimate_rt60(ir, fs):5.2f} s")

    # --- dereverberation --------------------------------------------------
    # WPE shortens the late tail; use a noise-like source and a decay the
    # filter taps can actually cover (matches tests/test_wpe.py)
    decay = 0.4
    short_ir = synthetic_ir(decay, fs, seconds=0.6, seed=2)
    rng = np.random.default_rng(1)
    source = rng.standard_normal(int(1.0 * fs)) * 0.3
    reverberant = fftconvolve(source, short_ir)[: len(source) + int(0.5 * fs)]
    enhanced = dereverb(reverberant, taps=10, delay=3, iterations=5)

    tail = slice(int(1.1 * fs), None)  # after the source stopped
    tail_in = float(np.sum(reverberant[tail] ** 2))
    tail_out = float(np.sum(enhanced[tail] ** 2))
    reduction = 10 * np.log10(tail_in / max(tail_out, 1e-12))
    print(f"\nlate-tail energy reduced by {reduction:5.1f} dB (RT60={decay:.1f} s)")


if __name__ == "__main__":
    main()
