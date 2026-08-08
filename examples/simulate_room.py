"""Simulate a shoebox room, reverberate a signal and dereverberate it.

Builds an image-source room impulse response, convolves a noise burst with
it, runs WPE, and reports how much the late tail was suppressed.

    python examples/simulate_room.py
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from roomkit.acoustics.rir import ShoeboxRoom, simulate_rir
from roomkit.acoustics.rt60 import estimate_rt60
from roomkit.acoustics.sabine import sabine_rt60
from roomkit.dereverb import dereverb


def main() -> None:
    fs = 16000
    room = ShoeboxRoom(dimensions=(5.0, 4.0, 3.0), absorption=0.2)
    rir = simulate_rir(
        room,
        source=(1.0, 1.0, 1.5),
        microphone=(3.5, 2.5, 1.0),
        sample_rate=fs,
        max_order=3,
    )
    design = sabine_rt60(room.volume, room.surface_area, room.absorption)
    measured = estimate_rt60(rir, fs)
    print(f"design RT60 (Sabine):  {design:5.2f} s")
    print(f"measured RT60 (from RIR): {measured:5.2f} s")

    rng = np.random.default_rng(0)
    source = rng.standard_normal(int(1.0 * fs)) * 0.3
    reverberant = fftconvolve(source, rir)[: len(source) + int(0.5 * fs)]

    enhanced = dereverb(reverberant, taps=10, delay=3, iterations=5)

    tail = slice(int(1.1 * fs), None)  # after the source stopped
    tail_in = float(np.sum(reverberant[tail] ** 2))
    tail_out = float(np.sum(enhanced[tail] ** 2))
    reduction = 10 * np.log10(tail_in / max(tail_out, 1e-12))
    print(f"late-tail energy reduced by {reduction:5.1f} dB")


if __name__ == "__main__":
    main()
