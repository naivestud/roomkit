"""Energy-based voice activity detection."""

from __future__ import annotations

import numpy as np

from roomkit.audio.framing import frame_signal


def frame_energy_db(
    samples: np.ndarray, frame_length: int = 400, hop_length: int = 160
) -> np.ndarray:
    """Short-time energy in dB for each analysis frame.

    Args:
        samples: One-dimensional input signal.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.

    Returns:
        Energy in dB for each frame, floored at -120 dB.
    """
    samples = np.asarray(samples, dtype=np.float64)
    frames = frame_signal(samples, frame_length, hop_length)
    power = np.mean(frames**2, axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-12))
