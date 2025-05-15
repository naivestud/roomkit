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


def energy_vad(
    samples: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold_db: float = 6.0,
    noise_frames: int = 10,
) -> np.ndarray:
    """Frame-level voice activity decisions from adaptive energy thresholding.

    The noise floor is the mean energy of the first ``noise_frames`` frames
    (assumed noise-only); a frame is marked speech when its energy exceeds
    the floor by ``threshold_db``.

    Args:
        samples: One-dimensional input signal.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.
        threshold_db: Margin above the noise floor required to call speech.
        noise_frames: Number of leading frames used to estimate the floor.

    Returns:
        Boolean array, one decision per frame.
    """
    if threshold_db <= 0:
        raise ValueError("threshold_db must be positive")
    energy = frame_energy_db(samples, frame_length, hop_length)
    if energy.shape[0] == 0:
        return energy.astype(bool)
    noise_floor = float(np.mean(energy[: max(noise_frames, 1)]))
    return energy > noise_floor + threshold_db
