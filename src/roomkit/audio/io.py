"""Reading and writing audio files with soundfile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Load an audio file as float samples in [-1, 1].

    Multi-channel files are returned with shape ``(num_samples, num_channels)``.

    Args:
        path: Path to any format supported by soundfile (wav, flac, ogg, ...).
        dtype: Numpy dtype string for the returned samples.

    Returns:
        Tuple of ``(samples, sample_rate)``.
    """
    data, sample_rate = sf.read(str(path), dtype=dtype)
    return np.asarray(data), int(sample_rate)


def save_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write samples to ``path``, creating parent directories as needed.

    Args:
        path: Destination file path; the format is inferred from the suffix.
        samples: Mono ``(num_samples,)`` or multi-channel ``(num_samples, channels)``.
        sample_rate: Sampling rate in Hz.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples)
    if samples.ndim > 2:
        raise ValueError(f"expected at most 2 dimensions, got {samples.ndim}")
    sf.write(str(path), samples, sample_rate)
    return path
