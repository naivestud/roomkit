"""Analysis features for the codec frontend."""

from __future__ import annotations

import numpy as np

from roomkit.signal.mel import log_mel_spectrogram


def codec_features(
    samples: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
) -> np.ndarray:
    """Log-mel frames used as the codec analysis representation.

    A compact spectral representation keeps the codebooks small while still
    carrying enough detail for intelligible resynthesis.

    Args:
        samples: One-dimensional input signal.
        sample_rate: Sampling rate in Hz.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.
        n_mels: Number of mel bands.

    Returns:
        Feature matrix of shape ``(num_frames, n_mels)``.
    """
    return log_mel_spectrogram(
        samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=frame_length,
        hop_length=hop_length,
    )
