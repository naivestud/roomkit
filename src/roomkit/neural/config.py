"""Configuration for the whisper-style encoder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    """Hyper-parameters for :class:`~roomkit.neural.encoder.WhisperStyleEncoder`.

    The frontend defaults follow the whisper recipe (16 kHz, 25 ms frames,
    10 ms hop, 80 mel bands); the transformer body defaults are small so a
    randomly initialised model stays cheap enough for tests and examples.
    """

    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_frames: int = 1500

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("sample_rate, n_fft and hop_length must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.d_model <= 0 or self.d_model % 2 != 0:
            raise ValueError("d_model must be a positive even number")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 0:
            raise ValueError("n_layers must be non-negative")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
