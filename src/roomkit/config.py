"""Configuration objects shared across roomkit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Basic assumptions about the signal being processed.

    Attributes:
        sample_rate: Sampling rate in Hz.
        frame_length: Analysis window length in samples.
        hop_length: Hop between consecutive frames in samples.
    """

    sample_rate: int = 16000
    frame_length: int = 400
    hop_length: int = 160

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.frame_length <= 0:
            raise ValueError(f"frame_length must be positive, got {self.frame_length}")
        if not 0 < self.hop_length <= self.frame_length:
            raise ValueError(f"hop_length must be in (0, frame_length], got {self.hop_length}")

    @property
    def frame_duration(self) -> float:
        """Analysis window duration in seconds."""
        return self.frame_length / self.sample_rate

    @property
    def hop_duration(self) -> float:
        """Hop duration in seconds."""
        return self.hop_length / self.sample_rate
