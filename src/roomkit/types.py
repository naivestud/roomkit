"""Shared data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A labelled time interval.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        label: Label of the interval (e.g. a speaker id).
    """

    start: float
    end: float
    label: str = "speech"

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end must not be before start")

    @property
    def duration(self) -> float:
        """Interval length in seconds."""
        return self.end - self.start
