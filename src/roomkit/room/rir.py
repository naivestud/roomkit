"""Shoebox room impulse response simulation via the image source method."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class ShoeboxRoom:
    """Axis-aligned shoebox room with uniform wall absorption.

    Attributes:
        dimensions: (length, width, height) in meters.
        absorption: Average wall absorption coefficient in (0, 1).
    """

    dimensions: tuple[float, float, float]
    absorption: float = 0.3

    def __post_init__(self) -> None:
        if any(d <= 0 for d in self.dimensions):
            raise ValueError("room dimensions must be positive")
        if not 0 < self.absorption < 1:
            raise ValueError("absorption must be in (0, 1)")

    @property
    def volume(self) -> float:
        length, width, height = self.dimensions
        return length * width * height

    @property
    def surface_area(self) -> float:
        length, width, height = self.dimensions
        return 2 * (length * width + length * height + width * height)


def _reflection_order(n: tuple[int, int, int], mirrored: tuple[int, int, int]) -> int:
    """Number of wall reflections for an image source indexed by (n, mirrored).

    An image at ``(-1)**p * x0 + 2 n L`` sits ``|2n - p|`` cells away from the
    original room along that axis, crossing one wall per cell boundary.
    """
    return sum(abs(2 * n_i - p_i) for n_i, p_i in zip(n, mirrored, strict=True))


def simulate_rir(
    room: ShoeboxRoom,
    source: Sequence[float],
    microphone: Sequence[float],
    sample_rate: int = 16000,
    max_order: int = 2,
) -> np.ndarray:
    """Simulate a room impulse response with the image source method.

    Each image source contributes an impulse delayed by its propagation time
    and attenuated by spherical spreading and wall reflections. The result is
    normalised to unit peak.

    Args:
        room: Shoebox room description.
        source: Source position (x, y, z) in meters, inside the room.
        microphone: Microphone position (x, y, z) in meters, inside the room.
        sample_rate: Sampling rate of the returned impulse response.
        max_order: Maximum number of reflections to simulate.

    Returns:
        Impulse response as a float64 array.
    """
    source_arr = np.asarray(source, dtype=np.float64)
    mic = np.asarray(microphone, dtype=np.float64)
    for point, name in ((source_arr, "source"), (mic, "microphone")):
        if point.shape != (3,) or not np.all((point > 0) & (point < room.dimensions)):
            raise ValueError(f"{name} must lie strictly inside the room")

    reflection = math.sqrt(1.0 - room.absorption)
    dims = np.asarray(room.dimensions, dtype=np.float64)

    delays = []
    amplitudes = []
    orders = range(-max_order, max_order + 1)
    for n, mirrored in itertools.product(orders, itertools.product((0, 1), repeat=3)):
        order = _reflection_order(n, mirrored)
        if order > max_order:
            continue
        signs = np.array([1.0 if p == 0 else -1.0 for p in mirrored])
        image = signs * source_arr + 2.0 * np.asarray(n) * dims
        distance = float(np.linalg.norm(image - mic))
        delay = int(round(distance / SPEED_OF_SOUND * sample_rate))
        amplitude = reflection**order / (4.0 * math.pi * distance)
        delays.append(delay)
        amplitudes.append(amplitude)

    response = np.zeros(max(delays) + 1)
    np.add.at(response, np.asarray(delays), np.asarray(amplitudes))
    return response / np.max(np.abs(response))
