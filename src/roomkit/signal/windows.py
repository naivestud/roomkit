"""Analysis and synthesis windows."""

from __future__ import annotations

import numpy as np

_REGISTRY = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "ones": np.ones,
}


def get_window(name: str, length: int) -> np.ndarray:
    """Return a symmetric window of ``length`` samples.

    Args:
        name: One of ``hann``, ``hamming``, ``blackman``, ``ones``.
        length: Window length in samples.

    Returns:
        The window as a float64 array.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown window {name!r}; choose from: {known}") from None
    return factory(length).astype(np.float64)


def is_cola(window: np.ndarray, hop_length: int, tolerance: float = 1e-6) -> bool:
    """Check the constant overlap-add (COLA) constraint for a given hop.

    A window satisfies COLA when the sum of its overlapping copies is
    constant, which makes plain overlap-add inversion exact.

    Args:
        window: Window to test.
        hop_length: Hop between copies.
        tolerance: Allowed deviation from a constant sum.

    Returns:
        True when the constraint holds within ``tolerance``.
    """
    length = window.shape[0]
    total = length * 4
    acc = np.zeros(total + length)
    for start in range(0, total, hop_length):
        acc[start : start + length] += window
    interior = acc[length:-length]
    return bool(np.max(interior) - np.min(interior) < tolerance)
