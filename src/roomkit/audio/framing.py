"""Framing helpers: split signals into overlapping frames and reassemble."""

from __future__ import annotations

import numpy as np


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split ``samples`` into overlapping frames.

    The signal is right-padded with zeros so the last frame is complete.

    Args:
        samples: One-dimensional input signal.
        frame_length: Number of samples per frame.
        hop_length: Hop between frame starts.

    Returns:
        Array of shape ``(num_frames, frame_length)``.
    """
    if samples.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {samples.shape}")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if samples.shape[0] < frame_length:
        padded = np.zeros(frame_length, dtype=samples.dtype)
        padded[: samples.shape[0]] = samples
        return padded[None, :]
    num_frames = 1 + (samples.shape[0] - frame_length) // hop_length
    shape = (num_frames, frame_length)
    strides = (samples.strides[0] * hop_length, samples.strides[0])
    frames = np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides)
    return frames.copy()


def overlap_add(
    frames: np.ndarray, hop_length: int, window: np.ndarray | None = None
) -> np.ndarray:
    """Reassemble ``frames`` into a signal with overlap-add.

    When a ``window`` is given each frame is weighted before summation and the
    result is normalised by the accumulated window energy, which inverts the
    analysis pattern ``frame * window`` for any window.

    Args:
        frames: Array of shape ``(num_frames, frame_length)``.
        hop_length: Hop between frame starts.
        window: Optional analysis window of length ``frame_length``.

    Returns:
        The reassembled signal.
    """
    num_frames, frame_length = frames.shape
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    out_length = (num_frames - 1) * hop_length + frame_length
    out = np.zeros(out_length, dtype=np.result_type(frames, np.float64))
    if window is None:
        for index, frame in enumerate(frames):
            start = index * hop_length
            out[start : start + frame_length] += frame
        return out
    weight = np.asarray(window, dtype=np.float64)
    norm = np.zeros(out_length)
    for index, frame in enumerate(frames):
        start = index * hop_length
        out[start : start + frame_length] += frame * weight
        norm[start : start + frame_length] += weight**2
    return out / np.maximum(norm, 1e-8)
