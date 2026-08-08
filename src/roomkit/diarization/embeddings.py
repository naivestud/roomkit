"""Frame-level spectral embeddings for speaker clustering.

The embeddings are hand-crafted spectral shape statistics (band energies,
spectral centroid) aggregated over short windows. They separate voices by
pitch and timbre without any trained model, which keeps the pipeline free of
checkpoint downloads.
"""

from __future__ import annotations

import numpy as np

from roomkit.signal.stft import stft

_NUM_BANDS = 8


def _band_energies(power: np.ndarray, num_bands: int = _NUM_BANDS) -> np.ndarray:
    """Average power per log-spaced frequency band, in dB."""
    num_bins = power.shape[-1]
    edges = np.logspace(np.log10(1), np.log10(num_bins), num_bands + 1).astype(int)
    # force strictly increasing edges so every band survives the int cast
    for index in range(1, edges.shape[0]):
        edges[index] = max(edges[index], edges[index - 1] + 1)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        bands.append(power[..., lo:hi].mean(axis=-1))
    stacked = np.stack(bands, axis=-1)
    return 10.0 * np.log10(np.maximum(stacked, 1e-12))


def _spectral_centroid(power: np.ndarray) -> np.ndarray:
    """Power-weighted mean bin index of each frame."""
    bins = np.arange(power.shape[-1])
    total = power.sum(axis=-1)
    return np.where(total > 0, (power * bins).sum(axis=-1) / np.maximum(total, 1e-12), 0.0)


def spectral_embeddings(
    samples: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    window_frames: int = 25,
    window_hop: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed sliding windows of the signal for speaker discrimination.

    Each window embedding concatenates the mean and standard deviation of
    the log band energies plus the mean spectral centroid, then L2
    normalisation makes cosine distances meaningful.

    Args:
        samples: One-dimensional input signal.
        frame_length: STFT frame length in samples.
        hop_length: STFT hop in samples.
        window_frames: Analysis windows in STFT frames (~250 ms at defaults).
        window_hop: Hop between windows in STFT frames.

    Returns:
        Tuple of ``(embeddings, window_starts)`` where embeddings has shape
        ``(num_windows, 2 * num_bands + 1)`` and window_starts gives the
        first STFT frame of each window.
    """
    if window_frames <= 0 or window_hop <= 0:
        raise ValueError("window_frames and window_hop must be positive")
    samples = np.asarray(samples, dtype=np.float64)
    spec = stft(samples, n_fft=frame_length, hop_length=hop_length)
    power = np.abs(spec) ** 2
    frames = np.concatenate(
        [_band_energies(power), _spectral_centroid(power)[..., None]], axis=-1
    )
    num_frames = frames.shape[0]
    if num_frames < window_frames:
        return np.empty((0, 2 * (_NUM_BANDS + 1))), np.empty(0, dtype=np.int64)
    starts = np.arange(0, num_frames - window_frames + 1, window_hop)
    embeddings = np.stack(
        [
            np.concatenate(
                [frames[start : start + window_frames].mean(axis=0),
                 frames[start : start + window_frames].std(axis=0)]
            )
            for start in starts
        ]
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12), starts
