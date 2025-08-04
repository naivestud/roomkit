"""Spectral distortion metrics."""

from __future__ import annotations

import numpy as np

from roomkit.audio.framing import frame_signal
from roomkit.signal.stft import stft


def log_spectral_distortion(
    reference: np.ndarray,
    estimate: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
) -> float:
    """Log-spectral distortion in dB.

    The root-mean-square difference of the log power spectra, averaged over
    frames and bins. Lower is better; 0 means identical spectra.

    Args:
        reference: Clean target signal.
        estimate: Enhanced estimate of the same length.
        n_fft: FFT size used for the analysis.
        hop_length: Hop between frames in samples.

    Returns:
        LSD in dB.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same length")
    ref_power = np.abs(stft(reference, n_fft=n_fft, hop_length=hop_length)) ** 2
    est_power = np.abs(stft(estimate, n_fft=n_fft, hop_length=hop_length)) ** 2
    frames = min(ref_power.shape[0], est_power.shape[0])
    log_diff = 10.0 * np.log10(np.maximum(ref_power[:frames], 1e-12)) - 10.0 * np.log10(
        np.maximum(est_power[:frames], 1e-12)
    )
    return float(np.sqrt(np.mean(log_diff**2)))


def segmental_snr(
    reference: np.ndarray,
    estimate: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
    min_snr_db: float = -10.0,
    max_snr_db: float = 35.0,
) -> float:
    """Segmental SNR in dB: the average of per-frame SNRs.

    Frames where the reference is silent are skipped, and per-frame values
    are clamped to ``[min_snr_db, max_snr_db]`` following common practice.

    Args:
        reference: Clean target signal.
        estimate: Enhanced estimate of the same length.
        frame_length: Analysis frame length in samples.
        hop_length: Hop between frames in samples.
        min_snr_db: Lower clamp for per-frame SNR.
        max_snr_db: Upper clamp for per-frame SNR.

    Returns:
        Segmental SNR in dB.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same length")
    ref_frames = frame_signal(reference, frame_length, hop_length)
    noise_frames = frame_signal(reference - estimate, frame_length, hop_length)
    ref_energy = np.sum(ref_frames**2, axis=1)
    noise_energy = np.sum(noise_frames**2, axis=1)
    active = ref_energy > 1e-10
    if not active.any():
        raise ValueError("reference is silent in every frame")
    snr = 10.0 * np.log10(ref_energy[active] / np.maximum(noise_energy[active], 1e-12))
    return float(np.mean(np.clip(snr, min_snr_db, max_snr_db)))
