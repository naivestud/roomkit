"""Torch log-mel frontend mirroring :mod:`roomkit.signal.mel`."""

from __future__ import annotations

import torch

from roomkit.neural.config import EncoderConfig
from roomkit.signal.mel import mel_filterbank


def log_mel_spectrogram(waveform: torch.Tensor, config: EncoderConfig) -> torch.Tensor:
    """Compute whisper-style log-mel features for a batch of waveforms.

    Args:
        waveform: Tensor of shape ``(num_samples,)`` or ``(batch, num_samples)``.
        config: Encoder configuration carrying the frontend parameters.

    Returns:
        Features of shape ``(batch, num_frames, n_mels)``.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    window = torch.hann_window(config.n_fft, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        config.n_fft,
        hop_length=config.hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = spectrum.abs()  # (batch, bins, frames)
    filterbank = mel_filterbank(config.n_mels, config.n_fft, config.sample_rate)
    fb = torch.from_numpy(filterbank).to(device=waveform.device, dtype=magnitudes.dtype)
    mel = fb @ magnitudes  # (batch, n_mels, frames)
    log_spec = torch.log10(torch.clamp(mel, min=1e-10))
    peak = log_spec.amax(dim=(-2, -1), keepdim=True)
    log_spec = torch.maximum(log_spec, peak - 8.0)
    scaled = (log_spec + 4.0) / 4.0
    return scaled.transpose(1, 2)
