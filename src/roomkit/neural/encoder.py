"""Whisper-style audio encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from roomkit.neural.config import EncoderConfig
from roomkit.neural.mel import log_mel_spectrogram
from roomkit.neural.positional import sinusoidal_positions
from roomkit.neural.transformer import TransformerBlock


class WhisperStyleEncoder(nn.Module):
    """Audio encoder following the whisper architecture.

    Log-mel features run through two 1-D convolutions (the second strided by
    two, halving the frame rate), fixed sinusoidal positions are added, and a
    stack of pre-LN transformer blocks produces one embedding per output
    frame. A final layer norm stabilises the output scale.

    Weights are randomly initialised; load a trained checkpoint to use the
    encoder as a feature extractor.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        c = self.config
        self.conv1 = nn.Conv1d(c.n_mels, c.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c.d_model, c.d_model, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional", sinusoidal_positions(c.max_frames, c.d_model))
        self.blocks = nn.ModuleList(
            TransformerBlock(c.d_model, c.n_heads) for _ in range(c.n_layers)
        )
        self.ln_post = nn.LayerNorm(c.d_model)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode waveforms into frame embeddings.

        Args:
            waveform: Tensor of shape ``(num_samples,)`` or
                ``(batch, num_samples)``.

        Returns:
            Embeddings of shape ``(batch, frames // 2, d_model)``.
        """
        mel = log_mel_spectrogram(waveform, self.config)  # (B, T, n_mels)
        x = functional.gelu(self.conv1(mel.transpose(1, 2)))
        x = functional.gelu(self.conv2(x)).transpose(1, 2)
        frames = x.shape[1]
        if frames > self.config.max_frames:
            raise ValueError(
                f"audio produces {frames} frames, exceeding max_frames="
                f"{self.config.max_frames}"
            )
        x = x + self.positional[:frames]
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)
