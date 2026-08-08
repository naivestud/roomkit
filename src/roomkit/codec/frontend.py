"""Codec frontend: waveform to discrete tokens and back."""

from __future__ import annotations

import numpy as np

from roomkit.codec.features import codec_features
from roomkit.codec.rvq import fit_rvq, rvq_decode, rvq_encode


class CodecFrontend:
    """Discrete token frontend in the style of a neural audio codec.

    Audio is mapped to log-mel frames which are quantized by a stack of
    residual vector quantizers; the resulting tokens are a compact,
    bandwidth-controllable representation of the signal.

    Call :meth:`fit` on representative audio before :meth:`encode`.

    Args:
        n_codebooks: Number of residual quantization stages; more stages
            mean higher fidelity and higher bitrate.
        codebook_size: Entries per codebook.
        sample_rate: Sampling rate the frontend operates on.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.
        n_mels: Number of mel bands in the analysis features.
    """

    def __init__(
        self,
        n_codebooks: int = 4,
        codebook_size: int = 32,
        sample_rate: int = 16000,
        frame_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
    ) -> None:
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._codebooks: list[np.ndarray] | None = None

    @property
    def fitted(self) -> bool:
        """Whether the codebooks have been trained."""
        return self._codebooks is not None

    def _features(self, samples: np.ndarray) -> np.ndarray:
        return codec_features(
            samples,
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

    def fit(self, samples: np.ndarray, iterations: int = 25, seed: int = 0) -> "CodecFrontend":
        """Train the codebooks on the analysis features of ``samples``."""
        features = self._features(np.asarray(samples, dtype=np.float64))
        self._codebooks = fit_rvq(
            features,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            iterations=iterations,
            seed=seed,
        )
        return self

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """Encode a waveform into tokens of shape ``(num_frames, n_codebooks)``."""
        if not self.fitted:
            raise RuntimeError("call fit() before encode()")
        features = self._features(np.asarray(samples, dtype=np.float64))
        assert self._codebooks is not None  # for type checkers
        return rvq_encode(features, self._codebooks)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode tokens back into analysis features.

        Args:
            tokens: Integer tokens of shape ``(num_frames, n_codebooks)``.

        Returns:
            Reconstructed feature frames of shape ``(num_frames, n_mels)``.
        """
        if not self.fitted:
            raise RuntimeError("call fit() before decode()")
        assert self._codebooks is not None  # for type checkers
        return rvq_decode(tokens, self._codebooks)
