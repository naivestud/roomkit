"""Enhancement pipelines combining multiple stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomkit.denoise import denoise
from roomkit.dereverb import dereverb


@dataclass(frozen=True)
class EnhancementPipeline:
    """Chain of enhancement stages applied to a waveform.

    Stages run in a fixed order: optional denoising followed by optional
    dereverberation. Parameters default to those of the underlying
    functions.

    Attributes:
        do_denoise: Run the denoising stage.
        do_dereverb: Run the WPE stage.
        denoise_method: ``wiener`` or ``spectral``.
        noise_frames: Lead-in frames used for the noise estimate.
        taps: WPE filter taps.
        delay: WPE prediction delay in frames.
        iterations: WPE iterations.
        n_fft: FFT size shared by both stages.
        hop_length: Hop shared by both stages.
    """

    do_denoise: bool = True
    do_dereverb: bool = False
    denoise_method: str = "wiener"
    noise_frames: int = 10
    taps: int = 8
    delay: int = 3
    iterations: int = 3
    n_fft: int = 400
    hop_length: int = 160

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Run the pipeline and return the enhanced waveform."""
        out = np.asarray(samples, dtype=np.float64)
        if self.do_denoise:
            out = denoise(
                out,
                method=self.denoise_method,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                noise_frames=self.noise_frames,
            )
        if self.do_dereverb:
            out = dereverb(
                out,
                taps=self.taps,
                delay=self.delay,
                iterations=self.iterations,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        return out
