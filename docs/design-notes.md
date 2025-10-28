# Design notes

A few decisions that are not obvious from the code alone.

## Why classical DSP next to a whisper-style encoder

The brief pairs a whisper-style encoder with a neural-codec frontend, but
neither should be a black box you must download to do useful work. The
classical path (STFT filtering, WPE, image-source simulation) runs on
numpy alone, is fully testable against closed-form expectations, and gives
the neural pieces something honest to be compared against. The encoder is
therefore optional (`roomkit.neural`), randomly initialised by default, and
treated as a feature extractor you can point at a checkpoint.

## STFT inversion with any window

`istft` applies the analysis window a second time on synthesis and divides
by the accumulated squared window. That makes reconstruction exact in the
interior of the signal for *any* window, not just COLA windows, and it is
what lets Griffin-Lim and the codec resynthesis share one code path. The
price is that the first and last `n_fft` samples are only approximately
right, so tests compare interiors.

## Minimum statistics, capped at the lead-in

A bare windowed minimum drifts down and stays down, which is fine for
stationary noise but underestimates badly once speech fills every window.
Since `denoise` already assumes a noise-only lead-in, the tracker is capped
at the lead-in average: it can dip below it inside pauses but never rise
above it. For pause-free speech this is exactly the guarantee you want; for
long recordings with drifting noise floors, re-estimate per chunk.

## WPE regularisation and the stable solve

Per-bin normal equations are frequently ill-conditioned (silent bins, very
short signals). Rather than add a fixed diagonal everywhere and hide the
problem, `_stable_solve` loads the diagonal lightly and falls back to a
per-bin least-squares solve only for bins that are still singular. The
fallback is rare in practice but keeps the function total.

## Hand-crafted diarization embeddings

Speaker embeddings come from log band-energy statistics and the spectral
centroid over ~250 ms windows rather than a trained network. That keeps the
pipeline dependency-free and deterministic, and it separates voices by
pitch and timbre well enough for the segment-then-cluster scheme. The
change-point threshold is deliberately low because overlapping windows cap
how large a cosine jump between neighbours can be.

## SI-SDR uses the projection energy

A common SI-SDR bug is dividing by the *reference* energy instead of the
*projection* energy; the latter is what makes the metric invariant to the
estimate's gain. Our `sdr`/`si_sdr` project the estimate onto the reference
and compare projection energy against the orthogonal residual.

## DER collar alignment

`np.diff` flags the frame *before* a label change, so the no-score collar
is centred on the first frame of the new label and extended symmetrically
by `collar` on both sides. Off-by-one here quietly inflates DER on every
boundary, and there is a regression test pinning the symmetric behaviour.
