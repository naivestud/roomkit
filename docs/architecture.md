# Architecture

roomkit is organised as a stack of small, single-purpose layers. Everything
above the audio IO layer operates on NumPy arrays; the one exception is
`roomkit.neural`, which is built on PyTorch and lives behind an optional
extra.

```
            CLI (roomkit.cli)        pipelines (roomkit.pipeline)
                    │                         │
   ┌────────┬───────┼─────────┬───────────┬──┴────────┬───────────┐
   ▼        ▼       ▼         ▼           ▼           ▼           ▼
 denoise  dereverb  vad   diarization   codec   acoustics     metrics
   │        │       │         │           │        │            │
   └────────┴───────┴────┬────┴───────────┴────────┘            │
                         ▼                                      │
                  signal (STFT, mel, Griffin-Lim)               │
                         │                                      │
                         ▼                                      │
                  audio (IO, resampling, framing)  ◄────────────┘
```

## Design rules

1. **Waveforms in, waveforms out.** Public entry points such as `denoise`,
   `dereverb` and `detect_voice` accept a 1-D float array and return a 1-D
   float array (or `Segment` lists). STFT bookkeeping stays internal.
2. **One framing convention.** The whole stack shares 25 ms frames with a
   10 ms hop at 16 kHz (`n_fft=400`, `hop_length=160`) unless a caller
   overrides it; that matches the whisper frontend and keeps components
   interchangeable.
3. **Deterministic by construction.** Every stochastic algorithm takes a
   `seed` argument and uses `numpy.random.default_rng`; there is no global
   RNG state.
4. **No mandatory heavyweight dependencies.** The base install is
   numpy/scipy/soundfile. Torch is only imported by `roomkit.neural`, so
   importing `roomkit` never pulls a deep-learning stack.

## Module tour

- `roomkit.audio` — `load_audio` / `save_audio` (soundfile), polyphase
  resampling, `frame_signal` / `overlap_add`.
- `roomkit.signal` — windows with COLA checks, an invertible STFT pair,
  mel filterbanks (Slaney normalisation, whisper log scaling) and
  Griffin-Lim phase recovery.
- `roomkit.denoise` — noise-floor estimation (windowed minimum statistics
  capped at the lead-in level), spectral subtraction with oversubtraction
  and spectral flooring, posterior Wiener gain.
- `roomkit.dereverb` — WPE: delayed-tap regressors, weighted normal
  equations per frequency bin, EM-style iterations.
- `roomkit.acoustics` — image-source shoebox RIR simulation, Sabine/Eyring
  RT60 formulas, Schroeder integration with T30/T20 regression.
- `roomkit.vad` — adaptive energy thresholding, hangover smoothing,
  decision-to-segment conversion.
- `roomkit.codec` — log-mel analysis features, k-means++ codebooks,
  residual vector quantization, token-based resynthesis.
- `roomkit.diarization` — spectral window embeddings, cosine change-point
  segmentation, average-linkage clustering.
- `roomkit.metrics` — SI-SDR/SDR, log-spectral distortion, segmental SNR,
  VAD precision/recall/F1, diarization error rate with collar.
- `roomkit.neural` — whisper-style encoder (mel frontend, strided conv
  stem, sinusoidal positions, pre-LN transformer, statistics pooling).

## Data flow of the enhancement pipeline

`EnhancementPipeline` runs denoising first and dereverberation second:
noise removal makes the WPE power estimates more reliable, and WPE then
shortens the remaining tail. Both stages re-use the same STFT parameters so
no resampling or reframing happens between them.
