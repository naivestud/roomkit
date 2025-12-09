# Changelog

All notable changes to roomkit are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/); the
project uses semantic versioning.

## [0.3.0] - 2026-08-08

### Added

- `roomkit.diarization` pipeline and the top-level `roomkit.diarize`
  entry point.
- Evaluation suite in `roomkit.metrics`: SI-SDR/SDR, log-spectral
  distortion, segmental SNR, VAD precision/recall/F1, and diarization
  error rate with a symmetric no-score collar.
- `EnhancementPipeline` chaining denoising and WPE.
- `roomkit` CLI with `denoise`, `dereverb`, `vad`, `diarize`, `rt60` and
  `codec encode`/`codec decode` subcommands.
- Self-contained examples under `examples/` and long-form documentation
  under `docs/`.

### Changed

- Renamed the `room` subpackage to `roomkit.acoustics`.
- VAD segment end times now include the analysis window of the last active
  frame.

### Fixed

- SDR uses the projection energy in the numerator (correct SI-SDR gain
  invariance).
- `estimate_rt60` falls back to a T20 regression for short decays.

## [0.2.0] - 2026-08-08

### Added

- Optional `roomkit.neural` subpackage with a whisper-style encoder
  (torch): log-mel frontend, strided conv stem, sinusoidal positions,
  pre-LN transformer blocks and statistics pooling.
- `roomkit.codec`: discrete token frontend — log-mel analysis features,
  k-means codebooks, residual vector quantization and Griffin-Lim
  resynthesis.
- `roomkit.vad`: adaptive energy VAD with hangover smoothing and segment
  output.
- `roomkit.room`: image-source shoebox RIR simulation (with air
  absorption), Sabine/Eyring RT60 formulas and Schroeder-based RT60
  estimation.

### Changed

- `denoise` now estimates the noise floor from the recording lead-in.

## [0.1.0] - 2026-08-08

### Added

- Core audio utilities: file IO, resampling, framing / overlap-add.
- Signal primitives: windows, STFT / ISTFT, mel filterbanks, Griffin-Lim.
- Single-channel denoising (spectral subtraction, Wiener filtering).
- WPE dereverberation.
