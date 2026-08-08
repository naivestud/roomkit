# Changelog

All notable changes to roomkit are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/); the
project uses semantic versioning.

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
