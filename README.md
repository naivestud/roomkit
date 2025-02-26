# roomkit

Speech enhancement and room-acoustics toolkit for Python.

Denoising, dereverberation, voice activity detection and speaker
diarization, with an evaluation suite and a command line interface.

## Module overview

- `roomkit.audio` — file IO, resampling, framing / overlap-add
- `roomkit.signal` — windows, STFT, mel spectrograms (growing)
- `roomkit.room` — room impulse response simulation, RT60 (planned)
- `roomkit.denoise` — spectral subtraction, Wiener filtering (planned)
- `roomkit.dereverb` — WPE dereverberation (planned)
- `roomkit.vad` — voice activity detection (planned)
- `roomkit.codec` — neural-codec-style token frontend (planned)
- `roomkit.diarization` — speaker diarization (planned)

Work in progress.
