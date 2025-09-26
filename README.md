# roomkit

Speech enhancement and room-acoustics toolkit for Python.

Denoising, dereverberation, voice activity detection and speaker
diarization, with an evaluation suite and a command line interface.

## Module overview

- `roomkit.audio` — file IO, resampling, framing / overlap-add
- `roomkit.signal` — windows, STFT, mel spectrograms (growing)
- `roomkit.acoustics` — room impulse response simulation, RT60 (planned)
- `roomkit.denoise` — spectral subtraction, Wiener filtering (planned)
- `roomkit.dereverb` — WPE dereverberation (planned)
- `roomkit.vad` — voice activity detection (planned)
- `roomkit.codec` — neural-codec-style token frontend (planned)
- `roomkit.diarization` — speaker diarization (planned)

Work in progress.

## Quickstart: denoising

```python
import roomkit

noisy, rate = roomkit.load_audio("noisy.wav")
enhanced = roomkit.denoise(noisy)  # expects a short noise-only lead-in
roomkit.save_audio("enhanced.wav", enhanced, rate)
```
