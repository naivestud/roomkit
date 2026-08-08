# API reference

Reference for the public surface. Signatures show defaults; docstrings in
the source are authoritative.

## Top level

| Name | Description |
| ---- | ----------- |
| `roomkit.load_audio(path, dtype="float32") -> (samples, rate)` | Read an audio file. |
| `roomkit.save_audio(path, samples, rate) -> Path` | Write an audio file, creating parent dirs. |
| `roomkit.denoise(samples, method="wiener", ...)` | STFT-domain denoising. |
| `roomkit.diarize(samples, num_speakers=None, sample_rate=16000)` | Speaker diarization. |
| `roomkit.__version__` | Package version string. |

## roomkit.config

- `AudioConfig(sample_rate=16000, frame_length=400, hop_length=160)` —
  frozen dataclass; exposes `frame_duration` and `hop_duration`.

## roomkit.audio

- `load_audio`, `save_audio` (re-exported from `roomkit.audio.io`).
- `resample(samples, source_rate, target_rate)` — polyphase resampling.
- `frame_signal(samples, frame_length, hop_length)` — overlapping frames.
- `overlap_add(frames, hop_length, window=None)` — reassemble frames.

## roomkit.signal

- `get_window(name, length)`, `is_cola(window, hop_length, tolerance=1e-6)`.
- `stft(samples, n_fft=400, hop_length=160, window="hann")`.
- `istft(frames, hop_length=160, window="hann", length=None)`.
- `hz_to_mel(freqs)`, `mel_to_hz(mels)`.
- `mel_filterbank(n_mels, n_fft, sample_rate, f_min=0.0, f_max=None)`.
- `log_mel_spectrogram(samples, sample_rate=16000, n_mels=80, ...)`.
- `griffin_lim(magnitude, hop_length=160, window="hann", n_iter=32, seed=0)`.

## roomkit.denoise

- `denoise(samples, method="wiener", n_fft=400, hop_length=160, noise_frames=10)`.
- `estimate_noise(magnitudes, initial_frames=10, window=30)`.
- `spectral_subtraction(noisy, noise_power, oversubtraction=1.0, spectral_floor=0.0)`.
- `wiener_gain(magnitudes, noise_power, gain_floor=0.0)`.

## roomkit.dereverb

- `dereverb(samples, taps=8, delay=3, iterations=3, n_fft=400, hop_length=160)`.
- `wpe(spectrum, taps=8, delay=3, iterations=3, psd_context=0, regularization=1e-6)`.

## roomkit.acoustics

- `ShoeboxRoom(dimensions, absorption=0.3)` — `.volume`, `.surface_area`.
- `simulate_rir(room, source, microphone, sample_rate=16000, max_order=2, air_absorption=0.0)`.
- `sabine_rt60(volume, surface_area, absorption)`.
- `eyring_rt60(volume, surface_area, absorption)`.
- `absorption_from_rt60(volume, surface_area, rt60)`.
- `schroeder_integration(impulse_response, noise_floor=None)`.
- `estimate_rt60(impulse_response, sample_rate, lower_db=-5.0, upper_db=-35.0, noise_floor=None)`.
