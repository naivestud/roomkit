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

## roomkit.vad

- `detect_voice(samples, sample_rate=16000, frame_length=400, hop_length=160,
  threshold_db=6.0, noise_frames=10, hangover_frames=8, min_duration=0.05)
  -> list[Segment]`.
- `energy_vad(samples, frame_length=400, hop_length=160, threshold_db=6.0, noise_frames=10)`.
- `frame_energy_db(samples, frame_length=400, hop_length=160)`.
- `hangover(decisions, hangover_frames=5)`.
- `decisions_to_segments(decisions, sample_rate, hop_length=160, frame_length=400,
  min_duration=0.0, label="speech")`.

## roomkit.codec

- `CodecFrontend(n_codebooks=4, codebook_size=32, sample_rate=16000,
  frame_length=400, hop_length=160, n_mels=40)` with `fit(samples)`,
  `encode(samples) -> tokens`, `decode(tokens) -> features`,
  `synthesize(tokens, n_iter=32, seed=0) -> waveform`, and a `fitted`
  property.
- `codec_features(samples, sample_rate=16000, frame_length=400, hop_length=160, n_mels=40)`.
- `kmeans(data, num_clusters, iterations=25, seed=0) -> (codebook, assignments)`.
- `fit_rvq(data, n_codebooks, codebook_size, iterations=25, seed=0) -> codebooks`.
- `rvq_encode(data, codebooks) -> tokens`, `rvq_decode(tokens, codebooks) -> vectors`.

## roomkit.metrics

- `sdr(reference, estimate)`, `si_sdr(reference, estimate)` — dB.
- `log_spectral_distortion(reference, estimate, n_fft=400, hop_length=160)` — dB.
- `segmental_snr(reference, estimate, frame_length=400, hop_length=160,
  min_snr_db=-10.0, max_snr_db=35.0)` — dB.
- `vad_frame_metrics(reference, estimate) -> {"precision", "recall", "f1", "accuracy"}`.
- `diarization_error_rate(reference, estimate, duration, collar=0.25)` — fraction.

## roomkit.pipeline

- `EnhancementPipeline(do_denoise=True, do_dereverb=False, denoise_method="wiener",
  noise_frames=10, taps=8, delay=3, iterations=3, n_fft=400, hop_length=160)` —
  callable on a waveform.

## roomkit.diarization

- `DiarizationPipeline(sample_rate=16000, frame_length=400, hop_length=160,
  window_frames=25, window_hop=10, threshold=0.06, min_segment=3)` —
  callable with `(samples, num_speakers=None)`.
- `diarize(samples, num_speakers=None, sample_rate=16000)` (also top-level).
- `spectral_embeddings(samples, frame_length=400, hop_length=160,
  window_frames=25, window_hop=10) -> (embeddings, window_starts)`.
- `segment_embeddings(embeddings, threshold=0.06, min_segment=3) -> [(start, end), ...]`.
- `cluster_embeddings(embeddings, num_speakers=None, distance_threshold=0.5) -> labels`.

## roomkit.types

- `Segment(start, end, label="speech")` — frozen dataclass with `.duration`.

## roomkit.neural (optional, requires torch)

- `EncoderConfig(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80,
  d_model=64, n_heads=4, n_layers=2, max_frames=1500)`.
- `WhisperStyleEncoder(config=None)` — `forward(waveform)` returns
  `(batch, frames // 2, d_model)` embeddings.
- `statistics_pool(embeddings, mask=None) -> (batch, 2 * d_model)`.

## Command line

`roomkit` exposes `denoise`, `dereverb`, `vad`, `diarize`, `rt60` and
`codec encode`/`codec decode` subcommands; `--help` on each lists options.
`main(argv=None)` returns a process exit code (0 success, 1 runtime error,
2 no command given).
