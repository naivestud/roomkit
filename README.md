# roomkit

[![CI](https://github.com/naivestud/roomkit/actions/workflows/ci.yml/badge.svg)](https://github.com/naivestud/roomkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Speech enhancement and room-acoustics toolkit for Python.

roomkit bundles the pieces needed to clean up and understand room
recordings: single-channel denoising, WPE dereverberation, room impulse
response simulation, voice activity detection and speaker diarization —
plus a neural-audio-codec-style token frontend, a whisper-style encoder, an
evaluation suite and a CLI. The core runs on numpy/scipy alone; torch is an
optional extra.

```python
import roomkit

noisy, rate = roomkit.load_audio("noisy.wav")
enhanced = roomkit.denoise(noisy)
roomkit.save_audio("clean.wav", enhanced, rate)
```

## Features

- **Denoising** — spectral subtraction and posterior Wiener filtering with
  lead-in noise-floor estimation.
- **Dereverberation** — weighted prediction error (WPE), mono or
  multichannel.
- **Room acoustics** — image-source shoebox RIR simulation with air
  absorption, Sabine/Eyring RT60 formulas, Schroeder-integration RT60
  estimation (T30 with T20 fallback).
- **Voice activity detection** — adaptive energy thresholding with hangover
  smoothing, segment output.
- **Speaker diarization** — spectral embeddings, change-point segmentation,
  agglomerative clustering; answers "who spoke when" without model
  downloads.
- **Codec frontend** — log-mel features quantized by residual vector
  quantization into discrete tokens, with Griffin-Lim resynthesis.
- **Whisper-style encoder** (`roomkit.neural`, optional) — mel frontend,
  strided conv stem, sinusoidal positions, pre-LN transformer, statistics
  pooling.
- **Evaluation suite** — SI-SDR/SDR, log-spectral distortion, segmental
  SNR, VAD precision/recall/F1, diarization error rate with collar.
- **CLI** — `roomkit denoise | dereverb | vad | diarize | rt60 | codec`.

## Installation

```bash
pip install .              # core
pip install ".[neural]"    # + torch, roomkit.neural
pip install -e ".[test]"   # development (pytest, ruff, mypy)
```

Requires Python 3.10+.

## Quickstart

Denoise a recording (the file should start with a short noise-only
lead-in):

```python
import roomkit

noisy, rate = roomkit.load_audio("noisy.wav")
enhanced = roomkit.denoise(noisy)  # or method="spectral"
roomkit.save_audio("enhanced.wav", enhanced, rate)
```

Diarize a meeting:

```python
segments = roomkit.diarize(samples, num_speakers=2)
for s in segments:
    print(s.label, f"{s.start:.2f}–{s.end:.2f}s")
```

Chain stages with the enhancement pipeline:

```python
from roomkit.pipeline import EnhancementPipeline

pipeline = EnhancementPipeline(do_denoise=True, do_dereverb=True)
enhanced = pipeline(noisy)
```

Simulate a room and measure it:

```python
from roomkit.acoustics.rir import ShoeboxRoom, simulate_rir
from roomkit.acoustics.rt60 import estimate_rt60

room = ShoeboxRoom(dimensions=(5.0, 4.0, 3.0), absorption=0.2)
rir = simulate_rir(room, source=(1.0, 1.0, 1.5), microphone=(3.5, 2.5, 1.0))
print(f"RT60 ≈ {estimate_rt60(rir, 16000):.2f} s")
```

Command line:

```bash
roomkit denoise noisy.wav clean.wav
roomkit diarize meeting.wav --num-speakers 2 --output who.json
roomkit rt60 measured_rir.wav
```

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)
- [Changelog](CHANGELOG.md)

Runnable, self-contained scripts live in [`examples/`](examples/).

## Development

```bash
pytest -q          # tests (seeded, no network)
ruff check .       # lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions.

## License

MIT — see [LICENSE](LICENSE).
