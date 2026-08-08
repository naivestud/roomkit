# Examples

Self-contained scripts. Each one generates its own synthetic audio, so
there is nothing to download — just run them from the repository root with
roomkit installed (`pip install -e .`).

| Script | Shows |
| ------ | ----- |
| [`denoise_wav.py`](denoise_wav.py) | Denoising a noisy recording with both methods; SI-SDR before/after. |
| [`simulate_room.py`](simulate_room.py) | Image-source RIR simulation, RT60 estimation, WPE dereverberation. |
| [`vad_and_diarize.py`](vad_and_diarize.py) | Voice activity detection and two-speaker diarization scored with DER. |
| [`codec_tokens.py`](codec_tokens.py) | RVQ codec frontend: token budget vs. fidelity, Griffin-Lim resynthesis. |

```bash
python examples/denoise_wav.py
python examples/simulate_room.py
python examples/vad_and_diarize.py
python examples/codec_tokens.py
```

The `neural` extra is not needed by any of these; they all run on the
numpy-only core.
