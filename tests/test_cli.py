import json

import numpy as np

from roomkit.audio.io import save_audio
from roomkit.cli import main


def _write_noisy(tmp_path, sample_rate=16000):
    rng = np.random.default_rng(0)
    t = np.arange(int(1.0 * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    noisy = tone + 0.05 * rng.standard_normal(t.shape)
    path = tmp_path / "noisy.wav"
    save_audio(path, noisy.astype(np.float32), sample_rate)
    return path


def test_denoise_end_to_end(tmp_path):
    noisy = _write_noisy(tmp_path)
    out = tmp_path / "clean.wav"
    assert main(["denoise", str(noisy), str(out)]) == 0
    assert out.exists()


def test_dereverb_end_to_end(tmp_path):
    noisy = _write_noisy(tmp_path)
    out = tmp_path / "dry.wav"
    assert main(["dereverb", str(noisy), str(out), "--iterations", "1"]) == 0
    assert out.exists()


def test_vad_writes_json(tmp_path):
    noisy = _write_noisy(tmp_path)
    out = tmp_path / "vad.json"
    assert main(["vad", str(noisy), "--output", str(out)]) == 0
    segments = json.loads(out.read_text())
    assert isinstance(segments, list)


def test_help_exits_with_code_two():
    assert main([]) == 2
