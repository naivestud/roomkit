import numpy as np
import pytest

from roomkit.audio.io import load_audio, save_audio


def test_roundtrip_mono(tmp_path):
    samples = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    path = tmp_path / "tone.wav"
    save_audio(path, samples, 16000)
    loaded, rate = load_audio(path)
    assert rate == 16000
    assert loaded.shape == samples.shape
    # 16-bit PCM quantizes; allow half an LSB of error.
    assert np.allclose(loaded, samples, atol=2e-4)


def test_roundtrip_multichannel(tmp_path):
    samples = np.zeros((4000, 2), dtype=np.float32)
    samples[:, 0] = 0.25
    samples[:, 1] = -0.25
    path = tmp_path / "stereo.wav"
    save_audio(path, samples, 8000)
    loaded, rate = load_audio(path)
    assert rate == 8000
    assert loaded.shape == (4000, 2)


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "out.wav"
    save_audio(path, np.zeros(100, dtype=np.float32), 16000)
    assert path.exists()


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        load_audio(tmp_path / "nope.wav")
