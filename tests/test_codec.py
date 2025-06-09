import numpy as np
import pytest

from roomkit.codec.features import codec_features
from roomkit.codec.frontend import CodecFrontend


def _audio(seed: int = 0, seconds: float = 1.0, fs: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    return tone + 0.01 * rng.standard_normal(t.shape)


def test_encode_token_shape_and_range():
    audio = _audio()
    frontend = CodecFrontend(n_codebooks=3, codebook_size=16).fit(audio)
    tokens = frontend.encode(audio)
    expected_frames = 1 + (len(audio) - 400) // 160
    assert tokens.shape == (expected_frames, 3)
    assert tokens.min() >= 0
    assert tokens.max() < 16


def test_decode_reconstructs_features():
    audio = _audio(seed=1)
    frontend = CodecFrontend(n_codebooks=4, codebook_size=16).fit(audio)
    tokens = frontend.encode(audio)
    features = frontend.decode(tokens)
    assert features.shape[1] == frontend.n_mels
    # more codebooks should reconstruct the fitted audio well
    reference = codec_features(audio, n_mels=frontend.n_mels)
    assert np.mean((features - reference) ** 2) < 0.1


def test_encode_before_fit_raises():
    frontend = CodecFrontend()
    with pytest.raises(RuntimeError):
        frontend.encode(_audio())
    with pytest.raises(RuntimeError):
        frontend.decode(np.zeros((4, 4), dtype=int))
