import pytest

torch = pytest.importorskip("torch")

from roomkit.neural import EncoderConfig, WhisperStyleEncoder  # noqa: E402


def _small_encoder():
    torch.manual_seed(0)
    config = EncoderConfig(n_mels=40, d_model=32, n_heads=2, n_layers=1, max_frames=400)
    return WhisperStyleEncoder(config).eval()


def test_encoder_output_shape():
    encoder = _small_encoder()
    waveform = torch.randn(2, 8000)
    with torch.no_grad():
        out = encoder(waveform)
    mel_frames = 1 + (8000 - encoder.config.n_fft) // encoder.config.hop_length
    assert out.shape == (2, (mel_frames - 1) // 2 + 1, encoder.config.d_model)


def test_encoder_is_deterministic():
    encoder = _small_encoder()
    waveform = torch.randn(1, 4000)
    with torch.no_grad():
        first = encoder(waveform)
        second = encoder(waveform)
    assert torch.allclose(first, second)


def test_single_waveform_gets_batch_dim():
    encoder = _small_encoder()
    with torch.no_grad():
        out = encoder(torch.randn(4000))
    assert out.ndim == 3
    assert out.shape[0] == 1


def test_config_validation():
    with pytest.raises(ValueError):
        EncoderConfig(d_model=31)
    with pytest.raises(ValueError):
        EncoderConfig(d_model=32, n_heads=3)
