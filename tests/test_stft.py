import numpy as np

from roomkit.signal.stft import istft, stft


def test_output_shape():
    samples = np.random.default_rng(0).standard_normal(4000)
    spec = stft(samples, n_fft=400, hop_length=160)
    expected_frames = 1 + (4000 - 400) // 160
    assert spec.shape == (expected_frames, 201)


def test_roundtrip_rect_window_exact():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(3200)
    spec = stft(samples, n_fft=400, hop_length=400, window="ones")
    out = istft(spec, hop_length=400, window="ones", length=samples.shape[0])
    assert np.allclose(out, samples, atol=1e-10)


def test_roundtrip_hann_interior():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal(8000)
    spec = stft(samples, n_fft=400, hop_length=160)
    out = istft(spec, hop_length=160, length=samples.shape[0])
    interior = slice(400, 7600)
    assert np.allclose(out[interior], samples[interior], atol=1e-8)


def test_dc_bin_matches_sum():
    samples = np.ones(800)
    spec = stft(samples, n_fft=400, hop_length=400, window="ones")
    # DC bin of an all-ones frame equals the frame length.
    assert np.allclose(spec[:, 0], 400.0)
