import numpy as np
import pytest

from roomkit.codec.rvq import fit_rvq, rvq_decode, rvq_encode


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    # smooth-ish vectors so quantization can actually fit them
    base = rng.standard_normal((400, 16))
    return base + 0.1 * rng.standard_normal((400, 16))


def test_residual_error_decreases_per_stage():
    data = _data()
    codebooks = fit_rvq(data, n_codebooks=4, codebook_size=16, seed=0)
    errors = []
    for depth in range(1, 5):
        tokens = rvq_encode(data, codebooks[:depth])
        reconstruction = rvq_decode(tokens, codebooks[:depth])
        errors.append(float(np.mean((data - reconstruction) ** 2)))
    assert all(
        later < earlier for earlier, later in zip(errors, errors[1:], strict=False)
    )


def test_token_shapes_and_ranges():
    data = _data(seed=1)
    codebooks = fit_rvq(data, n_codebooks=3, codebook_size=8, seed=2)
    tokens = rvq_encode(data, codebooks)
    assert tokens.shape == (400, 3)
    assert tokens.min() >= 0
    assert tokens.max() < 8


def test_encode_decode_deterministic():
    data = _data(seed=3)
    codebooks = fit_rvq(data, n_codebooks=2, codebook_size=8, seed=4)
    tokens = rvq_encode(data, codebooks)
    assert np.array_equal(rvq_decode(tokens, codebooks), rvq_decode(tokens, codebooks))


def test_invalid_inputs():
    data = _data()
    with pytest.raises(ValueError):
        fit_rvq(data, n_codebooks=0, codebook_size=8)
    codebooks = fit_rvq(data, n_codebooks=2, codebook_size=8)
    with pytest.raises(ValueError):
        rvq_decode(np.zeros((10, 5), dtype=int), codebooks)
