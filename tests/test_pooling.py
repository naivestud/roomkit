import pytest

torch = pytest.importorskip("torch")

from roomkit.neural import statistics_pool  # noqa: E402


def test_output_shape():
    embeddings = torch.randn(3, 10, 16)
    pooled = statistics_pool(embeddings)
    assert pooled.shape == (3, 32)


def test_padding_invariance_with_mask():
    torch.manual_seed(1)
    real = torch.randn(1, 6, 8)
    padded = torch.cat([real, torch.zeros(1, 4, 8)], dim=1)
    mask = torch.tensor([[True] * 6 + [False] * 4])
    assert torch.allclose(statistics_pool(real), statistics_pool(padded, mask))


def test_matches_mean_std():
    torch.manual_seed(2)
    embeddings = torch.randn(2, 12, 5)
    pooled = statistics_pool(embeddings)
    assert torch.allclose(pooled[:, :5], embeddings.mean(dim=1), atol=1e-6)
    assert torch.allclose(pooled[:, 5:], embeddings.std(dim=1), atol=1e-5)


def test_invalid_shape():
    with pytest.raises(ValueError):
        statistics_pool(torch.randn(4, 8))
