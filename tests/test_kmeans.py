import numpy as np
import pytest

from roomkit.codec.kmeans import kmeans


def _blobs(seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [-4.0, 3.0]])
    parts = [center + rng.standard_normal((60, 2)) * 0.3 for center in centers]
    return np.vstack(parts), centers


def test_finds_blob_centers():
    data, centers = _blobs()
    codebook, assignments = kmeans(data, num_clusters=3, seed=1)
    # match each fitted centroid to its closest true centre
    matched = [centers[np.argmin(np.linalg.norm(centers - c, axis=1))] for c in codebook]
    assert np.allclose(codebook, np.array(matched), atol=0.3)
    # points of the same blob share a label
    assert len(set(assignments[:60])) == 1
    assert len(set(assignments[60:120])) == 1
    assert assignments[0] != assignments[60]


def test_deterministic_with_seed():
    data, _ = _blobs(seed=2)
    a, _ = kmeans(data, 3, seed=7)
    b, _ = kmeans(data, 3, seed=7)
    assert np.allclose(a, b)


def test_single_cluster_is_mean():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((50, 4))
    codebook, assignments = kmeans(data, 1)
    assert np.allclose(codebook[0], data.mean(axis=0))
    assert np.all(assignments == 0)


def test_invalid_arguments():
    data = np.ones((10, 2))
    with pytest.raises(ValueError):
        kmeans(data, 0)
    with pytest.raises(ValueError):
        kmeans(data, 11)
    with pytest.raises(ValueError):
        kmeans(np.ones(10), 2)
