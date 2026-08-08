import numpy as np
import pytest

from roomkit.audio.framing import frame_signal, overlap_add


def test_frame_shape():
    frames = frame_signal(np.arange(10, dtype=float), frame_length=4, hop_length=2)
    assert frames.shape == (4, 4)
    assert np.array_equal(frames[0], [0, 1, 2, 3])
    assert np.array_equal(frames[1], [2, 3, 4, 5])


def test_short_signal_is_padded():
    frames = frame_signal(np.array([1.0, 2.0]), frame_length=4, hop_length=2)
    assert frames.shape == (1, 4)
    assert np.array_equal(frames[0], [1, 2, 0, 0])


def test_overlap_add_no_window():
    frames = np.arange(8, dtype=float).reshape(2, 4)
    out = overlap_add(frames, hop_length=4)
    assert np.array_equal(out, np.arange(8))


def test_overlap_add_with_window_inverts_weighting():
    # a single frame weighted by the window should come back unweighted
    frame = np.array([1.0, 2.0, 3.0, 4.0])
    window = np.array([0.5, 1.0, 1.0, 0.5])
    out = overlap_add(frame[None, :] , hop_length=4, window=window)
    assert np.allclose(out, frame)


def test_invalid_dims():
    with pytest.raises(ValueError):
        frame_signal(np.zeros((2, 2)), 4, 2)
