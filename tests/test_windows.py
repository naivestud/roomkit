import numpy as np
import pytest

from roomkit.signal.windows import get_window, is_cola


def test_hann_window_shape_and_symmetry():
    window = get_window("hann", 400)
    assert window.shape == (400,)
    assert np.allclose(window, window[::-1])


def test_hann_endpoints_are_zero():
    window = get_window("hann", 400)
    assert window[0] == pytest.approx(0.0, abs=1e-12)
    assert window[-1] == pytest.approx(0.0, abs=1e-12)


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        get_window("kaiser", 16)


def test_rect_window_is_cola():
    assert is_cola(get_window("ones", 400), 160)


def test_hann_is_cola_at_half_hop_with_loose_tolerance():
    # the symmetric (not periodic) hann window is only approximately COLA
    assert is_cola(get_window("hann", 400), 200, tolerance=1e-2)
