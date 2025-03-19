import pytest

from roomkit.room.sabine import absorption_from_rt60, eyring_rt60, sabine_rt60


def test_sabine_known_value():
    # 50 m^3 room, 80 m^2 surface, alpha = 0.2 -> 0.161 * 50 / 16
    assert sabine_rt60(50.0, 80.0, 0.2) == pytest.approx(0.503125)


def test_more_absorption_shorter_decay():
    assert sabine_rt60(50.0, 80.0, 0.5) < sabine_rt60(50.0, 80.0, 0.1)


def test_eyring_close_to_sabine_for_small_absorption():
    sabine = sabine_rt60(120.0, 150.0, 0.05)
    eyring = eyring_rt60(120.0, 150.0, 0.05)
    assert eyring == pytest.approx(sabine, rel=0.05)


def test_eyring_shorter_than_sabine_for_absorptive_rooms():
    assert eyring_rt60(120.0, 150.0, 0.6) < sabine_rt60(120.0, 150.0, 0.6)


def test_absorption_roundtrip():
    alpha = absorption_from_rt60(50.0, 80.0, sabine_rt60(50.0, 80.0, 0.3))
    assert alpha == pytest.approx(0.3)


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (sabine_rt60, {"volume": 0, "surface_area": 80, "absorption": 0.2}),
        (sabine_rt60, {"volume": 50, "surface_area": 80, "absorption": 0.0}),
        (eyring_rt60, {"volume": 50, "surface_area": 80, "absorption": 1.0}),
        (absorption_from_rt60, {"volume": 50, "surface_area": 80, "rt60": -1.0}),
    ],
)
def test_invalid_arguments_raise(fn, kwargs):
    with pytest.raises(ValueError):
        fn(**kwargs)
