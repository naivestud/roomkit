import pytest

from roomkit.config import AudioConfig


def test_defaults():
    config = AudioConfig()
    assert config.sample_rate == 16000
    assert config.frame_length == 400
    assert config.hop_length == 160


def test_durations():
    config = AudioConfig()
    assert config.frame_duration == pytest.approx(0.025)
    assert config.hop_duration == pytest.approx(0.010)


def test_frozen():
    config = AudioConfig()
    with pytest.raises(Exception):
        config.sample_rate = 8000  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"frame_length": -1},
        {"hop_length": 0},
        {"hop_length": 401},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        AudioConfig(**kwargs)
