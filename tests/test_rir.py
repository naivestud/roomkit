import numpy as np
import pytest

from roomkit.room.rir import SPEED_OF_SOUND, ShoeboxRoom, simulate_rir


def _room():
    return ShoeboxRoom(dimensions=(4.0, 3.0, 2.5), absorption=0.3)


def test_direct_path_arrival_time():
    room = _room()
    source = (1.0, 1.0, 1.0)
    mic = (2.5, 1.5, 1.2)
    fs = 16000
    response = simulate_rir(room, source, mic, sample_rate=fs, max_order=0)
    distance = np.linalg.norm(np.array(source) - np.array(mic))
    expected = int(round(distance / SPEED_OF_SOUND * fs))
    assert int(np.argmax(np.abs(response))) == expected


def test_peak_normalised_to_one():
    response = simulate_rir(_room(), (1.0, 1.0, 1.0), (2.0, 2.0, 1.0), max_order=1)
    assert np.max(np.abs(response)) == pytest.approx(1.0)


def test_higher_order_longer_response():
    room = _room()
    args = (room, (1.0, 1.0, 1.0), (3.0, 2.0, 1.5))
    short = simulate_rir(*args, max_order=0)
    long = simulate_rir(*args, max_order=3)
    assert long.shape[0] > short.shape[0]
    # more reflections mean more non-zero taps
    assert np.count_nonzero(long) > np.count_nonzero(short)


def test_air_absorption_reduces_far_taps():
    room = _room()
    args = (room, (0.5, 0.5, 0.5), (3.5, 2.5, 2.0), 16000, 2)
    dry = simulate_rir(*args, air_absorption=0.0)
    wet = simulate_rir(*args, air_absorption=5e-3)
    # normalisation hides the direct path change; compare tail energy
    direct = int(np.argmax(np.abs(dry)))
    assert np.sum(wet[direct + 1 :] ** 2) < np.sum(dry[direct + 1 :] ** 2)


def test_outside_room_raises():
    with pytest.raises(ValueError):
        simulate_rir(_room(), (5.0, 1.0, 1.0), (2.0, 2.0, 1.0))


def test_invalid_room_dimensions():
    with pytest.raises(ValueError):
        ShoeboxRoom(dimensions=(0.0, 3.0, 2.5))
