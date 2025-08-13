import pytest

from roomkit.metrics.der import diarization_error_rate
from roomkit.types import Segment


def _ref():
    return [Segment(0.0, 1.0, "alice"), Segment(1.0, 2.0, "bob")]


def test_perfect_hypothesis_scores_zero():
    assert diarization_error_rate(_ref(), _ref(), duration=2.0) == pytest.approx(0.0)


def test_permuted_labels_still_zero():
    # speaker identities are arbitrary; alignment must absorb the swap
    hyp = [Segment(0.0, 1.0, "bob"), Segment(1.0, 2.0, "alice")]
    assert diarization_error_rate(_ref(), hyp, duration=2.0) == pytest.approx(0.0)


def test_missed_speech_counts():
    hyp = [Segment(0.0, 1.5, "alice")]
    der = diarization_error_rate(_ref(), hyp, duration=2.0, collar=0.0)
    assert der == pytest.approx(0.25, abs=0.02)


def test_false_alarm_counts():
    hyp = _ref() + [Segment(2.0, 2.5, "bob")]
    der = diarization_error_rate(_ref(), hyp, duration=2.5, collar=0.0)
    assert der > 0.15


def test_collar_ignores_boundary_errors():
    hyp = [Segment(0.0, 1.1, "alice"), Segment(1.1, 2.0, "bob")]
    assert diarization_error_rate(_ref(), hyp, duration=2.0, collar=0.0) > 0
    assert diarization_error_rate(_ref(), hyp, duration=2.0, collar=0.25) == pytest.approx(0.0)


def test_invalid_duration():
    with pytest.raises(ValueError):
        diarization_error_rate(_ref(), _ref(), duration=0.0)
