import numpy as np

from roomkit.metrics.vad_metrics import vad_frame_metrics


def test_perfect_decisions():
    reference = np.array([False, True, True, False, True])
    metrics = vad_frame_metrics(reference, reference)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_all_false_estimates():
    reference = np.array([True, True, False])
    estimate = np.array([False, False, False])
    metrics = vad_frame_metrics(reference, estimate)
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0  # nothing predicted


def test_false_positives_lower_precision():
    reference = np.array([True, False, False, False])
    estimate = np.array([True, True, True, False])
    metrics = vad_frame_metrics(reference, estimate)
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1 / 3


def test_no_speech_in_reference():
    reference = np.array([False, False])
    estimate = np.array([False, False])
    metrics = vad_frame_metrics(reference, estimate)
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0
