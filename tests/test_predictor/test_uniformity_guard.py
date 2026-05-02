from ganyan.predictor import uniformity_guard


def test_uniform_predictions_flagged():
    probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert uniformity_guard.is_uniform(probs) is True


def test_normal_spread_not_flagged():
    probs = [0.35, 0.22, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02]
    assert uniformity_guard.is_uniform(probs) is False


def test_near_uniform_with_floating_point_noise_flagged():
    probs = [0.1 + 1e-6 * i for i in range(10)]
    assert uniformity_guard.is_uniform(probs) is True


def test_empty_predictions_flagged():
    assert uniformity_guard.is_uniform([]) is True


def test_single_prediction_flagged():
    assert uniformity_guard.is_uniform([1.0]) is True


def test_threshold_is_configurable():
    probs = [0.10, 0.11, 0.09, 0.10, 0.10]
    assert uniformity_guard.is_uniform(probs, stddev_threshold=0.001) is False
    assert uniformity_guard.is_uniform(probs, stddev_threshold=0.05) is True


def test_check_race_field_returns_reason_when_uniform():
    probs = [0.1] * 8
    reason = uniformity_guard.check_race_field(race_id=12345, probabilities=probs)
    assert reason is not None
    assert "12345" in reason
    assert "uniform" in reason.lower()


def test_check_race_field_returns_none_when_normal():
    probs = [0.4, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.01]
    assert uniformity_guard.check_race_field(race_id=12345, probabilities=probs) is None
