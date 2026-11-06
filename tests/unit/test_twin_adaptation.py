"""Tests for ai/twintrust_ap/twin_adaptation.py (age bridging, twin-aided beam tracking)."""

import math

import pytest

from ai.twintrust_ap.twin_adaptation import (
    angle_difference,
    bearing_to,
    bridge_horizon,
    track_beams,
)

ALLOWED = [1, 2, 3, 5, 8, 10]  # configs/model.yaml twintrust_ap TAHS discrete horizons


@pytest.mark.parametrize(
    "horizon,age,expected",
    [
        (1, 0.05, 1),  # fresh information: unchanged
        (1, 0.1, 1),  # exactly one step
        (1, 0.11, 2),
        (2, 0.35, 5),  # 0.35 s needs 4 steps -> next allowed horizon is 5
        (8, 0.35, 8),  # never lowers the horizon
        (1, 0.9, 10),
        (3, 2.5, 10),  # older than the longest horizon: the longest one
        (5, None, 5),  # no V2X information
    ],
)
def test_bridge_horizon(horizon: int, age: float | None, expected: int) -> None:
    assert bridge_horizon(horizon, age, ALLOWED) == expected


def test_bridge_horizon_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        bridge_horizon(1, -0.1, ALLOWED)
    with pytest.raises(ValueError):
        bridge_horizon(1, float("nan"), ALLOWED)
    with pytest.raises(ValueError):
        bridge_horizon(1, 0.2, [])
    with pytest.raises(ValueError):
        bridge_horizon(1, 0.2, ALLOWED, step_s=0)


def test_bridged_horizon_covers_age() -> None:
    for age_ms in range(0, 1001, 7):
        age = age_ms / 1000
        h = bridge_horizon(1, age, ALLOWED)
        assert h in ALLOWED
        assert h * 0.1 >= age - 1e-9 or h == max(ALLOWED)


def test_bearing_convention_matches_demo() -> None:
    # demo: forward is -z for heading 0, so a point straight ahead has bearing 0
    assert bearing_to((0, 0), 0.0, (0, -10)) == pytest.approx(0.0)
    assert bearing_to((0, 0), 0.0, (10, 0)) == pytest.approx(math.pi / 2)
    assert bearing_to((0, 0), math.pi / 2, (10, 0)) == pytest.approx(0.0)


def test_angle_difference_wraps() -> None:
    assert angle_difference(0.1, -0.1) == pytest.approx(0.2)
    assert angle_difference(math.pi - 0.05, -math.pi + 0.05) == pytest.approx(0.1)
    assert angle_difference(1.0, 1.0) == 0.0


def test_track_beams_steers_to_prediction() -> None:
    beams = {"a": 0.0, "b": 0.3, "c": 0.5}
    pred = {"a": (10.0, 0.0), "b": (0.0, -10.0)}  # no prediction for c
    out = track_beams(beams, pred, (0.0, 0.0), 0.0, isolated={"b"})
    assert out["a"] == pytest.approx(math.pi / 2)  # steered
    assert out["b"] == 0.3  # isolated: kept
    assert out["c"] == 0.5  # no prediction: kept
    assert beams == {"a": 0.0, "b": 0.3, "c": 0.5}  # input not modified


def test_track_beams_does_not_acquire_new_vehicles() -> None:
    out = track_beams({}, {"a": (1.0, 1.0)}, (0.0, 0.0), 0.0)
    assert out == {}


def test_tracking_reduces_misalignment_of_moving_vehicle() -> None:
    """A vehicle crossing in front of the ego: the swept bearing drifts, the tracked one follows."""
    swept = bearing_to((0, 0), 0.0, (-5.0, -20.0))
    tracked = {"v": swept}
    for k in range(1, 11):  # 1 s at 10 Hz, 10 m/s to the right
        true_xz = (-5.0 + k, -20.0)
        predicted_xz = (true_xz[0] + 0.3, true_xz[1])  # twin prediction 0.3 m off
        tracked = track_beams(tracked, {"v": predicted_xz}, (0, 0), 0.0)
        truth = bearing_to((0, 0), 0.0, true_xz)
        assert angle_difference(tracked["v"], truth) < angle_difference(swept, truth) or k == 0
    assert angle_difference(swept, truth) > math.radians(12) / 2  # untracked beam is misaligned
    assert angle_difference(tracked["v"], truth) < math.radians(1)
