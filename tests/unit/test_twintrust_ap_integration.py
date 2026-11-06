"""Tests for the TwinTrust-AP integration pieces: Platt calibration, evidence
normalisation, the TAHS full-range profile and the table-driven FSDP.

The "demo reference" values below were produced by the live-drive demo's
JavaScript port (Logapriya05/aegis-v2x-live-drive, research/port/
twintrust_core.js) with the calibration in configs/model.yaml, so these tests
also check that the Python code and the demo give the same numbers.
"""

from __future__ import annotations

import json
import math

import pytest

from ai.trust_estimator.calibration import (
    PlattCalibration,
    calibrated_trust_from_evidence,
    fit_platt,
)
from ai.trust_estimator.model import TwinTrustEstimator
from ai.twintrust_ap.evidence import (
    EvidenceScales,
    criticality_features,
    trust_evidence,
    twin_correctness,
)
from ai.twintrust_ap.fsdp import (
    CommunicationAction,
    CriticalityBin,
    TableFSDP,
    TrustBin,
    load_policy_table,
)
from ai.twintrust_ap.tahs import TAHS, TAHSParams
from ai.utils.config_loader import REPO_ROOT, load_config

# (e, u, a, q, criticality, authority, raw score, calibrated T, horizon with the full_range profile)
DEMO_REFERENCE = [
    (0.1, 0.05, 0.2, 0.9, 0.2, 0.8, 0.0925, 0.7241811337436818, 10),
    (0.5, 0.3, 0.6, 0.4, 0.7, 0.1, -0.29, 0.4260087094805974, 2),
    (0.32, 0.1, 0.19, 0.83, 0.44, 0.58, -0.008999999999999987, 0.6524958833724365, 8),
    (0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.7892457341320214, 10),
    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -0.8, 0.12102529172052315, 1),
]
GRID = [i / 4 for i in range(5)]


# ---------------------------------------------------------------- calibration
def test_raw_score_is_what_compute_trust_from_evidence_squashes():
    est = TwinTrustEstimator()
    for e in GRID:
        for q in GRID:
            raw = est.compute_raw_score_from_evidence(e, 0.3, 0.2, q)
            assert est.compute_trust_from_evidence(e, 0.3, 0.2, q) == pytest.approx(
                1 / (1 + math.exp(-raw / est.tau)), abs=1e-15
            )


def test_identity_calibration_reproduces_the_uncalibrated_estimator():
    est = TwinTrustEstimator()
    ident = PlattCalibration.identity(est.tau)
    for e in GRID:
        for u in GRID:
            for q in GRID:
                assert calibrated_trust_from_evidence(est, ident, e, u, 0.4, q) == pytest.approx(
                    est.compute_trust_from_evidence(e, u, 0.4, q), abs=1e-12
                )


def test_config_calibration_matches_demo_reference():
    est, cal = TwinTrustEstimator(), PlattCalibration.from_model_config()
    for e, u, a, q, _c, _A, raw, t_cal, _h in DEMO_REFERENCE:
        assert est.compute_raw_score_from_evidence(e, u, a, q) == pytest.approx(raw, abs=1e-12)
        assert calibrated_trust_from_evidence(est, cal, e, u, a, q) == pytest.approx(
            t_cal, abs=1e-12
        )


def test_calibrated_trust_is_monotone_in_the_evidence():
    est, cal = TwinTrustEstimator(), PlattCalibration.from_model_config()
    for lo, hi in zip(GRID, GRID[1:]):
        assert calibrated_trust_from_evidence(
            est, cal, lo, 0.2, 0.2, 0.5
        ) > calibrated_trust_from_evidence(est, cal, hi, 0.2, 0.2, 0.5)
        assert calibrated_trust_from_evidence(
            est, cal, 0.2, 0.2, 0.2, hi
        ) > calibrated_trust_from_evidence(est, cal, 0.2, 0.2, 0.2, lo)


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0), (1.0, float("inf"))])
def test_invalid_calibration_parameters_are_rejected(a, b):
    with pytest.raises(ValueError):
        PlattCalibration(a=a, b=b)


def test_fit_platt_recovers_known_parameters():
    raws = [-0.8 + 1.0 * i / 99 for i in range(100)]
    true = PlattCalibration(a=3.0, b=0.5)
    fitted = fit_platt(raws, [true.apply(x) for x in raws], tau=0.8)
    assert fitted.a == pytest.approx(3.0, abs=1e-6)
    assert fitted.b == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------- evidence
def test_evidence_scales_match_config_defaults():
    assert EvidenceScales.from_model_config() == EvidenceScales()


def test_trust_evidence_normalisation_and_clipping():
    s = EvidenceScales()
    ev = trust_evidence(1.5, [1.0, 1.2, 1.4, 1.6], 0.25, 0.9, 60.0, s)
    assert ev.prediction_error == pytest.approx(0.5)
    assert ev.sync_age_penalty == pytest.approx(0.25)
    assert ev.comm_quality == pytest.approx(0.9 * 0.8)
    assert 0.0 <= ev.prediction_uncertainty <= 1.0
    worst = trust_evidence(30.0, [0.0, 30.0, 60.0], 5.0, 1.2, 900.0, s)
    assert (worst.prediction_error, worst.sync_age_penalty, worst.comm_quality) == (1.0, 1.0, 0.0)


def test_trust_evidence_rejects_bad_input():
    s = EvidenceScales()
    with pytest.raises(ValueError):
        trust_evidence(-1.0, [1.0, 2.0], 0.1, 1.0, 10.0, s)
    with pytest.raises(ValueError):
        trust_evidence(1.0, [1.0], 0.1, 1.0, 10.0, s)
    with pytest.raises(ValueError):
        trust_evidence(float("nan"), [1.0, 2.0], 0.1, 1.0, 10.0, s)


def test_criticality_features():
    s = EvidenceScales()
    ev = trust_evidence(0.9, [0.8, 1.0], 0.3, 1.0, 30.0, s)
    f = criticality_features(10.0, 1, 4, ev, 10, s)
    assert f.relative_speed == pytest.approx(0.5)
    assert f.blockage_probability == pytest.approx(0.25)
    assert f.sync_age == ev.sync_age_penalty
    assert f.channel_degradation == pytest.approx(1 - ev.comm_quality)
    assert f.traffic_density == pytest.approx(0.5)
    assert criticality_features(-3.0, 0, 0, ev, 0, s).blockage_probability == 0.0
    with pytest.raises(ValueError):
        criticality_features(1.0, 3, 2, ev, 0, s)


def test_twin_correctness():
    s = EvidenceScales()
    assert twin_correctness([0.2, 0.99, 1.0, 3.0], s) == 0.5
    assert twin_correctness([], s) is None


# ---------------------------------------------------------------- TAHS profile
def test_default_tahs_coefficients_are_unchanged_and_reach_only_3_to_8():
    p = TAHSParams.from_model_config()
    assert (p.beta, p.gamma, p.delta) == (1.0, 1.0, 1.0)
    tahs = TAHS(p)
    hs = {tahs.select_horizon(t, c, a) for t in GRID for c in GRID for a in GRID}
    assert min(hs) == 3 and max(hs) == 8


def test_full_range_profile_reaches_1_to_10_and_keeps_neutral_point():
    tahs = TAHS(TAHSParams.from_model_config("full_range"))
    hs = {tahs.select_horizon(t, c, a) for t in GRID for c in GRID for a in GRID}
    assert min(hs) == 1 and max(hs) == 10
    assert tahs.select_horizon(0.5, 0.5, 0.5) == 5


def test_full_range_profile_matches_demo_reference():
    tahs = TAHS(TAHSParams.from_model_config("full_range"))
    for *_ev, c, a, _raw, t_cal, h in DEMO_REFERENCE:
        assert tahs.select_horizon(t_cal, c, a) == h


def test_unknown_tahs_profile_raises():
    with pytest.raises(KeyError):
        TAHSParams.from_model_config("does_not_exist")


# ---------------------------------------------------------------- FSDP table
def test_shipped_policy_table_is_complete_and_valid():
    path = REPO_ROOT / load_config("model")["fsdp"]["policy_table_path"]
    table = load_policy_table(path)
    assert len(table) == 9
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "provenance" in data


def test_table_fsdp_bins_and_lookup_are_deterministic():
    fsdp = TableFSDP()
    assert fsdp.discretize(0.39, 0.0) == (TrustBin.LOW, CriticalityBin.LOW)
    assert fsdp.discretize(0.4, 0.5) == (TrustBin.MEDIUM, CriticalityBin.MEDIUM)
    assert fsdp.discretize(0.6, 1.0) == (TrustBin.HIGH, CriticalityBin.HIGH)
    for t in GRID:
        for c in GRID:
            first = fsdp.decide(t, c)
            assert isinstance(first, CommunicationAction)
            assert all(fsdp.decide(t, c) == first for _ in range(3))


def test_table_fsdp_rejects_out_of_range_values():
    fsdp = TableFSDP()
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            fsdp.discretize(bad, 0.5)


def test_load_policy_table_rejects_incomplete_or_unknown_actions(tmp_path):
    full = {t: {c: "predict" for c in ("low", "medium", "high")} for t in ("low", "medium", "high")}
    missing = json.loads(json.dumps(full))
    del missing["high"]["high"]
    unknown = json.loads(json.dumps(full))
    unknown["low"]["low"] = "teleport"
    for bad in (missing, unknown, {"not_policy": full}):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"policy": bad} if "not_policy" not in bad else bad))
        with pytest.raises(ValueError):
            load_policy_table(p)
