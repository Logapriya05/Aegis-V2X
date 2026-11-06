"""
Platt calibration of the Calibrated Twin Trust Estimator.

`TwinTrustEstimator.compute_trust_from_evidence` returns

    T_t = sigmoid(raw_score / tau)

with fixed architecture defaults (weights, tau = 0.8). Calibration keeps
raw_score and the estimator unchanged and fits two numbers:

    T_cal = sigmoid(a * raw_score + b)

so that T_cal matches the observed probability that the Digital Twin is
correct. The uncalibrated estimator is the special case a = 1/tau, b = 0
(`PlattCalibration.identity(tau)`).

WHERE THE FITTED VALUES COME FROM (read before citing them)
-----------------------------------------------------------
The values in `configs/model.yaml` -> `trust_estimator.platt_calibration`
were fitted on recorded TwinTrust-AP cycles from the Aegis-V2X live-drive
browser simulation (repository Logapriya05/aegis-v2x-live-drive,
research/pipeline/calibrate_trust.py), not on CARLA or real vehicles:

  - target: the twin object is within 1.0 m of where the ego's LiDAR sees
    it on the next cycle (fractional, averaged over verifiable objects);
  - 8,000 training cycles, 3,992 held-out cycles, 8 conditions;
  - held-out ECE (5 bands): 0.1288 uncalibrated -> 0.0419 calibrated.

They are valid for that simulation. Re-fit them (`fit_platt`) on any other
data source before reporting a calibration result for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from ai.trust_estimator.model import TwinTrustEstimator
from ai.utils.config_loader import load_config


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class PlattCalibration:
    """T_cal = sigmoid(a * raw_score + b)."""

    a: float
    b: float
    source: str = ""

    def __post_init__(self) -> None:
        for name, v in (("a", self.a), ("b", self.b)):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if self.a <= 0:
            # a <= 0 would make trust fall (or stay flat) as the evidence improves.
            raise ValueError(f"a must be > 0 so calibrated trust stays monotone, got {self.a!r}")

    @classmethod
    def identity(cls, tau: float) -> "PlattCalibration":
        """The uncalibrated estimator, T = sigmoid(raw / tau)."""
        return cls(a=1.0 / tau, b=0.0, source="uncalibrated (a = 1/tau, b = 0)")

    @classmethod
    def from_model_config(cls) -> "PlattCalibration":
        c = load_config("model")["trust_estimator"]["platt_calibration"]
        return cls(a=float(c["a"]), b=float(c["b"]), source=str(c.get("fitted_on", "")))

    def apply(self, raw_score: float) -> float:
        if not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
            raise ValueError(f"raw_score must be a finite number, got {raw_score!r}")
        return _sigmoid(self.a * raw_score + self.b)


def calibrated_trust_from_evidence(
    estimator: TwinTrustEstimator,
    calibration: PlattCalibration,
    prediction_error: float,
    prediction_uncertainty: Any,
    sync_age_penalty: float,
    comm_quality: float,
) -> float:
    """Calibrated trust for explicitly supplied evidence (same inputs and
    validation as `TwinTrustEstimator.compute_trust_from_evidence`)."""
    raw = estimator.compute_raw_score_from_evidence(
        prediction_error, prediction_uncertainty, sync_age_penalty, comm_quality
    )
    return calibration.apply(raw)


def fit_platt(
    raw_scores: Sequence[float], outcomes: Sequence[float], tau: float, iterations: int = 50
) -> PlattCalibration:
    """Fit (a, b) by logistic regression of `outcomes` (each in [0, 1]; may be
    fractional) on `raw_scores`, using Newton's method started at the
    uncalibrated estimator (a = 1/tau, b = 0)."""
    if len(raw_scores) != len(outcomes) or len(raw_scores) < 2:
        raise ValueError("raw_scores and outcomes must have the same length >= 2")
    for y in outcomes:
        if not (0.0 <= y <= 1.0):
            raise ValueError(f"outcomes must be in [0, 1], got {y!r}")
    a, b = 1.0 / tau, 0.0
    for _ in range(iterations):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(raw_scores, outcomes):
            p = _sigmoid(a * x + b)
            w = p * (1.0 - p)
            ga += (p - y) * x
            gb += p - y
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-10:
            break
    return PlattCalibration(a=a, b=b, source=f"fit_platt on {len(raw_scores)} samples")
