"""
Evidence normalisation for TwinTrust-AP.

`TwinTrustEstimator.compute_trust_from_evidence` and
`CriticalityEstimator.compute_criticality_from_evidence` take evidence terms
that are already normalised to [0, 1], and their docstrings leave the
normalisation to the caller (see the "TRUST INPUT-CONTRACT GAP" in
`ai/trust_estimator/model.py`). This module writes that normalisation down,
so the estimators, the paper and the live-drive demo use the same
definitions.

Each raw measurement is divided by a reference scale and clipped to [0, 1]:

    Trust evidence (per cycle, over the twin objects the ego can verify with its LiDAR)
      e  prediction error        = clip(mean |twin - LiDAR| position error / error_ref_m)
      u  prediction uncertainty  = clip(robust_spread(recent mean errors) / error_ref_m)
      a  sync age penalty        = clip(mean information age / age_ref_s)
      q  comm quality            = clip(packet delivery ratio)
                                   * clip(1 - mean latency / latency_ref_ms)

    Criticality features
      relative speed             = clip(max closing speed of road users within 60 m
                                        / relative_speed_ref_mps)
      blockage probability       = degraded V2V links / V2V links  (0 when there are none);
                                   degraded = line of sight blocked OR mmWave beam misaligned
                                   (configs/model.yaml -> twintrust_ap.beam)
      sync age                   = a
      channel degradation        = 1 - q
      traffic density            = clip(road users within 50 m / density_ref)

    Twin correctness (the calibration target and the R term of J)
      fraction of verifiable twin objects within correct_within_m of the LiDAR position

The reference scales live in `configs/model.yaml` -> `twintrust_ap.evidence`.
They are documented design choices (the values used by the live-drive demo,
Logapriya05/aegis-v2x-live-drive), not fitted or validated values.
`robust_spread` is the GDCA module's own median/MAD estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ai.twintrust_ap.gdca import robust_spread
from ai.utils.config_loader import load_config


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _finite(name: str, x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    return float(x)


@dataclass(frozen=True)
class EvidenceScales:
    error_ref_m: float = 3.0
    age_ref_s: float = 1.0
    latency_ref_ms: float = 300.0
    relative_speed_ref_mps: float = 20.0
    density_ref: float = 20.0
    correct_within_m: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if _finite(name, getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")

    @classmethod
    def from_model_config(cls) -> "EvidenceScales":
        e = load_config("model")["twintrust_ap"]["evidence"]
        return cls(**{k: float(v) for k, v in e.items()})


@dataclass(frozen=True)
class TrustEvidence:
    prediction_error: float
    prediction_uncertainty: float
    sync_age_penalty: float
    comm_quality: float


@dataclass(frozen=True)
class CriticalityFeatures:
    relative_speed: float
    blockage_probability: float
    sync_age: float
    channel_degradation: float
    traffic_density: float


def trust_evidence(
    mean_twin_error_m: float,
    recent_mean_errors_m: Sequence[float],
    mean_info_age_s: float,
    packet_delivery_ratio: float,
    mean_latency_ms: float,
    scales: EvidenceScales,
) -> TrustEvidence:
    """Normalised (e, u, a, q). `recent_mean_errors_m` is the short history of
    per-cycle mean errors (the demo keeps the last 20 cycles, 2 s) and needs
    at least 2 values."""
    err = _finite("mean_twin_error_m", mean_twin_error_m)
    if err < 0:
        raise ValueError(f"mean_twin_error_m must be >= 0, got {err!r}")
    if len(recent_mean_errors_m) < 2:
        raise ValueError("recent_mean_errors_m needs at least 2 values for a robust spread")
    spread = robust_spread([_finite("recent_mean_errors_m[i]", v) for v in recent_mean_errors_m])
    age = _finite("mean_info_age_s", mean_info_age_s)
    pdr = _finite("packet_delivery_ratio", packet_delivery_ratio)
    lat = _finite("mean_latency_ms", mean_latency_ms)
    q = _clip01(pdr) * _clip01(1.0 - lat / scales.latency_ref_ms)
    return TrustEvidence(
        prediction_error=_clip01(err / scales.error_ref_m),
        prediction_uncertainty=_clip01(spread / scales.error_ref_m),
        sync_age_penalty=_clip01(age / scales.age_ref_s),
        comm_quality=q,
    )


def criticality_features(
    max_closing_speed_mps: float,
    blocked_links: int,
    total_links: int,
    evidence: TrustEvidence,
    road_users_within_50m: int,
    scales: EvidenceScales,
) -> CriticalityFeatures:
    """The five Context-Aware Criticality features, each in [0, 1]."""
    if total_links < 0 or blocked_links < 0 or blocked_links > total_links:
        raise ValueError(
            f"need 0 <= blocked_links <= total_links, got {blocked_links}/{total_links}"
        )
    if road_users_within_50m < 0:
        raise ValueError(f"road_users_within_50m must be >= 0, got {road_users_within_50m}")
    closing = max(0.0, _finite("max_closing_speed_mps", max_closing_speed_mps))
    return CriticalityFeatures(
        relative_speed=_clip01(closing / scales.relative_speed_ref_mps),
        blockage_probability=(blocked_links / total_links) if total_links else 0.0,
        sync_age=evidence.sync_age_penalty,
        channel_degradation=1.0 - evidence.comm_quality,
        traffic_density=_clip01(road_users_within_50m / scales.density_ref),
    )


def twin_correctness(position_errors_m: Sequence[float], scales: EvidenceScales) -> float | None:
    """Fraction of verifiable twin objects within `correct_within_m`; None if
    there is no verifiable object this cycle."""
    if not position_errors_m:
        return None
    errs = [_finite("position_errors_m[i]", v) for v in position_errors_m]
    return sum(1 for v in errs if v < scales.correct_within_m) / len(errs)
