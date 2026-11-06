"""Age bridging and twin-aided beam tracking for TwinTrust-AP.

Both mechanisms are used by the live-drive demo (Logapriya05/aegis-v2x-live-drive,
`research/README.md` section 10) and apply to TwinTrust-AP only, not to the fixed or
age-of-information baselines.

Age bridging
    The twin predicts each V2X vehicle forward from its last received message by
    min(H * step_s, age). If the prediction horizon H is shorter than the age of the information,
    the twin is left behind by (age - H * step_s). Age bridging keeps prediction on and raises the
    horizon chosen by TAHS (and adjusted by the FSDP action) to the smallest allowed horizon that
    covers the mean information age:

        H_bridged = max(H, min{h in allowed : h * step_s >= mean_age})   (largest allowed h if none)

Twin-aided beam tracking
    Without tracking, a beam keeps the bearing measured by the last sweep, and the misalignment
    grows as vehicles move. With tracking, between sweeps each tracked beam is steered to the
    bearing of the twin's predicted position of that vehicle. Vehicles isolated by the trust
    estimator are not tracked (their beam keeps the last swept bearing). Only beams that already
    exist are steered: acquiring a new vehicle still needs a sweep.

Bearing convention (as in the demo): positions are (x, z) in metres, the ego heading psi is in
radians, and the bearing of a point relative to the ego heading is
atan2(x - x_ego, -(z - z_ego)) - psi.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

STEP_S = 0.1  # one horizon step (s), as in TAHS


def bridge_horizon(
    horizon: int,
    mean_age_s: float | None,
    allowed_horizons: Sequence[int],
    step_s: float = STEP_S,
) -> int:
    """Raise `horizon` to the smallest allowed horizon covering the mean information age.

    `mean_age_s` is None when the twin holds no V2X information; the horizon is then unchanged.
    """
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if not allowed_horizons:
        raise ValueError("allowed_horizons must not be empty")
    if mean_age_s is None:
        return horizon
    if not math.isfinite(mean_age_s) or mean_age_s < 0:
        raise ValueError(f"mean_age_s must be a finite non-negative number, got {mean_age_s!r}")
    allowed = sorted(allowed_horizons)
    need = math.ceil(mean_age_s / step_s - 1e-9)
    floor = next((h for h in allowed if h >= need), allowed[-1])
    return max(horizon, floor)


def bearing_to(ego_xz: tuple[float, float], ego_psi: float, point_xz: tuple[float, float]) -> float:
    """Bearing (rad) of `point_xz` relative to the ego heading."""
    return math.atan2(point_xz[0] - ego_xz[0], -(point_xz[1] - ego_xz[1])) - ego_psi


def angle_difference(a: float, b: float) -> float:
    """Absolute difference of two angles, wrapped to [0, pi]."""
    d = (a - b) % (2 * math.pi)
    return 2 * math.pi - d if d > math.pi else d


def track_beams(
    beams: Mapping[str, float],
    predicted_xz: Mapping[str, tuple[float, float]],
    ego_xz: tuple[float, float],
    ego_psi: float,
    isolated: frozenset[str] | set[str] = frozenset(),
) -> dict[str, float]:
    """Return the beam bearings after one tracking step.

    `beams` maps vehicle id -> current beam bearing (from the last sweep or tracking step);
    `predicted_xz` maps vehicle id -> the twin's predicted position. A beam is steered when the
    twin has a prediction for that vehicle and the vehicle is not isolated; otherwise it is kept.
    """
    out = dict(beams)
    for vid in beams:
        if vid in isolated or vid not in predicted_xz:
            continue
        out[vid] = bearing_to(ego_xz, ego_psi, predicted_xz[vid])
    return out
