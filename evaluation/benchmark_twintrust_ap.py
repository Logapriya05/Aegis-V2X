"""CPU latency and memory of one TwinTrust-AP decision cycle (minimal edge-timing deliverable).

One cycle = evidence normalisation -> calibrated twin trust -> criticality ->
TAHS horizon -> FSDP table lookup, using the code and configuration in this repository
(configs/model.yaml, models/fsdp_policy_table.json). Inputs are random but valid raw
measurements drawn from a seeded generator, so runs are reproducible.

This measures the decision logic only, on the machine it runs on. It is not a Jetson
Orin measurement (Jetson deployment is future work, see claude/minimal_scope_final.md
in the project) and it does not include perception or the digital-twin update.

usage: python -m evaluation.benchmark_twintrust_ap [--cycles N] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from ai.criticality.model import CriticalityEstimator
from ai.trust_estimator.calibration import PlattCalibration, calibrated_trust_from_evidence
from ai.trust_estimator.model import TwinTrustEstimator
from ai.twintrust_ap.evidence import EvidenceScales, criticality_features, trust_evidence
from ai.twintrust_ap.fsdp import TableFSDP, load_policy_table
from ai.twintrust_ap.tahs import TAHS, TAHSParams
from ai.utils.config_loader import REPO_ROOT


def build_cycle() -> Callable[[dict[str, Any]], tuple[float, float, int, str]]:
    """Construct the estimators once (as a running vehicle would) and return one decision cycle."""
    scales = EvidenceScales.from_model_config()
    trust_est = TwinTrustEstimator()
    calibration = PlattCalibration.from_model_config()
    crit_est = CriticalityEstimator()
    tahs = TAHS(TAHSParams.from_model_config(profile="full_range"))
    fsdp = TableFSDP(load_policy_table(REPO_ROOT / "models" / "fsdp_policy_table.json"))

    def cycle(m: dict[str, Any]) -> tuple[float, float, int, str]:
        ev = trust_evidence(m["err"], m["hist"], m["age"], m["pdr"], m["lat"], scales)
        trust = calibrated_trust_from_evidence(
            trust_est,
            calibration,
            ev.prediction_error,
            ev.prediction_uncertainty,
            ev.sync_age_penalty,
            ev.comm_quality,
        )
        cf = criticality_features(m["closing"], m["blocked"], m["links"], ev, m["users"], scales)
        crit = crit_est.compute_criticality_from_evidence(
            cf.relative_speed,
            cf.blockage_probability,
            cf.sync_age,
            cf.channel_degradation,
            cf.traffic_density,
        )
        crit = min(1.0, max(0.0, crit))
        horizon = tahs.select_horizon(trust, crit, m["authority"])
        action = fsdp.decide(trust, crit)
        return trust, crit, horizon, action.value

    return cycle


def random_measurements(rng: random.Random, n: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        links = rng.randint(0, 12)
        out.append(
            dict(
                err=rng.uniform(0.0, 4.0),
                hist=[rng.uniform(0.0, 4.0) for _ in range(20)],
                age=rng.uniform(0.0, 1.5),
                pdr=rng.uniform(0.3, 1.0),
                lat=rng.uniform(5.0, 400.0),
                closing=rng.uniform(-5.0, 25.0),
                links=links,
                blocked=rng.randint(0, links),
                users=rng.randint(0, 30),
                authority=rng.uniform(0.0, 1.0),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cycles", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cycles < 100:
        ap.error("--cycles must be at least 100")

    cycle = build_cycle()
    inputs = random_measurements(random.Random(a.seed), a.cycles + a.warmup)
    for m in inputs[: a.warmup]:
        cycle(m)

    times_us = []
    for m in inputs[a.warmup :]:
        t0 = time.perf_counter_ns()
        cycle(m)
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)

    tracemalloc.start()
    for m in inputs[a.warmup : a.warmup + 1000]:
        cycle(m)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times_us.sort()
    q = statistics.quantiles(times_us, n=100)
    res = {
        "what": "one TwinTrust-AP decision cycle "
        "(evidence -> calibrated trust -> criticality -> TAHS -> FSDP)",
        "cycles": a.cycles,
        "seed": a.seed,
        "latency_us": {
            "mean": statistics.fmean(times_us),
            "median": q[49],
            "p95": q[94],
            "p99": q[98],
            "max": times_us[-1],
        },
        "peak_traced_python_memory_kib_over_1000_cycles": peak / 1024.0,
        "budget_note": "the demo runs the cycle at 10 Hz, i.e. a 100 ms budget",
        "machine": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
    }
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
