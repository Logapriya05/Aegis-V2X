"""Smoke test: the decision-cycle benchmark runs end to end and writes its result."""

import json

from evaluation.benchmark_twintrust_ap import main


def test_benchmark_runs_and_writes_json(tmp_path):
    out = tmp_path / "timing.json"
    assert main(["--cycles", "200", "--warmup", "10", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    lat = res["latency_us"]
    assert res["cycles"] == 200
    assert 0 < lat["median"] <= lat["p99"] <= lat["max"]
