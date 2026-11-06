"""Tests for the live telemetry endpoint (demo -> Prometheus)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_live_telemetry_updates_metrics(client: TestClient) -> None:
    payload = {"trust": 0.82, "criticality": 0.35, "horizon": 0.6, "action": "beam_switch"}
    response = client.post("/api/v1/telemetry/live", json=payload)
    assert response.status_code == 204

    metrics = client.get("/metrics").text
    assert "aegis_trust_score_bucket" in metrics
    assert "aegis_criticality_score_bucket" in metrics
    assert 'aegis_fsdp_decisions_total{action="beam_switch"}' in metrics
    assert "aegis_prediction_horizon_latest 0.6" in metrics


def test_live_telemetry_rejects_out_of_range_values(client: TestClient) -> None:
    payload = {"trust": 1.5, "criticality": 0.35, "horizon": 0.6, "action": "predict"}
    response = client.post("/api/v1/telemetry/live", json=payload)
    assert response.status_code == 422
