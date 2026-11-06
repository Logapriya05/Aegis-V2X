"""Live telemetry endpoint: demo -> Prometheus (Grafana live dashboard).

Records one TwinTrust-AP cycle into the domain metrics in
`app.core.metrics`. No database write happens here -- see the schema
module docstring for why.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.core.metrics import (
    criticality_score_histogram,
    fsdp_decisions_counter,
    prediction_horizon_gauge,
    trust_score_histogram,
)
from app.core.security import require_api_key
from app.schemas.telemetry import LiveTelemetry

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post(
    "/live",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    dependencies=[Depends(require_api_key)],
)
def record_live_telemetry(payload: LiveTelemetry) -> None:
    """Record one live-demo TwinTrust-AP cycle into the Prometheus metrics."""
    trust_score_histogram.observe(payload.trust)
    criticality_score_histogram.observe(payload.criticality)
    prediction_horizon_gauge.set(payload.horizon)
    fsdp_decisions_counter.labels(action=payload.action).inc()
