"""Schema for live telemetry pushed from the browser demo.

Unlike `frames`/`trust`/`criticality`, this has no scene/vehicle/frame to
attach to -- the live demo (aegis-v2x-live-drive) is a standalone
three.js page with no backend data model of its own. This endpoint exists
only to feed the Prometheus metrics in `app.core.metrics` so the
provisioned Grafana dashboard reflects a real driving session instead of
only the synthetic data generator.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiveTelemetry(BaseModel):
    """One TwinTrust-AP cycle's output, as read from `window.__aegis.ttap`."""

    trust: float = Field(ge=0.0, le=1.0, description="Calibrated trust T_t.")
    criticality: float = Field(ge=0.0, le=1.0, description="Criticality C_t.")
    horizon: float = Field(gt=0.0, description="Selected prediction horizon H_t, seconds.")
    action: str = Field(min_length=1, max_length=32, description="FSDP action taken this cycle.")
