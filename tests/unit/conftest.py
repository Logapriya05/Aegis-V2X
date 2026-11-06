"""Collection rules for tests/unit.

The Phase 4 neural-network modules (PointPillars, V2X-ViT, GRU) and the
tests that exercise them need PyTorch, which lives in
requirements/ai.txt. CI (.github/workflows/ci.yml) installs only
requirements.txt (base + dev), so without this file those five test
modules fail at import time and turn the whole CI run red.

When torch is installed they are collected and run as usual. When it is
not, they are left out of collection and the reason is printed in the
pytest header, so the skip is visible rather than silent.
"""

import importlib.util

TORCH_TEST_MODULES = [
    "test_gru.py",
    "test_gru_twin_predictor.py",
    "test_phase4_pipeline.py",
    "test_pointpillars.py",
    "test_trust_estimator.py",
    "test_v2x_vit.py",
]

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

collect_ignore = [] if TORCH_AVAILABLE else list(TORCH_TEST_MODULES)


def pytest_report_header(config):
    if TORCH_AVAILABLE:
        return None
    return (
        "torch is not installed: skipping tests/unit/"
        + ", ".join(TORCH_TEST_MODULES)
        + " (install requirements/ai.txt to run them)"
    )
