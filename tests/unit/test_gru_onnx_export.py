"""ONNX export correctness for the twin-motion GRUPredictor.

UNTESTED on this machine (no working torch/onnx/onnxruntime install here
-- see ai/gru/README_onnx_export.md). Written and believed correct
against the documented APIs, ready to run once that environment blocker
is resolved. `pytest.importorskip` means this file collects (and skips
cleanly) rather than erroring out when the ai.txt stack isn't installed,
matching the other phase-specific tests in this suite.

Tolerance: 5e-6 m, the exact bar `research/tests/check_embedded.py`
already uses in the live-drive repo to call the JavaScript port and the
PyTorch model "the same" -- applying the project's own existing agreement
threshold to ONNX-vs-PyTorch rather than inventing a new one. The model's
raw output is in /10 m units (POS_SCALE, see ai/gru/export_onnx.py), so
the raw-tensor tolerance is 5e-6 / 10 = 5e-7.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")

from ai.gru.export_onnx import TwinGRUForwardAll, load_trained_model  # noqa: E402

POS_SCALE = 10.0
TOLERANCE_RAW = 5e-6 / POS_SCALE  # 5e-6 m, this project's existing port-parity bar


@pytest.fixture(scope="module")
def deterministic_input() -> "torch.Tensor":
    """One fixed, reproducible (batch=3, seq_len=8, input_dim=5) sequence.

    torch.manual_seed pins the RNG so this is identical on every run (not
    hand-fabricated, but exactly reproducible) -- a fixed seed is the
    deterministic representative input the task asks for.
    """
    g = torch.Generator().manual_seed(1234)
    return torch.randn(3, 8, 5, generator=g, dtype=torch.float32)


def test_onnx_matches_pytorch(tmp_path, deterministic_input):
    pytorch_model = TwinGRUForwardAll(load_trained_model()).eval()
    onnx_path = tmp_path / "twin_gru.onnx"

    torch.onnx.export(
        pytorch_model,
        deterministic_input[:1],
        str(onnx_path),
        input_names=["historical_sequence"],
        output_names=["displacement_by_horizon"],
        dynamic_axes={"historical_sequence": {0: "batch"}, "displacement_by_horizon": {0: "batch"}},
        opset_version=17,
    )

    with torch.no_grad():
        torch_out = pytorch_model(deterministic_input).numpy()

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {"historical_sequence": deterministic_input.numpy()})

    max_abs_diff = float(abs(torch_out - onnx_out).max())
    print(f"max abs diff (raw units): {max_abs_diff:.3e}; tolerance: {TOLERANCE_RAW:.3e}")
    assert max_abs_diff < TOLERANCE_RAW, (
        f"ONNX output diverges from PyTorch by {max_abs_diff:.3e} (raw units), "
        f"exceeding the {TOLERANCE_RAW:.3e} tolerance (5e-6 m / POS_SCALE)"
    )
