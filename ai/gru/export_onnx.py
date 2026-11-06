"""Export the trained twin-motion GRUPredictor (ai/gru/model.py) to ONNX.

UNTESTED on this machine -- see ai/gru/README_onnx_export.md for the exact
dependency/environment blocker (requirements/ai.txt's pins predate Python
3.13 wheel support, and no C/C++ compiler is available to build from
source). This script is believed correct against the documented
sionna.rt-free, pure-PyTorch forward path (`forward_all` in
`ai/gru/train_twin_predictor.py`), but has never actually been run.

Loads the exact trained weights the live demo embeds as `GRU_W`
(extracted verbatim into trained_weights/twin_gru_weights.json -- no
retraining, no new/fabricated weights), builds the real GRUPredictor
architecture, reproduces train_twin_predictor.py's `forward_all` (all 6
horizons from one GRU pass, conditioned on the longest-horizon embedding
row), and exports that computation graph to ONNX (opset 17).

Usage:
    python -m ai.gru.export_onnx [--out ai/gru/trained_weights/twin_gru.onnx]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ai.gru.model import VALID_HORIZONS, GRUPredictor

WEIGHTS_PATH = Path(__file__).parent / "trained_weights" / "twin_gru_weights.json"


def load_trained_model() -> GRUPredictor:
    """Builds GRUPredictor (gru_twin profile) and loads the exact trained weights."""
    w = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    assert w["hidden"] == 32 and w["input_dim"] == 5 and w["horizons"] == list(VALID_HORIZONS), (
        "trained_weights/twin_gru_weights.json doesn't match the expected gru_twin shape "
        f"-- got hidden={w['hidden']} input_dim={w['input_dim']} horizons={w['horizons']}"
    )
    output_dim = 2 * len(VALID_HORIZONS)
    config = {"perception": {"gru": {"hidden_size": 32, "num_layers": 1,
                                      "input_features": w.get("about", [])}}}
    m = GRUPredictor(config=config, input_dim=5, output_dim=output_dim, seed=0)

    def t(key: str, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(w[key], dtype=torch.float32).reshape(shape)

    hidden = 32
    with torch.no_grad():
        m.input_projection.weight.copy_(t("W_in", (hidden, 5)))
        m.input_projection.bias.copy_(t("b_in", (hidden,)))
        m.gru.weight_ih_l0.copy_(t("W_ih", (3 * hidden, hidden)))
        m.gru.weight_hh_l0.copy_(t("W_hh", (3 * hidden, hidden)))
        m.gru.bias_ih_l0.copy_(t("b_ih", (3 * hidden,)))
        m.gru.bias_hh_l0.copy_(t("b_hh", (3 * hidden,)))
        # e_h is the embedding row for the longest horizon only (index 5, horizon=10) --
        # forward_all always conditions on this one row (see module docstring).
        m.horizon_embedding.weight[len(VALID_HORIZONS) - 1].copy_(t("e_h", (hidden,)))
        m.prediction_head.linear.weight.copy_(t("W_out", (output_dim, hidden)))
        m.prediction_head.linear.bias.copy_(t("b_out", (output_dim,)))
    m.eval()
    return m


class TwinGRUForwardAll(nn.Module):
    """Exportable wrapper: reproduces train_twin_predictor.py's `forward_all`.

    Input: (batch, seq_len=8, input_dim=5) float32, oldest-first.
    Output: (batch, 6, 2) float32 -- displacement at horizons
    (1, 2, 3, 5, 8, 10) x 0.1 s, in the latest message's frame, / 10 m
    (POS_SCALE), same convention as GRU_W / gruForward in index.html.
    """

    def __init__(self, m: GRUPredictor):
        super().__init__()
        self.input_projection = m.input_projection
        self.gru = m.gru
        self.last_horizon_embedding = nn.Parameter(
            m.horizon_embedding.weight[len(VALID_HORIZONS) - 1].detach().clone(),
            requires_grad=False,
        )
        self.prediction_head = m.prediction_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(self.input_projection(x))
        last = h_n[-1] + self.last_horizon_embedding
        pred = self.prediction_head(last)
        return pred.view(x.shape[0], len(VALID_HORIZONS), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH.parent / "twin_gru.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    m = load_trained_model()
    wrapper = TwinGRUForwardAll(m).eval()

    dummy = torch.zeros(1, 8, 5, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(a.out),
        input_names=["historical_sequence"],
        output_names=["displacement_by_horizon"],
        dynamic_axes={"historical_sequence": {0: "batch"}, "displacement_by_horizon": {0: "batch"}},
        opset_version=a.opset,
    )
    print(f"Exported ONNX model (opset {a.opset}) to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
