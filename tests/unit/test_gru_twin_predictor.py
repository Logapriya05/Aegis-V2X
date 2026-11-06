"""GRU twin motion predictor: sample construction, model profile and exported weights."""

import json
import math
import random

import pytest
import torch

from ai.gru.model import VALID_HORIZONS
from ai.gru.train_twin_predictor import K, build_model, forward_all, make_samples
from ai.utils.config_loader import REPO_ROOT


def straight_track(n=60, speed=10.0):
    # demo convention: heading 0 moves towards -z
    return [(0.1 * i, 0.0, -speed * 0.1 * i, 0.0, -speed, 0.0) for i in range(n)]


def test_samples_have_documented_shapes_and_frame():
    x, y, ctrv = make_samples([straight_track()], random.Random(0), loss=0.0, per_seg_stride=1)
    assert x.shape[1:] == (K, 5) and y.shape[1:] == (len(VALID_HORIZONS), 2)
    assert ctrv.shape[1] == len(VALID_HORIZONS)
    # straight motion at 10 m/s: forward target grows by about 1 m per 0.1 s step, lateral stays ~0
    fwd = y[:, :, 0].mean(0) * 10
    for k, h in enumerate(VALID_HORIZONS):
        assert fwd[k].item() == pytest.approx(h * 1.0, abs=0.15)
    assert abs(y[:, :, 1].mean().item()) < 0.02
    # dead reckoning error is the position noise only (0.4 m per axis -> about 0.5 m)
    assert 0.3 < ctrv.mean().item() < 0.7


def test_model_profile_and_forward_shape():
    m = build_model()
    assert (m.hidden_size, m.num_layers) == (32, 1)
    pred, unc = forward_all(m, torch.zeros(3, K, 5))
    assert pred.shape == unc.shape == (3, len(VALID_HORIZONS), 2)
    assert (unc >= 0).all()


def test_exported_weights_match_profile():
    w = json.loads((REPO_ROOT / "models" / "gru_twin_predictor.json").read_text(encoding="utf-8"))
    h = w["hidden"]
    assert (w["K"], w["input_dim"], w["horizons"]) == (K, 5, list(VALID_HORIZONS))
    assert len(w["W_ih"]) == len(w["W_hh"]) == 3 * h * h
    assert len(w["W_out"]) == 2 * len(VALID_HORIZONS) * h
    assert all(math.isfinite(v) for v in w["W_out"])
