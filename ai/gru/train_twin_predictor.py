"""Train the GRU predictor (ai/gru/model.py) as the digital twin's motion predictor.

The twin predicts where a V2X vehicle is now from its last received messages. The baseline
predictor is dead reckoning with constant speed and turn rate (CTRV), as in the live-drive demo.
This script trains `GRUPredictor` (same architecture: input projection -> GRU -> horizon embedding
-> prediction and uncertainty heads) to predict the displacement from the latest message at
each documented horizon (`VALID_HORIZONS`, steps of 0.1 s), and compares it with CTRV on held-out
runs.

Data: true tracks recorded from the live-drive simulation (research/pipeline/collect_tracks.py in
Logapriya05/aegis-v2x-live-drive). Received messages are simulated from them as the demo sends
them: every 0.1 s, each one lost with probability `--loss`, positions with Gaussian noise of
0.4 m (the demo's V2V position noise), speed and heading exact.

Features per received message, in the frame of the latest message (its noisy position and its
heading): forward offset / 10 m, lateral offset / 10 m, speed / 15 m/s, sin(heading change),
age / 1 s.
Target: true position at the horizon minus the latest noisy position, same frame, / 10 m.

The model size is the "twin_predictor" profile (configs/model.yaml -> perception.gru_twin):
hidden 32, 1 layer. The 128 x 2 perception profile is too slow for 10 Hz prediction of every
road user in the browser demo.

usage: python -m ai.gru.train_twin_predictor TRACKS.jsonl OUT_WEIGHTS.json [--test-run 5]
       (also writes OUT_WEIGHTS.report.json with the offline comparison against dead reckoning)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ai.gru.model import VALID_HORIZONS, GRUPredictor
from ai.utils.config_loader import load_config

K = 8  # received messages per input sequence
NOISE_M = 0.4
POS_SCALE, SPD_SCALE = 10.0, 15.0
MAX_H = max(VALID_HORIZONS)


def load_segments(path: Path) -> dict[int, list[list[tuple[float, ...]]]]:
    """Tracks grouped by run; a vehicle's track is split where it jumps (respawn) or has a gap."""
    raw: dict[tuple[int, str], list[tuple[float, ...]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        run, t, vid, x, z, vx, vz, h = json.loads(line)
        raw[(run, vid)].append((t, x, z, vx, vz, h))
    out: dict[int, list[list[tuple[float, ...]]]] = defaultdict(list)
    for (run, _), rows in raw.items():
        rows.sort()
        seg = [rows[0]]
        for a, b in zip(rows, rows[1:]):
            if b[0] - a[0] > 0.15 or math.hypot(b[1] - a[1], b[2] - a[2]) > 5.0:
                out[run].append(seg)
                seg = []
            seg.append(b)
        out[run].append(seg)
    return out


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def frame(h0: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Forward and right unit vectors for heading h0 (demo: forward = (-sin h, -cos h))."""
    return (-math.sin(h0), -math.cos(h0)), (math.cos(h0), -math.sin(h0))


def make_samples(segs, rng: random.Random, loss: float, per_seg_stride: int = 2):
    xs, ys, ctrv = [], [], []
    for seg in segs:
        n = len(seg)
        for i in range(K + 4, n - MAX_H, per_seg_stride):
            hist = [i]
            j = i - 1
            while len(hist) < K and j >= 0 and i - j <= 25:
                if rng.random() >= loss:
                    hist.append(j)
                j -= 1
            if len(hist) < K:
                continue
            hist.reverse()  # oldest first, latest last
            noisy = {
                j: (seg[j][1] + rng.gauss(0, NOISE_M), seg[j][2] + rng.gauss(0, NOISE_M))
                for j in hist
            }
            t0, h0 = seg[i][0], seg[i][5]
            (fx, fz), (rx, rz) = frame(h0)
            x0, z0 = noisy[i]
            feats = []
            for j in hist:
                dx, dz = noisy[j][0] - x0, noisy[j][1] - z0
                sp = math.hypot(seg[j][3], seg[j][4])
                feats.append(
                    [
                        (dx * fx + dz * fz) / POS_SCALE,
                        (dx * rx + dz * rz) / POS_SCALE,
                        sp / SPD_SCALE,
                        math.sin(wrap(seg[j][5] - h0)),
                        t0 - seg[j][0],
                    ]
                )
            tgt, base = [], []
            jp = hist[-2]
            sp0 = math.hypot(seg[i][3], seg[i][4])
            w = wrap(h0 - seg[jp][5]) / max(1e-3, t0 - seg[jp][0])
            for hz in VALID_HORIZONS:
                dx, dz = seg[i + hz][1] - x0, seg[i + hz][2] - z0
                tgt.append([(dx * fx + dz * fz) / POS_SCALE, (dx * rx + dz * rz) / POS_SCALE])
                a = hz * 0.1  # CTRV from the latest message (as twinPos in the demo)
                if sp0 > 0.5:
                    hm = h0 + w * a / 2
                    px, pz = x0 - math.sin(hm) * sp0 * a, z0 - math.cos(hm) * sp0 * a
                else:
                    px, pz = x0 + seg[i][3] * a, z0 + seg[i][4] * a
                base.append(math.hypot(px - seg[i + hz][1], pz - seg[i + hz][2]))
            xs.append(feats)
            ys.append(tgt)
            ctrv.append(base)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ctrv)


def build_model(seed: int = 0) -> GRUPredictor:
    cfg = load_config("model")
    prof = cfg["perception"]["gru_twin"]
    config: dict[str, Any] = {"perception": {"gru": prof}}
    return GRUPredictor(config=config, input_dim=5, output_dim=2 * len(VALID_HORIZONS), seed=seed)


def forward_all(m: GRUPredictor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions and uncertainties for every horizon: (batch, n_horizons, 2) each.

    All horizons come from one pass (output_dim = 2 x len(VALID_HORIZONS)), conditioned on the
    embedding of the longest horizon. GRUPredictor adds the horizon embedding to the hidden state
    before a linear head, so on its own the embedding can only shift the output by a constant per
    horizon; it cannot scale a displacement with the horizon (tried first: errors at H = 1 and 10
    were 2-3 times the dead-reckoning error).
    """
    _, h_n = m.gru(m.input_projection(x))
    last = h_n[-1] + m.horizon_embedding.weight[len(VALID_HORIZONS) - 1]
    shape = (x.shape[0], len(VALID_HORIZONS), 2)
    return m.prediction_head(last).view(shape), m.uncertainty_head(last).view(shape)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--test-run", type=int, default=5)
    ap.add_argument("--loss", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    runs = load_segments(a.tracks)
    train = [s for r, ss in runs.items() if r != a.test_run for s in ss]
    test = runs[a.test_run]
    xtr, ytr, _ = make_samples(train, rng, a.loss)
    xte, yte, cte = make_samples(test, random.Random(a.seed + 1), a.loss)
    n_val = len(xtr) // 10
    perm = torch.randperm(len(xtr))
    xva, yva = xtr[perm[:n_val]], ytr[perm[:n_val]]
    xtr, ytr = xtr[perm[n_val:]], ytr[perm[n_val:]]

    m = build_model(a.seed)
    mods = [m.input_projection, m.gru, m.horizon_embedding, m.prediction_head, m.uncertainty_head]
    params = [p for mod in mods for p in mod.parameters()]
    for mod in mods:
        mod.train()
    opt = torch.optim.Adam(params, lr=3e-3)
    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        idx = torch.randperm(len(xtr))
        for k in range(0, len(idx), 256):
            b = idx[k : k + 256]
            pred, unc = forward_all(m, xtr[b])
            mse = ((pred - ytr[b]) ** 2).mean()
            sd = unc + 1e-3  # uncertainty head as a per-dimension standard deviation
            nll = (torch.log(sd) + (pred.detach() - ytr[b]) ** 2 / (2 * sd**2)).mean()
            opt.zero_grad()
            (mse + 0.01 * nll).backward()
            opt.step()
        with torch.no_grad():
            v = ((forward_all(m, xva)[0] - yva) ** 2).mean().item()
        if v < best:
            best, best_state = v, [
                {k: t.clone() for k, t in mod.state_dict().items()} for mod in mods
            ]
        print(f"epoch {ep:2d} val mse {v:.5f}", flush=True)
    for mod, st in zip(mods, best_state or []):
        mod.load_state_dict(st)
        mod.eval()

    with torch.no_grad():
        pred, _ = forward_all(m, xte)
    err = torch.linalg.norm((pred - yte) * POS_SCALE, dim=-1)  # metres, (n, horizons)
    report: dict[str, Any] = {
        "test_run": a.test_run,
        "test_samples": len(xte),
        "train_samples": len(xtr),
        "message_loss": a.loss,
        "horizons": {},
    }
    for k, hz in enumerate(VALID_HORIZONS):
        g, c = err[:, k], cte[:, k]
        report["horizons"][hz] = {
            "gru_mean_err_m": g.mean().item(),
            "ctrv_mean_err_m": c.mean().item(),
            "gru_within_1m": (g < 1).float().mean().item(),
            "ctrv_within_1m": (c < 1).float().mean().item(),
        }
        print(
            f"H={hz:2d}  GRU {g.mean():.3f} m ({(g < 1).float().mean():.3f} <1 m)   "
            f"CTRV {c.mean():.3f} m ({(c < 1).float().mean():.3f} <1 m)"
        )

    def t(x: torch.Tensor) -> list:
        return [round(v, 6) for v in x.flatten().tolist()]

    gru = m.gru
    weights = {
        "about": "GRUPredictor (ai/gru/model.py) trained as the twin motion predictor "
        "(ai/gru/train_twin_predictor.py)",
        "K": K,
        "pos_scale": POS_SCALE,
        "spd_scale": SPD_SCALE,
        "horizons": list(VALID_HORIZONS),
        "hidden": gru.hidden_size,
        "input_dim": 5,
        "W_in": t(m.input_projection.weight),
        "b_in": t(m.input_projection.bias),
        "W_ih": t(gru.weight_ih_l0),
        "W_hh": t(gru.weight_hh_l0),
        "b_ih": t(gru.bias_ih_l0),
        "b_hh": t(gru.bias_hh_l0),
        "e_h": t(m.horizon_embedding.weight[len(VALID_HORIZONS) - 1]),
        "W_out": t(m.prediction_head.linear.weight),
        "b_out": t(m.prediction_head.linear.bias),
    }
    # compact weights (the same JSON is embedded in the live demo as GRU_W); the offline
    # evaluation goes to a separate report file next to it
    a.out.write_text(json.dumps(weights, separators=(",", ":")) + "\n", encoding="utf-8")
    a.out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
