# GRU twin predictor -> ONNX export — environment blocker, not a result

Status: **BLOCKED on this machine**. No export ran, no comparison ran.
`export_onnx.py` and `tests/unit/test_gru_onnx_export.py` are written and
believed correct but have **never been executed** — say so, don't claim
otherwise.

## What's here

- `model.py` — the real `GRUPredictor` architecture (brought in from
  `latest-main`, where it was already merged/reviewed; not present on
  `feature/logapriya-software-engineering` before this task). Not
  modified from that version.
- `trained_weights/twin_gru_weights.json` — the exact trained weights the
  live demo embeds as `GRU_W` in `index.html`, extracted verbatim (not
  retrained, not fabricated) from the live-drive repo. Shapes checked:
  `W_in` 160 = 32×5, `W_ih`/`W_hh` 3072 = 3×32×32 (GRU's 3 gates), `W_out`
  384 = 12×32 (12 = 2 × 6 horizons) — all consistent with the `gru_twin`
  profile (hidden 32, 1 layer, input_dim 5) documented in
  `research/README.md` section 8 of the live-drive repo.
- `export_onnx.py` — loads those weights into a real `GRUPredictor`,
  wraps the exact computation `train_twin_predictor.py`'s `forward_all`
  uses (all 6 horizons from one GRU pass, conditioned on the
  longest-horizon embedding row), and exports it via `torch.onnx.export`
  at opset 17.
- `tests/unit/test_gru_onnx_export.py` — runs that same model through
  ONNX Runtime on a fixed deterministic input and asserts the max
  absolute difference against PyTorch is under a stated tolerance (see
  below). Uses `pytest.importorskip` so it collects and skips cleanly,
  rather than erroring, on an environment without the stack installed.

Also changed: `configs/model.yaml` gained the `perception.gru_twin`
profile (hidden 32, num_layers 1, the twin-predictor's 5 input features)
— copied verbatim from `latest-main`, where it already exists; it was
missing here and `GRUPredictor` needs it to instantiate. Nothing else in
that file changed.

## Blocker (reproducible)

`requirements/ai.txt` pins `torch==2.5.1`, `onnx==1.17.0`,
`onnxruntime==1.20.0` (plus `numpy==1.26.4` via `requirements/base.txt`).
This machine runs Python 3.13.15. None of these four pins have a
prebuilt wheel for Python 3.13:

```
$ .venv/Scripts/python.exe -m pip download --no-deps torch==2.5.1
ERROR: Could not find a version that satisfies the requirement torch==2.5.1
  (from versions: 2.6.0, 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0,
   2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1, 2.15.0)
ERROR: No matching distribution found for torch==2.5.1
```

i.e. the earliest `torch` release with any Python-3.13-compatible
distribution is 2.6.0 — 2.5.1 has none, for any platform.

`numpy==1.26.4` and `onnx==1.17.0` do have source distributions, so pip
falls back to building them from source — and that fails too, because
this machine has no C/C++ compiler:

```
$ .venv/Scripts/python.exe -m pip install --dry-run -r requirements/ai.txt
...
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
Running `cl /?` gave "[WinError 2] The system cannot find the file specified"
Running `gcc --version` gave "[WinError 2] The system cannot find the file specified"
...
error: metadata-generation-failed
```

**Root cause:** `requirements/ai.txt`'s pins predate Python 3.13 wheel
support across the board, and this machine has no C/C++ toolchain to
build the ones that ship a source distribution.

## How this could be unblocked (not done here — out of this task's scope)

Either, without touching `requirements/ai.txt` (a shared, deliberately
pinned file, "owner: Vaishnavi" per its header — not mine to re-pin):
1. Run this on a machine/environment with Python 3.11 or 3.12 (what
   these exact pins were written for), or
2. Install a C/C++ compiler (e.g. Visual Studio Build Tools) here so
   `numpy`/`onnx` can build from source, and separately accept that
   `torch==2.5.1` specifically still has no distribution at all for
   Python 3.13 regardless of a compiler — that one needs either a
   different Python or a newer, unpinned torch version, which is a
   decision for whoever owns `requirements/ai.txt`, not something to
   change unprompted.

Per the task's own instruction ("document the exact blocker rather than
changing unrelated dependencies"), neither was attempted.

## What was verified before hitting the blocker

- The real trained architecture and weights were located and inspected
  (`ai/gru/model.py`, `ai/gru/train_twin_predictor.py` on `latest-main`;
  `GRU_W` in the live-drive repo's `index.html`) — not fabricated.
- The weight shapes extracted from `GRU_W` match the `gru_twin` profile
  exactly (see above).
- `export_onnx.py` and the test both pass a plain syntax check
  (`python -m py_compile`) but have not been run, since `import torch`
  fails immediately in the isolated venv (`.venv/` at the repo root, not
  committed, matching `.gitignore`'s existing `.venv/` entry).

## Claims

None. No ONNX model, no max-abs-diff number, no pass/fail result exists.
Treat `export_onnx.py` and the test as the ready-to-run next step once
the environment above is sorted out, not as verified-working code.
