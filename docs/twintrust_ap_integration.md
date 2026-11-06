# TwinTrust-AP integration (2026-11-04)

This note records what was added to the Phase 4/5 TwinTrust-AP code so that it runs end to end. It also says where each number came from.

The live-drive demo (repository `Logapriya05/aegis-v2x-live-drive`, live at https://logapriya05.github.io/aegis-v2x-live-drive/) runs the same algorithms at 10 Hz. It uses an exact JavaScript port that is checked against this repository's Python code on 12,000 cases.

## What was added

| File | What | Changes existing behaviour? |
|---|---|---|
| `ai/trust_estimator/model.py` | `compute_raw_score_from_evidence` (the score before the sigmoid). `compute_trust_from_evidence` now calls it. | No. Outputs are byte-identical on 2,000 test cases. |
| `ai/trust_estimator/calibration.py` | `PlattCalibration`, `calibrated_trust_from_evidence`, `fit_platt`: T_cal = σ(a·raw + b). | No. It is a new, separate path, and the uncalibrated estimator is a = 1/τ, b = 0. |
| `ai/twintrust_ap/evidence.py` | Evidence normalisation: raw measurements become e, u, a, q and the five criticality features. | No. It writes down the "caller supplies the evidence" step that `model.py` leaves open. |
| `ai/twintrust_ap/tahs.py` | `TAHSParams.from_model_config(profile=...)`. | No. The default still reads β = γ = δ = 1. |
| `ai/twintrust_ap/fsdp.py` | `load_policy_table`, `TableFSDP`: the concrete FSDP, a 3 × 3 table lookup. | No. `BaseFSDP` is unchanged. |
| `models/fsdp_policy_table.json` | The generated policy table, with provenance. | New file. |
| `configs/model.yaml` | New keys only: `trust_estimator.platt_calibration`, `tahs.profiles.full_range`, `fsdp.trust_bin_edges` / `criticality_bin_edges`, `objective_function.constraints`, `twintrust_ap.evidence`, `twintrust_ap.beam`. | No existing value changed. |
| `tests/unit/test_twintrust_ap_integration.py` | 22 tests, including reference values from the demo's JavaScript port. | New. |

## Where the numbers come from

All fitted or selected values below come from the **live-drive browser simulation**, not from CARLA, Sionna RT or real vehicles. Re-fit them before reporting results on any other data.

- **Calibration** (a = 3.303, b = 0.660):
  - Target: a twin object is within 1 m of the LiDAR position on the next cycle.
  - Data: 8,000 training and 3,992 held-out cycles.
  - Held-out ECE: 0.129 → 0.042.
- **TAHS `full_range` profile** (β = 3, γ = 5, δ = 2): a design choice.
  - With β = γ = δ = 1 the sigmoid argument stays in [−1, 2], so only horizons 3–8 are reachable.
  - The profile reaches 1–10 and keeps (0.5, 0.5, 0.5) at H = 5.
- **Evidence scales** (3 m error, 1 s age, 300 ms latency, 20 m/s closing speed, 20 road users, 1 m "correct"): design choices.
- **FSDP table** (`models/fsdp_policy_table.json`):
  - The objective is the lowest J among tables whose twin correctness is at least the fixed baseline's (horizon 5, sync and beam sweep every 1 s).
  - Minimising J alone had cut synchronisation and let twin correctness fall from 57% to 46%.
  - Candidate tables were run in closed loop with the mmWave beam model (`twintrust_ap.beam`: 12° beams, 20 ms sweep) on separate selection seeds. **None met the constraint.**
  - Chosen: b1, "synchronize when trust is low, beam switch when it is medium, increase the horizon when it is high" (lowest J, highest accuracy of the beam-aware tables).
  - Evaluation seeds:
    - J 0.090 vs 0.128, 30% lower, with the 95% CI of the difference −0.057 to −0.018;
    - twin correct 51.2% vs 54.7%;
    - synchronisation 48.7 vs 45.8 per second, moved to degraded links rather than reduced.

## Known limitations found

- **Short horizons under delay.** When information is stale and trust is low, TAHS picks short horizons. The twin then cannot bridge the delay, and a fixed 0.5 s horizon does better.
- **Trust that stays high.** Calibrated trust can stay high while the twin degrades (seen under spoofing and in slow dense traffic on some seeds), so the table does not react there.
- **Noisy per-condition results.** Per-condition differences of a few points changed between seed sets at 90–150 s per condition. Longer and repeated runs are needed before per-condition claims.

## CI

This branch now passes the same checks as `main`: `make lint`, `make typecheck`, `scripts/verify_env.py`, `make test`.

- The five PyTorch test modules are skipped when torch is not installed; CI installs `requirements.txt` only.
- `tests/unit/conftest.py` prints that in the pytest header.
- The modules run normally when `requirements/ai.txt` is installed.
