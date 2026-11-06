# Aegis-V2X

**A Calibrated Trust-Driven Autonomic Framework for Joint Prediction, Synchronization and Beam Adaptation in Digital Twin-Assisted 6G V2X Networks**

B.Tech Research Project — Department of Computer Science and Engineering, Hindustan Institute of Technology and Science, Chennai. Target venue: **IEEE PerCom 2027**.

## Team

| Member | Role |
|---|---|
| Vaishnavi | Research Lead & AI/ML Architect |
| Logapriya | Software Engineering Lead |
| Khushi | Edge AI & Deployment Lead |
| Haridharani | Simulation & Dataset Lead |

## Project Status

**Current Phase: 1 — Research & Architecture** (owner: Vaishnavi)

See `docs/module_decomposition.md` for the full phase → folder → owner map.

## Core Idea

Every communication decision (prediction triggering, prediction horizon,
Digital Twin synchronization, beam adaptation, communication mode) is
governed by **TwinTrust-AP**: a unified policy driven by a *calibrated*
Digital Twin trust probability `T ∈ [0,1]` and a context-aware
criticality score `C ∈ [0,1]`, rather than fixed thresholds or
independently optimized heuristics.

```
Physical Env (CARLA) → Wireless Env (Sionna RT) → Multimodal Sync
  → Perception (PointPillars → V2X-ViT → GRU) → Digital Twin State
  → Trust Estimator + Criticality Estimator
  → TwinTrust-AP (TAHS + FSDP) → Execution → Feedback (closed loop)
```

Full diagrams: `architecture/system_architecture.mmd` and related files.

## Repository Structure

```
Aegis-V2X/
├── docs/            architecture, module decomposition, interfaces, standards
├── architecture/    Mermaid diagram sources
├── datasets/        synchronized multimodal dataset (Phase 2)
├── simulation/      CARLA + Sionna RT scenarios (Phase 2)
├── ai/              perception, prediction, trust/criticality, TwinTrust-AP (Phase 4-5)
├── digital_twin/    DigitalTwinState + state manager interfaces
├── backend/         FastAPI + PostgreSQL services (Phase 3)
├── deployment/       ONNX/TensorRT/Jetson Orin (Phase 6)
├── evaluation/       baselines, ablations, statistical tests (Phase 7)
├── dashboard/        Grafana dashboards (Phase 3)
├── configs/          YAML configuration (all phases)
├── scripts/          setup, verification, lint/format helpers
├── tests/            unit / integration / api / deployment tests
├── models/           checkpoints, ONNX, TensorRT engines, policy table
├── notebooks/        exploratory analysis
├── results/          experiment outputs
├── papers/            IEEE manuscript, thesis
└── presentations/     conference / defense materials
```

## Getting Started (Phase 1)

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
make test
```

Phase-specific stacks are installed only when that phase begins:

```bash
pip install -r requirements/simulation.txt   # Phase 2 (Haridharani)
pip install -r requirements/backend.txt      # Phase 3 (Logapriya)
pip install -r requirements/ai.txt           # Phase 4-5 (Vaishnavi)
pip install -r requirements/deployment.txt   # Phase 6 (Khushi)
```

## Documentation Index

- `docs/architecture_overview.md` — system design and layer breakdown
- `docs/module_decomposition.md` — phase/owner/folder mapping (authoritative)
- `docs/interfaces.md` — full interface reference
- `docs/coding_standards.md` — style, typing, docstring conventions
- `docs/git_workflow.md` — branching and commit conventions
- `architecture/*.mmd` — system, data flow, control flow, sequence, offline pipeline diagrams

## Mathematical Foundation

Trust, criticality, TAHS, and FSDP are formally defined in the project's
`03_Mathematical_Formulation` document and mirrored in
`configs/model.yaml` and the interfaces under `ai/` and `digital_twin/`.
Key invariants (boundedness, monotonicity, determinism) are enforced via
unit tests introduced in Phase 4-5.
