"""
Abstract contract for the Finite-State Decision Policy (FSDP).

Mathematical basis (03_Mathematical_Formulation.docx, Section 8):
    Trust, Criticality each discretized into {Low, Medium, High}
    => 3 x 3 = 9 runtime states
    A_t = pi(T, C)   (offline-generated policy table, O(1) lookup)

Determinism invariant (Section 11): for a fixed state pair (T, C),
FSDP always returns the same action.

Implemented in: Phase 5 (ai/twintrust_ap/fsdp.py — this file, extended)
Depends on: an offline-generated policy table (configs/model.yaml ->
            fsdp.policy_table_path), produced by the Phase 5 offline
            Pareto optimization pipeline described in Document 2, Section 5.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from ai.utils.config_loader import REPO_ROOT, load_config


class TrustBin(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CriticalityBin(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommunicationAction(str, Enum):
    SYNCHRONIZE = "synchronize"
    PREDICT = "predict"
    BEAM_SWITCH = "beam_switch"
    REDUCE_HORIZON = "reduce_horizon"
    INCREASE_HORIZON = "increase_horizon"
    MAINTAIN_STATE = "maintain_state"


class BaseFSDP(ABC):
    """Contract every concrete FSDP implementation must satisfy."""

    @abstractmethod
    def discretize(self, trust: float, criticality: float) -> tuple[TrustBin, CriticalityBin]:
        """
        Map continuous (trust, criticality) values to their discrete bins.

        Parameters
        ----------
        trust : float
            Calibrated trust probability, in [0, 1].
        criticality : float
            Criticality score, in [0, 1].

        Returns
        -------
        tuple[TrustBin, CriticalityBin]
            The discretized (trust_bin, criticality_bin) pair.
        """
        raise NotImplementedError

    @abstractmethod
    def lookup_action(
        self, trust_bin: TrustBin, criticality_bin: CriticalityBin
    ) -> CommunicationAction:
        """
        Perform the constant-time policy-table lookup for a discretized
        state pair.

        Parameters
        ----------
        trust_bin : TrustBin
        criticality_bin : CriticalityBin

        Returns
        -------
        CommunicationAction
            The single coordinated action selected by the offline policy.
        """
        raise NotImplementedError

    def decide(self, trust: float, criticality: float) -> CommunicationAction:
        """
        Convenience method combining discretize() + lookup_action().
        Concrete subclasses generally do not need to override this.

        Parameters
        ----------
        trust : float
        criticality : float

        Returns
        -------
        CommunicationAction
        """
        trust_bin, criticality_bin = self.discretize(trust, criticality)
        return self.lookup_action(trust_bin, criticality_bin)


_BIN_ORDER = (TrustBin.LOW, TrustBin.MEDIUM, TrustBin.HIGH)
_CBIN_ORDER = (CriticalityBin.LOW, CriticalityBin.MEDIUM, CriticalityBin.HIGH)


def _check_edges(name: str, edges: Any) -> tuple[float, float]:
    if (
        not isinstance(edges, (list, tuple))
        or len(edges) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in edges)
        or not (0.0 < edges[0] < edges[1] < 1.0)
    ):
        raise ValueError(f"{name} must be two increasing numbers inside (0, 1), got {edges!r}")
    return float(edges[0]), float(edges[1])


def load_policy_table(
    path: str | Path,
) -> dict[tuple[TrustBin, CriticalityBin], CommunicationAction]:
    """Read and validate an offline-generated policy table.

    File format (JSON): {"policy": {"<trust bin>": {"<criticality bin>": "<action>"}}, ...}
    with all 9 states present and every action one of `CommunicationAction`.
    Other top-level keys (provenance, statistics) are allowed and ignored here.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    policy = data.get("policy") if isinstance(data, dict) else None
    if not isinstance(policy, dict):
        raise ValueError(f"{path}: missing top-level 'policy' object")
    table: dict[tuple[TrustBin, CriticalityBin], CommunicationAction] = {}
    for tb in _BIN_ORDER:
        row = policy.get(tb.value)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: missing trust bin {tb.value!r}")
        for cb in _CBIN_ORDER:
            if cb.value not in row:
                raise ValueError(f"{path}: missing state {tb.value}/{cb.value}")
            try:
                table[(tb, cb)] = CommunicationAction(row[cb.value])
            except ValueError:
                raise ValueError(
                    f"{path}: {tb.value}/{cb.value} has unknown action {row[cb.value]!r}"
                ) from None
    return table


class TableFSDP(BaseFSDP):
    """Concrete FSDP: discretise (T, C) with fixed bin edges, then look the
    action up in an offline-generated 3 x 3 table (O(1), deterministic).

    Bins: value < edges[0] -> LOW, edges[0] <= value < edges[1] -> MEDIUM,
    otherwise HIGH. Defaults come from `configs/model.yaml` (`fsdp.trust_bin_edges`,
    `fsdp.criticality_bin_edges`, `fsdp.policy_table_path`).

    The table shipped in `models/fsdp_policy_table.json` was generated from the
    live-drive simulation, not from CARLA; see that file's "provenance" entry.
    """

    def __init__(
        self,
        table: dict[tuple[TrustBin, CriticalityBin], CommunicationAction] | None = None,
        trust_edges: tuple[float, float] | None = None,
        criticality_edges: tuple[float, float] | None = None,
    ):
        cfg = (
            load_config("model")["fsdp"] if None in (table, trust_edges, criticality_edges) else {}
        )
        self.trust_edges = _check_edges("trust_edges", trust_edges or cfg["trust_bin_edges"])
        self.criticality_edges = _check_edges(
            "criticality_edges", criticality_edges or cfg["criticality_bin_edges"]
        )
        if table is None:
            table = load_policy_table(REPO_ROOT / cfg["policy_table_path"])
        missing = [(t, c) for t in _BIN_ORDER for c in _CBIN_ORDER if (t, c) not in table]
        if missing:
            raise ValueError(f"policy table is missing states {missing}")
        self.table = dict(table)

    @staticmethod
    def _bin(value: float, edges: tuple[float, float], name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (0.0 <= value <= 1.0)
        ):
            raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
        return 0 if value < edges[0] else 1 if value < edges[1] else 2

    def discretize(self, trust: float, criticality: float) -> tuple[TrustBin, CriticalityBin]:
        return (
            _BIN_ORDER[self._bin(trust, self.trust_edges, "trust")],
            _CBIN_ORDER[self._bin(criticality, self.criticality_edges, "criticality")],
        )

    def lookup_action(
        self, trust_bin: TrustBin, criticality_bin: CriticalityBin
    ) -> CommunicationAction:
        return self.table[(TrustBin(trust_bin), CriticalityBin(criticality_bin))]
