"""ReactionBackend protocol + StoichiometricBackend (synthetic multiphase chemistry).

A backend receives, per liquid cluster, the newly released mol per kinetic phase
plus the existing dissolved-solution inventory and available water. It never sees
unreacted clinker and knows no coordinates. Fields it cannot compute are NaN with
status "not_available", never 0.

Products travel as Parcels carrying their OWN element vector and skeleton volume
(cm^3) — variable-composition solution phases (GEMS CSHQ) are never reinterpreted
as fixed formulas (PRD §2.3 product-parcel ledger).

Retryable failures (nonconvergence, timeouts) raise BackendTransientError, which
the engine turns into a trial reject; any other exception is a hard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np

from .config import ReactionRule
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, Registry, element_vector)

STATUS_OK = "ok"
STATUS_INSUFFICIENT_WATER = "insufficient_water"
PH_NOT_AVAILABLE = "not_available"


class BackendTransientError(RuntimeError):
    """Retryable backend failure — the engine rejects the trial (dt halving)."""


@dataclass
class Parcel:
    phase_id: str
    mol: float
    elements: np.ndarray      # (E,) over ELEMENT_IDS, total element mol
    skel_vol_cm3: float       # solid skeleton volume of this parcel
    # per-endmember (DC) mols of this parcel (E1, PRD 2.3 rev.3). None means
    # the phase is single-endmember: the engine books {phase_id: mol}. A dict
    # must sum to `mol` (worker-verified endmember-sum) — never invented.
    endmember_mol: Optional[Dict[str, float]] = None


@dataclass
class ReactionResult:
    status: str
    parcels: List[Parcel] = field(default_factory=list)
    water_consumed_mol: float = 0.0
    residual_inventory: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))
    injected_elements: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))  # seeds/solver floors
    ph: float = math.nan
    ph_status: str = PH_NOT_AVAILABLE


class ReactionBackend(Protocol):
    backend_id: str
    hydrate_ids: tuple  # fixed channel order of every parcel this backend emits
    # E1 (PRD 2.3 rev.3): per-channel endmember (DC) names in fixed order, and
    # each endmember's element row — both derived from the backend's own
    # definition source (bundle DCH / registry), never hardcoded.
    hydrate_endmembers: Dict[str, tuple]
    endmember_elements: Dict[str, np.ndarray]
    # "incremental": parcels are NEW precipitates appended to holdings.
    # "snapshot": parcels are the cluster's ENTIRE new assemblage (absolute
    # replacement; water_consumed_mol may be negative on re-dissolution).
    mode: str

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray, solid_elements=None) -> ReactionResult:
        ...


class StoichiometricBackend:
    """Deterministic fixed-stoichiometry backend: every released mol precipitates
    immediately per the config's element-balanced rules; the dissolved-solution
    inventory passes through unchanged (nothing accumulates in solution)."""

    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "incremental"

    def __init__(self, rules: Dict[str, ReactionRule], registry: Registry):
        self._rules = rules
        self._registry = registry
        # every stoichiometric hydrate is single-endmember: the channel IS the
        # endmember, with its registry formula as the element row
        self.hydrate_endmembers = {h: (h,) for h in HYDRATE_PHASE_IDS}
        self.endmember_elements = {
            h: element_vector(registry.get(h).formula, 1.0)
            for h in HYDRATE_PHASE_IDS}

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray, solid_elements=None) -> ReactionResult:
        if solid_elements is not None:
            raise RuntimeError(
                "StoichiometricBackend is incremental-only and cannot "
                "re-equilibrate existing solids (no silent fallback)")
        water_need = 0.0
        totals: Dict[str, float] = {}
        for phase_id, n_mol in released_mol.items():
            if n_mol <= 0.0:
                continue
            rule = self._rules.get(phase_id)
            if rule is None:
                raise RuntimeError(
                    f"stoichiometric backend has no reaction rule for released "
                    f"phase {phase_id!r} (SCM glasses and the E3 soluble salt "
                    f"carriers require the gems3k backend)")
            water_need += n_mol * rule.water_mol
            for hid, coeff in rule.products.items():
                totals[hid] = totals.get(hid, 0.0) + n_mol * coeff
        if water_need > water_available_mol:
            return ReactionResult(status=STATUS_INSUFFICIENT_WATER,
                                  residual_inventory=inventory.copy())
        parcels = []
        for hid in HYDRATE_PHASE_IDS:
            if hid not in totals:
                continue
            entry = self._registry.get(hid)
            mol = totals[hid]
            parcels.append(Parcel(
                phase_id=hid, mol=mol,
                elements=element_vector(entry.formula, mol),
                skel_vol_cm3=mol * entry.skeleton_molar_volume_cm3))
        return ReactionResult(status=STATUS_OK, parcels=parcels,
                              water_consumed_mol=water_need,
                              residual_inventory=inventory.copy())
