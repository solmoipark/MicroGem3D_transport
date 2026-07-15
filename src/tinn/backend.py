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
from typing import Dict, List, Protocol

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

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray) -> ReactionResult:
        ...


class StoichiometricBackend:
    """Deterministic fixed-stoichiometry backend: every released mol precipitates
    immediately per the config's element-balanced rules; the dissolved-solution
    inventory passes through unchanged (nothing accumulates in solution)."""

    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS

    def __init__(self, rules: Dict[str, ReactionRule], registry: Registry):
        self._rules = rules
        self._registry = registry

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray) -> ReactionResult:
        water_need = 0.0
        totals: Dict[str, float] = {}
        for phase_id, n_mol in released_mol.items():
            if n_mol <= 0.0:
                continue
            rule = self._rules.get(phase_id)
            if rule is None:
                raise RuntimeError(
                    f"stoichiometric backend has no reaction rule for released "
                    f"phase {phase_id!r} (SCM glasses require the gems3k backend)")
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
