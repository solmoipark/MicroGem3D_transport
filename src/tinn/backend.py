"""ReactionBackend protocol + StoichiometricBackend (synthetic multiphase chemistry).

A backend receives, per liquid cluster, the newly released mol per kinetic phase
plus the existing dissolved-solution inventory and available water. It never sees
unreacted clinker and knows no coordinates. Fields it cannot compute are NaN with
status "not_available", never 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

import numpy as np

from .config import ReactionRule
from .registry import ELEMENT_IDS, HYDRATE_PHASE_IDS, Registry

STATUS_OK = "ok"
STATUS_INSUFFICIENT_WATER = "insufficient_water"
PH_NOT_AVAILABLE = "not_available"


@dataclass
class ReactionResult:
    status: str
    parcels: List[Tuple[str, float]] = field(default_factory=list)  # (hydrate_id, mol)
    water_consumed_mol: float = 0.0
    residual_inventory: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))
    ph: float = math.nan
    ph_status: str = PH_NOT_AVAILABLE


class ReactionBackend(Protocol):
    backend_id: str

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray) -> ReactionResult:
        ...


class StoichiometricBackend:
    """Deterministic fixed-stoichiometry backend: every released mol precipitates
    immediately per the config's element-balanced rules; the dissolved-solution
    inventory passes through unchanged (nothing accumulates in solution)."""

    backend_id = "stoichiometric"

    def __init__(self, rules: Dict[str, ReactionRule], registry: Registry):
        self._rules = rules
        self._registry = registry

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray) -> ReactionResult:
        water_need = 0.0
        parcels: Dict[str, float] = {}
        for phase_id, n_mol in released_mol.items():
            if n_mol <= 0.0:
                continue
            rule = self._rules[phase_id]
            water_need += n_mol * rule.water_mol
            for hid, coeff in rule.products.items():
                parcels[hid] = parcels.get(hid, 0.0) + n_mol * coeff
        if water_need > water_available_mol:
            return ReactionResult(status=STATUS_INSUFFICIENT_WATER,
                                  residual_inventory=inventory.copy())
        ordered = [(h, parcels[h]) for h in HYDRATE_PHASE_IDS if h in parcels]
        return ReactionResult(status=STATUS_OK, parcels=ordered,
                              water_consumed_mol=water_need,
                              residual_inventory=inventory.copy())
