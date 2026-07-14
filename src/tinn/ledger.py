"""Component/phase/water/volume ledger invariants (PRD §6.1 — the only blocking checks).

All checks compare the authoritative mol ledger against itself (element closure)
and against the dense volume arrays (spatial consistency). Violations are returned
as stable identifiers so the engine can reject a trial with a nameable reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, INERT_PHASE_ID,
                       KINETIC_PHASE_IDS, SOLID_PHASE_IDS, Registry)
from .state import SimulationState, formula_elements

ELEMENT_ATOL_MOL = 1e-24
ELEMENT_RTOL = 1e-8
VOXEL_IDENTITY_TOL = 1e-12
DENSE_LEDGER_RTOL = 1e-9
PLACEMENT_RTOL = 1e-9


@dataclass
class PlacementBalance:
    """Backend volume == requested == placed (PRD §6.1 배치 수지)."""
    backend_bulk_vol_vox: float
    requested_bulk_vol_vox: float
    placed_bulk_vol_vox: float


@dataclass
class LedgerReport:
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def current_elements(state: SimulationState, registry: Registry) -> np.ndarray:
    """Element totals from the mol ledger (anhydrous + parcel element vectors +
    free/gel water + dissolved cluster inventories). Bound water lives inside
    the parcels' element vectors — variable-composition phases (CSHQ) are never
    reinterpreted as fixed formulas."""
    e = np.zeros(len(ELEMENT_IDS))
    for i, p in enumerate(KINETIC_PHASE_IDS):
        e += formula_elements(registry.get(p).formula) * state.phase_mol[i]
    e = e + state.hydrate_elements
    e += formula_elements(registry.get("H2O").formula) * (
        state.water_free_mol + state.water_gel_mol)
    if state.cluster_inventory.size:
        e += state.cluster_inventory.sum(axis=0)
    return e


def check_all(state: SimulationState, registry: Registry,
              placement: Optional[PlacementBalance] = None) -> LedgerReport:
    rep = LedgerReport()

    # 1. element balance (expected = initial + anything the backend injected:
    # redox seeds and solver floors, both tracked exactly)
    cur = current_elements(state, registry)
    expected = state.initial_elements + state.injected_elements
    err = np.abs(cur - expected)
    bound = ELEMENT_ATOL_MOL + ELEMENT_RTOL * np.abs(expected)
    rep.metrics["max_element_err_mol"] = float(err.max())
    if np.any(err > bound):
        bad = [ELEMENT_IDS[i] for i in np.flatnonzero(err > bound)]
        rep.violations.append(f"balance_element:{','.join(bad)}")
    # injected mass (seeds/floors/verification slack) must stay negligible —
    # otherwise the element check would be certifying solver-fabricated mass
    scale = float(np.abs(state.initial_elements).max())
    inj_max = float(np.abs(state.injected_elements).max())
    rep.metrics["injected_max_rel"] = inj_max / scale if scale > 0.0 else 0.0
    if scale > 0.0 and inj_max > 1e-6 * scale:
        rep.violations.append("balance_element:injected_excess")

    # 2. voxel occupancy identity (no negatives, no clipping)
    total = (state.anhydrous_fraction.sum(axis=0) + state.hydrate_fraction.sum(axis=0)
             + state.capillary_liquid + state.capillary_gas)
    rep.metrics["voxel_identity_max_err"] = float(np.abs(total - 1.0).max())
    if rep.metrics["voxel_identity_max_err"] > VOXEL_IDENTITY_TOL:
        rep.violations.append("balance_voxel:identity")
    min_channel = min(state.anhydrous_fraction.min(), state.hydrate_fraction.min(),
                      state.capillary_liquid.min(), state.capillary_gas.min())
    rep.metrics["min_channel_value"] = float(min_channel)
    if min_channel < -VOXEL_IDENTITY_TOL:
        rep.violations.append("balance_voxel:negative")

    # 3. water partition: free + gel + bound = initial total
    w_err = abs(state.water_free_mol + state.water_gel_mol + state.water_bound_mol
                - state.initial_water_mol)
    rep.metrics["water_partition_err_mol"] = w_err
    if w_err > ELEMENT_ATOL_MOL + ELEMENT_RTOL * abs(state.initial_water_mol):
        rep.violations.append("balance_water:partition")

    # 4. dense arrays vs mol ledger (volumes in voxel units)
    max_rel = 0.0
    for i, p in enumerate(KINETIC_PHASE_IDS):
        dense = float(state.anhydrous_fraction[SOLID_PHASE_IDS.index(p)].sum())
        led = state.phase_mol[i] * state.vm_vox(registry, p)
        max_rel = max(max_rel, abs(dense - led) / (1.0 + abs(led)))
    for i in range(len(state.hydrate_ids)):
        dense = float(state.hydrate_fraction[i].sum())
        led = float(state.hydrate_env_vol_vox[i])
        max_rel = max(max_rel, abs(dense - led) / (1.0 + abs(led)))
    liq = float(state.capillary_liquid.sum())
    # free water fills capillary space; gel water volume lives inside hydrate envelopes
    led_liq = state.water_free_mol * state.vm_vox(registry, "H2O")
    max_rel = max(max_rel, abs(liq - led_liq) / (1.0 + abs(led_liq)))
    inert_dense = float(state.anhydrous_fraction[SOLID_PHASE_IDS.index(INERT_PHASE_ID)].sum())
    max_rel = max(max_rel, abs(inert_dense - state.inert_volume_vox)
                  / (1.0 + abs(state.inert_volume_vox)))
    rep.metrics["dense_ledger_max_rel_err"] = max_rel
    if max_rel > DENSE_LEDGER_RTOL:
        rep.violations.append("balance_voxel:dense_vs_ledger")

    # 5. placement balance
    if placement is not None:
        scale = 1.0 + abs(placement.requested_bulk_vol_vox)
        e1 = abs(placement.backend_bulk_vol_vox - placement.requested_bulk_vol_vox) / scale
        e2 = abs(placement.requested_bulk_vol_vox - placement.placed_bulk_vol_vox) / scale
        rep.metrics["placement_rel_err"] = max(e1, e2)
        if max(e1, e2) > PLACEMENT_RTOL:
            rep.violations.append("balance_placement")

    return rep
