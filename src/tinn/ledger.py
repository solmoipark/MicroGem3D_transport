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
    """Backend volume == requested == placed, on BOTH legs: growth and, under
    full re-equilibration, removal (PRD §6.1 배치 수지, gross accounting)."""
    backend_bulk_vol_vox: float
    requested_bulk_vol_vox: float
    placed_bulk_vol_vox: float
    backend_removal_vol_vox: float = 0.0
    removed_vol_vox: float = 0.0


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
    e = e + state.hydrate_elements_ch.sum(axis=0)
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
    # boundary_exchanged_elements is the RT-W2 reservation (FORMAT_VERSION 3):
    # zeros today, so this is a no-op until boundary reservoirs exist
    expected = (state.initial_elements + state.injected_elements
                + state.boundary_exchanged_elements)
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

    # 1b. endmember closure (E1, PRD 6.1 rev.3), two identities per channel:
    #     (a) sum of the channel's endmember mols == hydrate_mol[h]
    #     (b) endmember mols x DCH element rows == hydrate_elements_ch[h]
    # Both sides are accumulated from the SAME backend responses, so any gap
    # is bookkeeping corruption, never physics.
    if len(state.endmember_ids):
        em = state.endmember_mol
        n_h = len(state.hydrate_ids)
        h_of = {h: i for i, h in enumerate(state.hydrate_ids)}
        sum_mol = np.zeros(n_h)
        elem_from_em = np.zeros((n_h, len(ELEMENT_IDS)))
        for j, (h, _dc) in enumerate(state.endmember_ids):
            hi = h_of[h]
            sum_mol[hi] += em[j]
            elem_from_em[hi] += em[j] * state.endmember_elements[j]
        # tolerance reference: a channel that FULLY redissolves keeps signed
        # float dust ~eps x its pre-vanish holdings while its post-step
        # magnitude collapses to ~0 — bounding by the post-step channel value
        # alone misreports that legitimate dust as corruption (review finding,
        # reproduced at 128^3 scales). The dust is bounded by eps x total
        # turnover, so the reference includes the GLOBAL holdings scale; real
        # corruption remains orders of magnitude above rtol x global.
        scale_mol = float(np.abs(state.hydrate_mol).sum())
        err_m = np.abs(sum_mol - state.hydrate_mol)
        bound_m = ELEMENT_ATOL_MOL + ELEMENT_RTOL * np.maximum(
            np.abs(state.hydrate_mol), scale_mol)
        rep.metrics["endmember_sum_err_mol"] = float(err_m.max()) if n_h else 0.0
        if np.any(err_m > bound_m):
            bad = [state.hydrate_ids[i] for i in np.flatnonzero(err_m > bound_m)]
            rep.violations.append(f"balance_endmember:{','.join(bad)}")
        col_scale = np.abs(state.hydrate_elements_ch).sum(axis=0)  # (E,)
        err_e = np.abs(elem_from_em - state.hydrate_elements_ch)
        bound_e = ELEMENT_ATOL_MOL + ELEMENT_RTOL * np.maximum(
            np.abs(state.hydrate_elements_ch), col_scale[None, :])
        rep.metrics["endmember_element_err_mol"] = float(err_e.max()) if n_h else 0.0
        if np.any(err_e > bound_e):
            rows = sorted({state.hydrate_ids[i]
                           for i in np.flatnonzero(np.any(err_e > bound_e, axis=1))})
            rep.violations.append(
                f"balance_endmember_elements:{','.join(rows)}")

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
        scale_r = 1.0 + abs(placement.backend_removal_vol_vox)
        e3 = abs(placement.backend_removal_vol_vox - placement.removed_vol_vox) / scale_r
        rep.metrics["placement_rel_err"] = max(e1, e2, e3)
        if max(e1, e2, e3) > PLACEMENT_RTOL:
            rep.violations.append("balance_placement")

    return rep
