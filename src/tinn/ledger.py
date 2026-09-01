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
class ExchangeBalance:
    """v4.0/RT mode C (PRD §6.1 교환 수지): the summed inventory change the
    transport operator applied must equal the net boundary exchange (zero
    while sealed). Edges are applied antisymmetrically, so a violation is
    bookkeeping corruption, not discretization error."""
    applied_delta_elements: np.ndarray   # (E,) sum over domains
    boundary_net_elements: np.ndarray    # (E,) + = into the system
    abs_flux_scale: float                # sum |q| for the relative tolerance


@dataclass
class SorptionBalance:
    """Tier 1 (RT-S1, PRD 6.1 extension): the S stage moves mass between
    the pore solution and the sorbed store with the SAME floats, so the
    two applied deltas must cancel exactly - a violation is bookkeeping
    corruption, not discretization error."""
    applied_inventory_delta: np.ndarray   # (E,) sum over domains
    applied_sorbed_delta: np.ndarray      # (E,)
    abs_scale: float                      # sum |moved| for the tolerance


@dataclass
class DomainPartition:
    """v4.0/RT mode C (PRD §6.1 도메인 분할 정합): the domain map must refine
    the cluster map — every wet voxel in a domain, every domain inside
    exactly one cluster (integer-exact check)."""
    labels: np.ndarray                   # (N,N,N) cluster labels
    domain_id: np.ndarray                # (N,N,N)
    domain_to_cluster: np.ndarray        # (D,)


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
    if state.domain_sorbed_mol.size:
        # Tier 1 (RT-S1): surface-sorbed elements are part of the total -
        # solution + solids + sorbed = everything (zero rows until the
        # sorption operator is active, so pre-S1 behavior is unchanged)
        e += state.domain_sorbed_mol.sum(axis=0)
    return e


def check_all(state: SimulationState, registry: Registry,
              placement: Optional[PlacementBalance] = None,
              exchange: Optional[ExchangeBalance] = None,
              partition: Optional[DomainPartition] = None,
              sorption: Optional[SorptionBalance] = None) -> LedgerReport:
    rep = LedgerReport()

    if sorption is not None:
        # RT-S1: same-float transfer between solution and the sorbed store
        gap = np.abs(sorption.applied_inventory_delta
                     + sorption.applied_sorbed_delta)
        bound = ELEMENT_ATOL_MOL + 1e-12 * sorption.abs_scale
        rep.metrics["sorption_balance_max_mol"] = float(gap.max())
        if np.any(gap > bound):
            rep.violations.append("balance_sorption")

    # 1. element balance (expected = initial + anything the backend injected:
    # redox seeds and solver floors, both tracked exactly)
    cur = current_elements(state, registry)
    # boundary_exchanged_elements is the RT-W3 reservation: zeros until
    # boundary reservoirs exist, so this is a no-op while sealed
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
        # (c) per-endmember holdings must stay non-negative beyond vanish dust
        # (same GLOBAL scale reference as (a)/(b)). The closure identities
        # cannot see an overdraft — both sides move by the same fed amounts —
        # so a material negative here is the ONLY witness that some cluster
        # was fed composition that does not exist (E2 review finding).
        bound_n = ELEMENT_ATOL_MOL + ELEMENT_RTOL * scale_mol
        rep.metrics["endmember_min_mol"] = float(em.min()) if len(em) else 0.0
        if np.any(-em > bound_n):
            bad = sorted({f"{h}:{dc}" for j in np.flatnonzero(-em > bound_n)
                          for h, dc in (state.endmember_ids[j],)})
            rep.violations.append(f"balance_endmember_negative:{','.join(bad)}")

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

    # 3. water partition: free + gel + bound = initial + boundary-exchanged
    # solvent (v4.0/RT: the boundary term is zero until RT-W3 reservoirs)
    w_expected = state.initial_water_mol + state.boundary_water_mol
    w_err = abs(state.water_free_mol + state.water_gel_mol + state.water_bound_mol
                - w_expected)
    rep.metrics["water_partition_err_mol"] = w_err
    if w_err > ELEMENT_ATOL_MOL + ELEMENT_RTOL * abs(w_expected):
        rep.violations.append("balance_water:partition")

    # v4.0/RT mode C gates (PRD 6.1)
    if partition is not None:
        d_id = partition.domain_id
        wet_cl = partition.labels >= 0
        wet_dom = d_id >= 0
        bad_cover = bool(np.any(wet_cl != wet_dom))
        bad_parent = False
        if not bad_cover and partition.domain_to_cluster.size:
            mapped = np.where(wet_dom,
                              partition.domain_to_cluster[
                                  np.clip(d_id, 0, None)], -1)
            bad_parent = bool(np.any(mapped[wet_dom]
                                     != partition.labels[wet_dom]))
        if bad_cover or bad_parent:
            rep.violations.append("balance_domain:partition")
    if exchange is not None:
        ex_err = np.abs(exchange.applied_delta_elements
                        - exchange.boundary_net_elements)
        ex_bound = ELEMENT_ATOL_MOL + 1e-12 * exchange.abs_flux_scale
        rep.metrics["exchange_net_err_mol"] = float(ex_err.max())
        if np.any(ex_err > ex_bound):
            rep.violations.append("balance_exchange")

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
