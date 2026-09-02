"""Transactional orchestrator: trial -> kinetics -> dissolution -> backend ->
morphology -> ledger checks -> atomic commit or full rollback with dt halving.

The engine is the only mutator of SimulationState. Rejection reasons are stable
identifiers: placement_capacity, insufficient_water, cluster_dryout,
backend_failure, balance_* (from ledger checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import analysis, backend as backend_mod
from . import dissolution, ledger, morphology, storage, transport
from .config import TinnConfig
from .geometry import initialize_rve
from .kinetics import KineticsModel, make_kinetics
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                       Registry, SALT_PHASE_IDS, SOLID_PHASE_IDS,
                       registry_for)
from .state import SimulationState, code_version

REJECT_BACKEND_FAILURE = "backend_failure"
REJECT_CLUSTER_DRYOUT = "cluster_dryout"
REJECT_TRANSPORT_FAILURE = "transport_failure"

# PRD 4.6.3 dryout surrender: both witnesses (solute row vs the aqueous
# pool, parent-cluster water vs total water) must sit below this fraction
# for a whole-cluster death to surrender its row to the boundary ledger
# instead of rejecting. W4.1 measured margins (leach OPC 32^3): dust rows
# of dying orphan pockets 1e-7..1e-6 of pool with water 7e-7 of total;
# the smallest MATERIAL rows (percolated-cluster bands, which cannot die
# whole) start at 6e-5 with water ratio 1.0 - three-plus decades of
# separation on each side of 1e-3.
DRYOUT_SURRENDER_REL = 1e-3


class EngineError(RuntimeError):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class StepReject:
    reason: str


def _dryout_surrender(rows: np.ndarray, hard: List[int],
                      dom_to_cl: np.ndarray, cl_labels: np.ndarray,
                      prev_liquid: np.ndarray) -> Optional[List[int]]:
    """v4.0/RT (PRD 4.6.3): disposition of whole-cluster deaths under an
    ACTIVE bath. The sealed->exposed switch orphans wrap-connected pocket
    clusters; they self-desiccate (their own equilibrium binds the last
    water) and vanish from the labeling in the same step - a real,
    dt-independent event, not a defect. A dead cluster whose row is float
    dust surrenders it to the boundary ledger (exact floats, closure
    identity untouched - the flush precedent); a dead cluster carrying
    MATERIAL solutes or water keeps the v2 hard reject. Returns the
    surrendered domain ids, or None if any death is material."""
    pool = float(np.abs(rows).sum())
    tot_liq = float(prev_liquid[prev_liquid > 0.0].sum())
    out: List[int] = []
    for pd in hard:
        row_sum = float(np.abs(rows[pd]).sum())
        cl_liq = float(prev_liquid[cl_labels == int(dom_to_cl[pd])].sum())
        if (row_sum >= DRYOUT_SURRENDER_REL * pool
                or cl_liq >= DRYOUT_SURRENDER_REL * tot_liq):
            return None
        out.append(int(pd))
    return out


def _neighbor_best_label(labels: np.ndarray, liquid: np.ndarray,
                         periodic_axes=(True, True, True)) -> np.ndarray:
    """Label of the neighboring cluster with the largest liquid contact — the
    cluster that actually supplies the dissolution weight (deterministic:
    argmax over the fixed axis order breaks ties). Liquid is clamped to >= 0:
    placement float dust can leave ~-1e-19 in a voxel, and a labeled neighbor
    must never be disqualified by noise (it would stall the ring propagation
    for coated sites and misreport cluster_dryout)."""
    labs = []
    liqs = []
    for ax, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        lab_r = np.roll(labels, shift, axis=ax)
        liq_r = np.roll(liquid, shift, axis=ax)
        if not periodic_axes[ax]:
            # v4.0/RT-W3: an exposed axis is a wall for attribution too -
            # a dry face site must not feed the reactor on the OPPOSITE
            # face through the wrap (review repro: released mol teleported
            # across the sealed/exposed cut)
            idx: list = [slice(None)] * 3
            idx[ax] = 0 if shift > 0 else -1
            lab_r[tuple(idx)] = np.int64(-1)
            liq_r[tuple(idx)] = -1.0
        labs.append(lab_r)
        liqs.append(liq_r)
    labs = np.stack(labs)
    liqs = np.where(labs >= 0, np.maximum(np.stack(liqs), 0.0), -1.0)
    best = np.argmax(liqs, axis=0)
    best_lab = np.take_along_axis(labs, best[None], axis=0)[0]
    best_liq = np.take_along_axis(liqs, best[None], axis=0)[0]
    return np.where(best_liq >= 0.0, best_lab, np.int64(-1))



_SORB_SOLUTE_COLS = [i for i, el in enumerate(ELEMENT_IDS)
                     if el not in ("O", "H")]


def sorption_reactor_dry(water_mol: float, offer: np.ndarray,
                         dust_floor: float) -> bool:
    """S-stage wetness/dust contract (RT-S1c, PRD 4.6.5): a reactor is an
    aqueous phase only when water dominates the solutes (mol ratio >= 10)
    AND the solute total sits above the ledger noise floor. Pure function
    so the contract stays unit-testable against the measured failure
    inputs (the femto-water dust cluster, the solute-noise micro-pocket)."""
    solute = float(np.clip(offer[_SORB_SOLUTE_COLS], 0.0, None).sum())
    return (water_mol <= 0.0 or water_mol < 10.0 * solute
            or solute <= dust_floor)


class Engine:
    def __init__(self, config: TinnConfig, registry: Optional[Registry] = None,
                 reaction_backend: Optional[backend_mod.ReactionBackend] = None,
                 kinetics: Optional[KineticsModel] = None):
        self.config = config
        self.registry = registry or registry_for(config)
        if reaction_backend is not None:
            self.backend = reaction_backend
        elif config.chemistry.backend == "stoichiometric":
            self.backend = backend_mod.StoichiometricBackend(
                config.chemistry.stoichiometric_rules, self.registry)
        else:
            import os
            from .gems import GemsBackend, GemsWorker
            worker = GemsWorker(
                config.chemistry.gems_bundle_lst,
                python_executable=(os.environ.get("TINN_GEMS_PYTHON")
                                   or config.chemistry.gems_worker_python),
                work_root=None)
            self.backend = GemsBackend(worker, config.temperature_K,
                                       registry=self.registry)
        self.kinetics = kinetics or make_kinetics(config)
        self.hydrate_ids = tuple(self.backend.hydrate_ids)
        # E1 endmember metadata (PRD 2.3 rev.3): backend-declared; a backend
        # without it (legacy test fakes) gets the single-endmember default in
        # SimulationState.from_geometry
        self._hydrate_endmembers = getattr(
            self.backend, "hydrate_endmembers",
            {h: (h,) for h in self.hydrate_ids})
        self._endmember_elements = getattr(self.backend, "endmember_elements",
                                           None)
        # per-channel slices into the flat endmember vector, fixed run-scoped
        self._em_index: Dict[Tuple[str, str], int] = {}
        self._em_slice: Dict[str, slice] = {}
        pos = 0
        for h in self.hydrate_ids:
            dcs = self._hydrate_endmembers[h]
            self._em_slice[h] = slice(pos, pos + len(dcs))
            for dc in dcs:
                self._em_index[(h, dc)] = pos
                pos += 1
        self._n_em = pos
        # single implementation of the gel-porosity policy lives in analysis
        self._gel_eps = analysis.gel_porosity_vector(config, self.hydrate_ids,
                                                     self.registry)
        # E3: kinetic-vector positions of the soluble salt carriers, whose
        # demand rule differs (cumulative, see try_step)
        self._salt_channels = np.array(
            [KINETIC_PHASE_IDS.index(p) for p in SALT_PHASE_IDS], dtype=np.intp)
        # v4.0/RT mode B (PRD 4.6.1): per-channel exchange time in hours.
        # Solid solutions (multi-endmember channels) take the global tau;
        # single-endmember crystallines stay at 0 (always fully offered)
        # unless overridden per phase. None = full re-equilibration.
        self._tau_ch: Optional[np.ndarray] = None
        tr = config.transport
        if tr is not None and tr.rate_limited():
            tau = np.zeros(len(self.hydrate_ids))
            if tr.exchange_tau_h is not None:
                for i, h in enumerate(self.hydrate_ids):
                    if len(self._hydrate_endmembers[h]) > 1:
                        tau[i] = tr.exchange_tau_h
            for name, t in (tr.exchange_tau_h_per_phase or {}).items():
                if name not in self.hydrate_ids:
                    raise ValueError(
                        f"exchange_tau_h_per_phase names unknown hydrate "
                        f"channel {name!r}; this bundle declares: "
                        f"{sorted(self.hydrate_ids)}")
                tau[self.hydrate_ids.index(name)] = t
            if not np.any(tau > 0.0):
                # the config DECLARED rate limiting (its hash records mode B)
                # but no channel of this bundle ends up limited — running
                # exact legacy physics under a mode-B label is the silent
                # no-op the config validator refuses elsewhere (PRD 5)
                raise ValueError(
                    "transport declares rate limiting but no hydrate channel "
                    "of this bundle gets a positive exchange time (a global "
                    "exchange_tau_h needs at least one multi-endmember "
                    "solid-solution channel; per-phase 0.0 disables its "
                    "channel) - refusing a silent full-re-equilibration run "
                    "under a mode-B config hash")
            self._tau_ch = tau
        # v4.0/RT mode C (PRD 4.6.2): sub-cluster equilibration domains
        self._domains_cfg = (config.transport.domains
                             if config.transport is not None else None)
        if self._domains_cfg is not None:
            h_m = config.rve.voxel_size_um * 1e-6
            vox2_h_per_m2_s = 3600.0 / (h_m * h_m)
            self._d0_vox2_h = (
                None if self._domains_cfg.d0_m2_s is None
                else self._domains_cfg.d0_m2_s * vox2_h_per_m2_s)
            # Tier 0 (RT-P0, PRD 4.6.4): per-species diffusivities from the
            # vendored dw table, resolved against the RUN's aqueous species
            # universe (bundle DCH via the backend) - never hardcoded
            self._np_cfg = self._domains_cfg.species
            if self._np_cfg is not None:
                from .gems import GemsError, load_species_dw
                worker = getattr(self.backend, "_worker", None)
                if worker is not None:
                    worker.require_speciation()
                aq_ids = getattr(self.backend, "aq_species_ids", ())
                if not aq_ids:
                    raise RuntimeError(
                        "transport.domains.species needs a backend with an "
                        "aqueous species universe (gems3k)")
                table = load_species_dw(self._np_cfg.dw_table)
                se = transport.stokes_einstein_factor(config.temperature_K)
                dw = np.zeros(len(aq_ids))
                z_dch = np.asarray(self.backend.aq_species_charge,
                                   dtype=np.float64)
                defaulted = []
                unmapped = []
                for k, dc in enumerate(aq_ids):
                    hit = table.lookup(dc)
                    if hit is not None:
                        dw[k] = hit[0]
                        if hit[1] != z_dch[k]:
                            raise GemsError(
                                f"dw table charge {hit[1]:+g} for {dc!r} "
                                f"contradicts the bundle DCH ({z_dch[k]:+g})"
                                f" - alias curation error", kind="config")
                    elif self._np_cfg.default_dw_m2_s is not None:
                        dw[k] = self._np_cfg.default_dw_m2_s
                        defaulted.append(dc)
                    else:
                        unmapped.append(dc)
                if unmapped:
                    raise GemsError(
                        f"species dw table maps no diffusivity for "
                        f"{len(unmapped)} aqueous species (e.g. "
                        f"{unmapped[:10]}) and default_dw_m2_s is null - "
                        f"declare a default or extend the aliases",
                        kind="config")
                self._np_dc_index = {dc: k for k, dc in enumerate(aq_ids)}
                self._np_dw_vox2_h = (dw * se * self._np_cfg.geometry_factor
                                      * vox2_h_per_m2_s)
                self._np_z = z_dch
                self._np_nu = np.asarray(self.backend.aq_species_elements,
                                         dtype=np.float64)
                self._np_provenance = {
                    "dw_table": self._np_cfg.dw_table,
                    "dw_table_sha256": table.sha256,
                    "dw_source": table.source,
                    "se_factor": se,
                    "geometry_factor": self._np_cfg.geometry_factor,
                    "default_dw_m2_s": self._np_cfg.default_dw_m2_s,
                    "default_dw_species": defaulted,
                    "diagnostics_only": self._np_cfg.diagnostics_only,
                }
            # RT-W2b GEM-call economy active iff either knob departs from
            # the equilibrate-everything default
            self._economy = (self._domains_cfg.dirty_rtol > 0.0
                             or self._domains_cfg.max_gem_calls_per_step
                             is not None)
            if self._economy:
                # deferred domains raise the memo pressure: quiescent
                # inputs recur across steps (PRD 4.6.2)
                tz, ty, tx = self._domains_cfg.tiles(config.rve.grid_size)
                n_tiles = ((config.rve.grid_size // tz)
                           * (config.rve.grid_size // ty)
                           * (config.rve.grid_size // tx))
                cap = max(256, min(4 * n_tiles, 8192))
                if hasattr(self.backend, "memo_cap"):
                    self.backend.memo_cap = cap
            # per-kinetic-phase element rows for booking skipped releases
            # into the inventory element-exactly
            from .registry import element_vector
            self._kin_elements = np.stack([
                element_vector(self.registry.get(p).formula, 1.0)
                for p in KINETIC_PHASE_IDS])
        else:
            self._economy = False
            self._np_cfg = None
        # v4.0/RT-W3 boundary reservoir (PRD 4.6.3): precompute the static
        # parts; the coupling itself is rebuilt each step from the labeled
        # liquid (pure function of state - restart determinism free)
        # Tier 1 (RT-S1b): PHREEQC surface-sorption operator. Independent
        # of transport.domains - without mode C the connected cluster is
        # the reactor, same as the chemistry.
        self._sorb_op = None
        self._sorb_density = None
        if config.sorption is not None:
            if "CSHQ" not in self.hydrate_ids:
                raise RuntimeError(
                    "sorption needs the CSHQ solid solution as the sorbent "
                    f"- this bundle declares {self.hydrate_ids[:6]}...")
            if (any(el in ("Na", "K") for el in config.sorption.elements)
                    and "CNASH" in self.hydrate_ids):
                raise RuntimeError(
                    "alkali sorption with a CNASH-bearing bundle would "
                    "double-count alkali uptake (the solid solution binds "
                    "them thermodynamically) - refused, mechanism table")
            cshq_dcs = self._hydrate_endmembers["CSHQ"]
            unknown = sorted(set(config.sorption.site_density_mol_per_mol)
                             - set(cshq_dcs))
            if unknown:
                raise RuntimeError(
                    f"site_density_mol_per_mol names endmembers {unknown} "
                    f"the bundle's CSHQ does not declare ({cshq_dcs})")
            self._sorb_density = np.array(
                [config.sorption.site_density_mol_per_mol.get(dc, 0.0)
                 for dc in cshq_dcs])
            from .backend import SorptionOperator
            self._sorb_op = SorptionOperator(config.sorption,
                                             config.temperature_K)
        self._boundary = (config.transport.boundary
                          if config.transport is not None else None)
        if self._boundary is not None:
            self._b_axis = {"z": 0, "y": 1, "x": 2}[self._boundary.axis]
            self._b_low = self._boundary.side in ("low", "both")
            self._b_high = self._boundary.side in ("high", "both")
            self._periodic_axes = tuple(
                i != self._b_axis for i in range(3))
            start = self._boundary.start_h or 0.0
            if start > 0.0:
                # snap to the matching output time (the validator guarantees
                # one within 1e-9): activation and the switch trigger then
                # compare EXACT floats - review finding: a 1e-10 offset
                # passed validation but de-periodized mid-window with no
                # row remap
                start = min(config.schedule.output_times_h,
                            key=lambda t: abs(t - start))
            self._b_start = start
            # mol per m^3 of capillary water -> mol per vox^3 of liquid
            vox_m3 = (config.rve.voxel_size_um * 1e-6) ** 3
            c_res = np.zeros(len(ELEMENT_IDS))
            for el, c in self._boundary.composition_mol_per_m3.items():
                c_res[ELEMENT_IDS.index(el)] = c * vox_m3
            self._c_res_vox = c_res

    # ------------------------------------------------------------------ setup
    def initial_state(self) -> SimulationState:
        rve = initialize_rve(self.config, self.registry)
        st = SimulationState.from_geometry(
            self.config, self.registry, rve, self.backend.backend_id,
            hydrate_ids=self.hydrate_ids,
            hydrate_endmembers=self._hydrate_endmembers,
            endmember_elements=self._endmember_elements)
        if self._np_cfg is not None and not self._np_cfg.diagnostics_only:
            # RT-P0b: the speciation cache's column order is run-scoped and
            # header-owned (endmember_ids precedent); diagnostics-only runs
            # stay stateless
            st.aq_species_ids = tuple(self.backend.aq_species_ids)
        return st

    # ------------------------------------------------------------------- step
    def try_step(self, state: SimulationState, dt_h: float
                 ) -> Tuple[Optional[SimulationState], Optional[StepReject], dict]:
        """One trial step of size dt_h. Returns (trial, None, metrics) on success
        or (None, StepReject, metrics) on rejection; `state` is never mutated."""
        trial = state.clone()
        reg = self.registry
        n_h = len(self.hydrate_ids)
        h_index = {h: i for i, h in enumerate(self.hydrate_ids)}
        vm_w = trial.vm_vox(reg, "H2O")
        gel_eps = self._gel_eps

        # kinetic target for THIS interval only: dn = initial_mol * delta_alpha
        # (PRD §4.2). A shortfall stays recorded as cumulative unmet — it is
        # never re-demanded, so per-step demand always shrinks with dt and a
        # transient blockage cannot balloon into an unplaceable catch-up burst.
        d_alpha = (self.kinetics.alpha_at(trial.time_h + dt_h)
                   - self.kinetics.alpha_at(trial.time_h))
        dn = np.clip(trial.initial_phase_mol * d_alpha, 0.0, trial.phase_mol)
        # Soluble salt carriers are SOLUBILITY-controlled, not rate-controlled
        # (PRD 4.2 v3.0/E3): gypsum keeps dissolving while the solution is
        # undersaturated and stops when it is not, and the alkali sulfates are
        # readily soluble. There is no rate law to write down, so the engine
        # does not invent one — it offers the equilibrium everything the water
        # can currently reach and lets GEMS decide what stays solid. Anything
        # the equilibrium keeps comes back as its own solid phase (these
        # phases are never suppressed), so the carriers are governed by their
        # solubility product rather than by a fitted time constant.
        if self._salt_channels.size:
            s = self._salt_channels
            dn[s] = np.minimum(
                dissolution.accessible_mol(trial, reg, SALT_PHASE_IDS)[s],
                trial.phase_mol[s])

        prev_liquid = trial.capillary_liquid.copy()
        # RT-W4: a declared bath activates at start_h (an output boundary,
        # validated) - before it the run is SEALED and fully periodic,
        # bit-identical to a boundary-free config (mature-paste protocol:
        # hydrate first, then expose)
        bath_active = (self._boundary is not None
                       and trial.time_h >= self._b_start - 1e-12)
        axes_now = self._periodic_axes if bath_active else (True, True, True)
        cl_labels, n_cl = transport.label_clusters(prev_liquid, axes_now)
        # v4.0/RT mode C (PRD 4.6.2): the reactor key is the DOMAIN — cluster
        # intersected with a static tile. The degenerate tile == grid case
        # WITHOUT the economy takes the mode-full aliases outright, so the
        # 1-domain bit-identity gate holds structurally (every mode-C branch
        # keys on dom_to_cl is None), not by numerical luck; with the
        # economy on, tile == grid legitimately means cluster-level
        # deferral and keeps the domain machinery.
        dom_active = self._domains_cfg is not None and (
            any(t != trial.grid_size
                for t in self._domains_cfg.tiles(trial.grid_size))
            or self._economy or bath_active)
        if dom_active:
            labels, n_clusters, dom_to_cl = transport.label_domains(
                cl_labels, n_cl,
                self._domains_cfg.tiles(trial.grid_size))
        else:
            labels, n_clusters, dom_to_cl = cl_labels, n_cl, None

        dis = dissolution.dissolve(trial, reg, dn)
        vacated = dis.removed_vol.sum(axis=0)

        # site -> cluster attribution (own label, else wettest neighboring cluster)
        site_cluster = np.where(labels >= 0, labels,
                                _neighbor_best_label(labels, prev_liquid,
                                                     axes_now))
        site_mask = vacated > 0.0
        # a coated site (gel-conduit dissolution, PRD §4.2) may sit several
        # voxels from any liquid: propagate labels ring by ring, RELAYED ONLY
        # through gel-bearing voxels or the sites themselves — the physical
        # conduit is gel pore water, so labels must not flood across capillary
        # gas or bare particle interiors (rev.2 review finding). Applied to
        # SITE cells only so the recon ownership of non-site dry regions is
        # untouched.
        if np.any(site_mask & (site_cluster < 0)):
            relay = ((trial.hydrate_fraction.sum(axis=0) > transport.LIQ_EPS)
                     | site_mask)
            deep = site_cluster.copy()
            for _ in range(2 * trial.grid_size):
                if not (site_mask & (deep < 0)).any():
                    break
                nxt = np.where(relay,
                               _neighbor_best_label(deep, prev_liquid,
                                                    axes_now),
                               np.int64(-1))
                nxt = np.where(deep >= 0, deep, nxt)
                if int((nxt >= 0).sum()) == int((deep >= 0).sum()):
                    deep = nxt  # front stopped growing — rest is unreachable
                    break
                deep = nxt
            site_cluster = np.where(site_mask, deep, site_cluster)
        # a site with no conduit path to ANY cluster cannot dissolve — its
        # target goes back to the solid as honest unmet (never a reject: the
        # shortfall is dt-independent, so rejecting could only abort the run;
        # rev.2 review finding: total dryout must degrade gracefully too)
        unreachable = site_mask & (site_cluster < 0)
        if unreachable.any():
            for k, p in enumerate(KINETIC_PHASE_IDS):
                give_back = np.where(unreachable, dis.removed_vol[k], 0.0)
                back_vol = float(give_back.sum())
                if back_vol <= 0.0:
                    continue
                chan = SOLID_PHASE_IDS.index(p)
                trial.anhydrous_fraction[chan] += give_back
                dis.removed_vol[k] -= give_back
                vm = trial.vm_vox(reg, p)
                dis.removed_mol[k] -= back_vol / vm
                dis.unmet_mol[k] += back_vol / vm
            vacated = dis.removed_vol.sum(axis=0)
            site_mask = vacated > 0.0

        # per-cluster released mol and water availability
        released = np.zeros((n_clusters, len(KINETIC_PHASE_IDS)))
        for k, p in enumerate(KINETIC_PHASE_IDS):
            vm = trial.vm_vox(reg, p)
            vols = np.bincount(site_cluster[site_mask],
                               weights=dis.removed_vol[k][site_mask],
                               minlength=n_clusters)
            released[:, k] = vols / vm
        liq_vol_c = np.bincount(labels[labels >= 0],
                                weights=prev_liquid[labels >= 0],
                                minlength=n_clusters)
        water_mol_c = liq_vol_c / vm_w

        # RT-W4 sealed -> exposed switch (start_h): the stored rows were
        # keyed on the fully periodic partition; the first bath-active
        # step de-periodizes the exposed axis, splitting clusters that
        # were connected ONLY through the seam. Remap the rows onto the
        # new partition conservatively - identical liquid on both sides,
        # so the transfer is an exact liquid-volume split and a dryout is
        # impossible. Pure function of (config, state): restart-safe.
        inv_rows = state.cluster_inventory
        pool_rows = state.cluster_endmember_mol
        spec_state_rows = state.domain_species_mol
        sorb_state_rows = state.domain_sorbed_mol
        eq_rows_ok = True
        if (bath_active and self._b_start > 0.0
                and abs(trial.time_h - self._b_start) <= 1e-12):
            # First bath-active step. De-periodization can SPLIT seam-only
            # clusters while PRESERVING the domain count and permuting ids
            # (three review repros), so no count test can detect it: remap
            # UNCONDITIONALLY from the recomputed old periodic partition
            # (identical liquid on both sides = exact volume split; a pure
            # identity transfer is frac = 1.0, bit-exact). Economy
            # snapshots never survive the event - all domains forced.
            old_cl, old_ncl = transport.label_clusters(prev_liquid)
            if dom_to_cl is not None:
                old_labels, old_n, _ = transport.label_domains(
                    old_cl, old_ncl, self._domains_cfg.tiles(trial.grid_size))
            else:
                old_labels, old_n = old_cl, old_ncl
            eq_rows_ok = False
            if inv_rows.shape[0] == old_n and old_n > 0                     and not np.array_equal(old_labels, labels):
                sw = transport.remap_inventories(
                    old_labels, prev_liquid, labels, prev_liquid,
                    inv_rows, n_clusters)
                assert not sw.dryout
                inv_rows = sw.inventory
                if pool_rows.shape[0] == old_n:
                    pool_rows = transport.remap_inventories(
                        old_labels, prev_liquid, labels, prev_liquid,
                        pool_rows, n_clusters).inventory
                if spec_state_rows.shape[0] == old_n:
                    spec_state_rows = transport.remap_inventories(
                        old_labels, prev_liquid, labels, prev_liquid,
                        spec_state_rows, n_clusters).inventory
                if sorb_state_rows.shape[0] == old_n:
                    sorb_state_rows = transport.remap_inventories(
                        old_labels, prev_liquid, labels, prev_liquid,
                        sorb_state_rows, n_clusters).inventory

        if inv_rows.shape[0] == n_clusters:
            inv_in = inv_rows
        elif inv_rows.shape[0] == 0:
            inv_in = np.zeros((n_clusters, len(ELEMENT_IDS)))
        else:
            raise RuntimeError(
                f"cluster inventory has {inv_rows.shape[0]} rows but "
                f"{n_clusters} clusters were labeled - state is corrupted (no fallback)")
        # E2: per-cluster endmember pools ride the same labeling contract
        if pool_rows.shape[0] == n_clusters \
                and pool_rows.shape[1] == self._n_em:
            em_pool_in = pool_rows
        elif pool_rows.shape[0] == 0:
            em_pool_in = np.zeros((n_clusters, self._n_em))
        else:
            raise RuntimeError(
                f"cluster endmember pool has shape "
                f"{pool_rows.shape} but "
                f"({n_clusters}, {self._n_em}) was expected - state is "
                f"corrupted (no fallback)")
        # RT-S1b: the sorbed store rides the same labeling contract
        sorb_in = None
        if self._sorb_op is not None:
            if sorb_state_rows.shape[0] == n_clusters:
                sorb_in = sorb_state_rows
            elif sorb_state_rows.shape[0] == 0:
                sorb_in = np.zeros((n_clusters, len(ELEMENT_IDS)))
            else:
                raise RuntimeError(
                    f"sorbed store has {sorb_state_rows.shape[0]} rows but "
                    f"{n_clusters} reactors were labeled - state is "
                    f"corrupted (no fallback)")
        # RT-P0b: frozen speciation rows ride the same labeling contract
        np_active = (self._np_cfg is not None
                     and not self._np_cfg.diagnostics_only)
        spec_in = None
        if np_active:
            n_s = len(self.backend.aq_species_ids)
            if (spec_state_rows.shape[0] == n_clusters
                    and spec_state_rows.shape[1] == n_s):
                spec_in = spec_state_rows
            elif spec_state_rows.shape[0] == 0:
                spec_in = np.zeros((n_clusters, n_s))
            else:
                raise RuntimeError(
                    f"domain speciation cache has shape "
                    f"{spec_state_rows.shape} but ({n_clusters}, {n_s}) was "
                    f"expected - state is corrupted (no fallback)")

        # ---- transport phase (v4.0/RT mode C, Lie split T -> R) -------------
        # implicit solute diffusion between the domains of each cluster on
        # the step-start frozen snapshot; reactors then consume the
        # post-transport inventories. Full mode: no partition, no exchange.
        exchange_bal = None
        exchange_metrics: Dict[str, float] = {}
        bath_cluster_domain = None
        if dom_to_cl is not None:
            inv_eff = inv_in.copy()
            # clip like the report solver (analysis.py): placement dust can
            # leave ~-1e-19 in a voxel, and a negative face conductance
            # breaks the M-matrix property the BE solve relies on (measured:
            # CG stall -> transport_failure at a leached face). The report's
            # FLOOR stays out - only the clip is physics.
            g_field = np.clip(transport.conductance_field(
                prev_liquid, trial.hydrate_fraction, gel_eps,
                analysis.GEL_REL_DIFFUSIVITY), 0.0, None)
            tiles = self._domains_cfg.tiles(trial.grid_size)
            graph = transport.build_domain_graph(
                labels, n_clusters, g_field, prev_liquid, axes_now,
                tuple(float(t) for t in tiles))
            bath = None
            bnd_coupled = None
            if bath_active:
                g_ar = transport.boundary_coupling(
                    labels, n_clusters, g_field, self._b_axis,
                    self._b_low, self._b_high)
                g_ar[graph.dust] = 0.0   # dust domains skip the bath too
                # ghost sits at half the exposed-axis pitch: the /pitch of
                # the TPFA distance lives here, mirroring edge_g
                g_ar = g_ar / float(tiles[self._b_axis])
                bath = transport.BoundaryBath(g_bnd=g_ar,
                                              c_res=self._c_res_vox)
                bnd_coupled = g_ar > 0.0
            if np_active:
                npc = transport.np_effective_conductance(
                    graph, spec_in, graph.water, self._np_dw_vox2_h,
                    self._np_z, self._np_nu,
                    g_bnd=(bath.g_bnd if bath is not None else None),
                    c_res=(bath.c_res if bath is not None else None))
                if self._np_cfg.phi_clamp_report:
                    for key, v in npc.counts.items():
                        exchange_metrics[key] = float(v)
                    exchange_metrics["np_charge_flux_rel_max"] = (
                        npc.charge_flux_rel_max)
                ex = transport.exchange_be(graph, inv_eff, dt_h, None,
                                           bath=bath, np_cond=npc)
            else:
                ex = transport.exchange_be(graph, inv_eff, dt_h,
                                           self._d0_vox2_h, bath=bath)
            if ex.status != "ok":
                return None, StepReject(REJECT_TRANSPORT_FAILURE), {}
            inv_eff = inv_eff + ex.delta
            if bath is not None:
                trial.boundary_exchanged_elements = (
                    trial.boundary_exchanged_elements + ex.boundary_net)
                # bath-sweep floor (W4 measured): the BE drain leaves solute
                # traces four-plus decades below their step-start cluster
                # amounts, and feeding those to GEM sits on the AIA knife
                # edge - measured as near-total per-step nonconvergence
                # freezing, which stalls the front. Physically the RENEWED
                # bath carries the last traces away: flush sub-threshold
                # entries of bath-connected clusters to the boundary
                # ledger (same floats - closure exact; the exchange gate
                # compares only the operator's own fluxes).
                bath_cl = np.unique(dom_to_cl[bnd_coupled])
                in_bath_cl = np.isin(dom_to_cl, bath_cl)
                bath_cluster_domain = in_bath_cl
                # molality floor: 1e-8 mol solute per mol water (~6e-7
                # mol/L) - negligible against any cement pore solution
                # (autoprotolysis itself is 1e-7 mol/L) and far below the
                # post-equilibration resupply (~1e-2 mol/L), yet well
                # above the measured drain residue. Water-based, so it is
                # state-independent and bootstrap-safe.
                sweep = ((inv_eff > 0.0)
                         & (inv_eff < 1e-8 * water_mol_c[:, None])
                         & in_bath_cl[:, None])
                if sweep.any():
                    moved = np.where(sweep, inv_eff, 0.0)
                    inv_eff = inv_eff - moved
                    swept = moved.sum(axis=0)
                    trial.boundary_exchanged_elements = (
                        trial.boundary_exchanged_elements - swept)
                    exchange_metrics["bath_swept_mol"] = float(swept.sum())
            exchange_bal = ledger.ExchangeBalance(
                applied_delta_elements=ex.delta.sum(axis=0),
                boundary_net_elements=ex.boundary_net,
                abs_flux_scale=float(np.abs(ex.delta).sum()))
            exchange_metrics.update({
                "n_domains": float(n_clusters),
                "exchange_cg_iterations": float(ex.cg_iterations),
                "exchange_max_edge_flux_mol": ex.max_edge_flux_mol,
                "exchange_repair_rel": ex.repair_rel,
            })
            if bnd_coupled is not None:
                # supply-limit witness (PRD 4.6.3 dt policy): per-step
                # outflux over the BATH-CONNECTED CLUSTERS' solution
                # inventory (the implicit step draws through the face
                # bands from the whole cluster, so the face-row inventory
                # alone is not the supply)
                inv_pool = float(np.abs(
                    inv_in[bath_cluster_domain]).sum())
                out_mol = float(np.clip(-ex.boundary_net, 0.0, None).sum())
                exchange_metrics.update({
                    "n_boundary_coupled": float(bnd_coupled.sum()),
                    "boundary_net_mol_max": float(
                        np.abs(ex.boundary_net).max()),
                    "boundary_supply_ratio": (out_mol / inv_pool
                                              if inv_pool > 0.0 else 0.0),
                })
        else:
            inv_eff = inv_in

        # ---- RT-W2b GEM-call economy: pick the domains to equilibrate -------
        # A skipped domain is ELEMENT-EXACT: its releases enter its inventory
        # unchanged and its assemblage stays frozen this step — the only
        # approximation is the timing of the phase-assemblage update
        # (PRD 4.6.2). Salt-releasing domains are always forced: their
        # dissolution is solubility-controlled and booking the whole
        # accessible carrier into solution without an equilibrium would
        # bypass the solubility product (E3b).
        equilibrate: Optional[np.ndarray] = None
        released_elem: Optional[np.ndarray] = None
        eq_inv_in = eq_w_in = eq_age_in = None
        if self._economy and dom_to_cl is not None:
            dcfg = self._domains_cfg
            if not eq_rows_ok:
                eq_inv_in = np.zeros((n_clusters, len(ELEMENT_IDS)))
                eq_w_in = np.zeros(n_clusters)
                eq_age_in = np.full(n_clusters, -1, dtype=np.int64)
            elif state.domain_eq_inventory.shape[0] == n_clusters:
                eq_inv_in = state.domain_eq_inventory
                eq_w_in = state.domain_eq_water
                eq_age_in = state.domain_eq_age
            elif state.domain_eq_inventory.shape[0] == 0:
                eq_inv_in = np.zeros((n_clusters, len(ELEMENT_IDS)))
                eq_w_in = np.zeros(n_clusters)
                eq_age_in = np.full(n_clusters, -1, dtype=np.int64)
            else:
                raise RuntimeError(
                    f"economy snapshots have "
                    f"{state.domain_eq_inventory.shape[0]} rows but "
                    f"{n_clusters} domains were labeled - state is "
                    f"corrupted (no fallback)")
            released_elem = released @ self._kin_elements
            salt_rel = released[:, self._salt_channels].sum(axis=1) > 0.0
            valid = eq_age_in >= 0
            drift = (np.abs(inv_eff - eq_inv_in).sum(axis=1)
                     + np.abs(released_elem).sum(axis=1))
            # normalize element drift by the domain's SOLVENT mols, not the
            # last equilibrated inventory: fresh water has a near-zero
            # inventory and an inventory-relative scale would explode there,
            # making every threshold unreachable. dirty_rtol therefore reads
            # as a molality-scale tolerance (mol solute drift per mol water).
            r = (drift / (1e-24 + water_mol_c)
                 + np.abs(water_mol_c - eq_w_in) / (1e-24 + eq_w_in))
            forced = (~valid) | salt_rel | (eq_age_in >= dcfg.eq_max_age_steps)
            if bnd_coupled is not None:
                # bath-coupled domains re-equilibrate every step: their
                # assemblage sets the interface concentration that drives
                # the boundary flux (PRD 4.6.2/4.6.3)
                forced = forced | bnd_coupled
            dirty = valid & (r > dcfg.dirty_rtol)
            equilibrate = forced | dirty
            budget = dcfg.max_gem_calls_per_step
            if budget is not None:
                cand = np.flatnonzero(dirty & ~forced)
                if cand.size > budget:
                    # top-B by accumulated drift, id as the deterministic
                    # tie-break; deferred drift keeps growing, so priority
                    # is monotone and no domain starves
                    order = np.lexsort((cand, -r[cand]))
                    equilibrate = forced.copy()
                    equilibrate[cand[order[:budget]]] = True
            deferred = ~equilibrate
            exchange_metrics.update({
                "n_gem_selected": float(equilibrate.sum()),
                "n_deferred": float(deferred.sum()),
                "max_r_deferred": (float(r[deferred].max())
                                   if deferred.any() else 0.0),
                "max_eq_age": float(eq_age_in.max(initial=0)),
            })

        # ---- reaction phase (mode-dependent) --------------------------------
        # incremental (stoichiometric): parcels are NEW precipitates appended.
        # snapshot (gems3k, PRD v2.2): each cluster's owned hydrates + solution
        # inventory + released elements + free water are re-equilibrated as ONE
        # system; the result replaces the cluster's assemblage (re-dissolution,
        # CH consumption and phase rearrangement emerge from equilibrium).
        recon = np.where(labels >= 0, labels, site_cluster)
        snapshot = self.backend.mode == "snapshot"
        # cluster-scope views (v4.0/RT): placement overflow and the water
        # ledger stay hydraulically cluster-wide; domains partition solute
        # mixing and chemistry only. Mode-full aliases are bit-identical.
        if dom_to_cl is None or n_clusters == 0:
            # n_clusters == 0 (total dryout) must degrade gracefully in
            # mode C too: recon is all -1 and indexing the empty
            # dom_to_cl would raise (review finding)
            recon_cl = recon
            site_cluster_cl = site_cluster
            if dom_to_cl is not None:
                claimed_vol = np.zeros(n_cl)
                claimed_gas = np.zeros(n_cl)
        else:
            recon_cl = np.where(recon >= 0,
                                dom_to_cl[np.clip(recon, 0, None)],
                                np.int64(-1))
            site_cluster_cl = np.where(site_cluster >= 0,
                                       dom_to_cl[np.clip(site_cluster, 0,
                                                         None)],
                                       np.int64(-1))
            claimed_vol = np.zeros(n_cl)
            claimed_gas = np.zeros(n_cl)

        chem_mol_c = np.zeros(n_clusters)
        gel_mol_c = np.zeros(n_clusters)
        residual = np.zeros((n_clusters, len(ELEMENT_IDS)))
        parcel_env = np.zeros((n_clusters, n_h))   # NEW assemblage envelopes, vox
        parcel_mol = np.zeros((n_clusters, n_h))
        parcel_elem = np.zeros((n_clusters, n_h, len(ELEMENT_IDS)))
        parcel_em = np.zeros((n_clusters, self._n_em))   # E1 endmember mols
        injected_add = np.zeros(len(ELEMENT_IDS))
        cluster_ph: Dict[int, float] = {}
        parcel_rows: List[tuple] = []
        scale_c = np.ones(n_clusters)
        solved = np.zeros(n_clusters, dtype=bool)
        # Tier 0 (RT-P0a): per-domain aqueous speciation captured from the
        # SAME accepted responses that feed residual[c] — diagnostics only,
        # not persisted (P0b adds the frozen state array)
        spec_rows = (np.zeros((n_clusters, len(self.backend.aq_species_ids)))
                     if self._np_cfg is not None else None)

        def _envelope_vox(parcel) -> float:
            try:
                hi = h_index[parcel.phase_id]
            except KeyError:
                raise RuntimeError(
                    f"backend produced unknown phase {parcel.phase_id!r}; declared "
                    f"channels: {self.hydrate_ids}") from None
            return (parcel.skel_vol_cm3 / trial.vox_cm3) / (1.0 - gel_eps[hi])

        # cluster ownership of the spatial hydrates (Scheme S, PRD 4.5): the
        # recon partition splits each channel's dense volume; element/mol pools
        # are shared proportionally. The dry bin (recon == -1) keeps the exact
        # remainder untouched — dry hydrates are chemically frozen this step.
        # v4.0/RT mode B (PRD 4.6.1): offered fraction per channel for THIS
        # attempted dt — a halved retry offers half as much (f is a pure
        # function of (config, dt), so restart determinism is free). None =
        # every channel fully offered; the legacy path then uses the owned
        # arrays as-is (aliases, zero extra arithmetic — bit-identical).
        f_ch: Optional[np.ndarray] = None
        if self._tau_ch is not None:
            f = np.ones(n_h)
            m = self._tau_ch > 0.0
            f[m] = np.minimum(1.0, dt_h / self._tau_ch[m])
            if np.any(f < 1.0):
                f_ch = f

        sorb_sites_mol = None
        sorb_cov_frac = 0.0
        own_vol = np.zeros((n_clusters, n_h))
        owned_elem = np.zeros((n_clusters, n_h, len(ELEMENT_IDS)))
        owned_mol = np.zeros((n_clusters, n_h))
        owned_em = np.zeros((n_clusters, self._n_em))
        if f_ch is None:
            offered_vol, offered_mol = own_vol, owned_mol
            withheld_mol = pool_fed = None
        else:
            offered_vol = np.zeros((n_clusters, n_h))
            offered_mol = np.zeros((n_clusters, n_h))
            withheld_mol = np.zeros((n_clusters, n_h))
            pool_fed = np.zeros((n_clusters, self._n_em))
        if snapshot and n_clusters > 0:
            recon_b = (recon + 1).ravel()
            for h in range(n_h):
                dense = trial.hydrate_fraction[h]
                tot = float(dense.sum())
                if tot <= 0.0:
                    continue
                b = np.bincount(recon_b, weights=dense.ravel(),
                                minlength=n_clusters + 1)
                own_vol[:, h] = b[1:]
                share = b[1:] / tot
                owned_mol[:, h] = share * trial.hydrate_mol[h]
                if f_ch is not None:
                    # mode B: only the aged fraction enters the transaction;
                    # the withheld remainder stays in the dense field and the
                    # global ledgers, chemically frozen this step
                    offered_vol[:, h] = own_vol[:, h] * f_ch[h]
                    offered_mol[:, h] = owned_mol[:, h] * f_ch[h]
                    withheld_mol[:, h] = owned_mol[:, h] - offered_mol[:, h]
                # E2 (PRD 4.5 v3.0): the AMOUNT stays volume-share derived,
                # but the composition comes from the cluster's OWN pool — for
                # exactly the COVERED portion, the mass the pool actually
                # remembers. The uncovered remainder (rewetted or dry-region
                # holdings the pool never saw, or a whole pool-less cluster)
                # is fed at the global channel ratio (E1 behavior), so the
                # per-endmember withdrawal is bounded by pool content plus a
                # holdings-proportional term and can never invent composition
                # at the full volume-share scale (E2 review finding: an
                # unbounded pool-ratio feed could drive endmember_mol
                # negative while every closure identity stayed green). Both
                # ratio sources are clipped to the simplex: pools/holdings
                # are physically non-negative, and dividing a MIXED-SIGN dust
                # vector by its cancelled sum would amplify dust by up to
                # 1/eps into the backend feed (review finding).
                sl = self._em_slice[self.hydrate_ids[h]]
                g_pos = np.clip(trial.endmember_mol[sl], 0.0, None)
                gp = float(g_pos.sum())
                g_ratio = (g_pos / gp if gp > 0.0
                           else np.zeros(sl.stop - sl.start))
                pool = np.clip(em_pool_in[:, sl], 0.0, None)
                psum = pool.sum(axis=1)
                amounts = offered_mol[:, h]
                if f_ch is None:
                    covered = np.minimum(psum, np.clip(amounts, 0.0, None))
                else:
                    # mode B: the offered feed draws only the AGED FRACTION
                    # of the pool memory — f x the pool-covered portion of
                    # the holdings. min(psum, offered) would let an under-
                    # covering pool be consumed whole, destroying the stored
                    # composition at up to twice the configured exchange
                    # rate (review finding)
                    covered = f_ch[h] * np.minimum(
                        psum, np.clip(owned_mol[:, h], 0.0, None))
                safe = np.where(psum > 0.0, psum, 1.0)
                pool_part = pool * (covered / safe)[:, None]
                owned_em[:, sl] = (pool_part
                                   + (amounts - covered)[:, None]
                                   * g_ratio[None, :])
                if f_ch is not None:
                    pool_fed[:, sl] = pool_part
                # fed elements follow the fed COMPOSITION exactly, so the
                # global element/endmember ledgers stay mutually consistent
                # by construction ("subtract what was fed")
                owned_elem[:, h, :] = owned_em[:, sl] @ trial.endmember_elements[sl]
                if (self._sorb_density is not None
                        and self.hydrate_ids[h] == "CSHQ"):
                    # RT-S1b sorbent sites: the SAME covered-pool +
                    # global-fallback endmember split as the feed, but over
                    # the FULL owned amount - surfaces exist regardless of
                    # the mode-B aging fraction (endmember-preserving per
                    # the S1 plan decision)
                    amounts_full = np.clip(owned_mol[:, h], 0.0, None)
                    covered_full = np.minimum(psum, amounts_full)
                    site_em = (pool * (covered_full / safe)[:, None]
                               + (amounts_full - covered_full)[:, None]
                               * g_ratio[None, :])
                    sorb_sites_mol = site_em @ self._sorb_density
                    sorb_cov_frac = (float(covered_full.sum())
                                     / max(float(amounts_full.sum()), 1e-30))
        if snapshot and dom_to_cl is not None and n_clusters > 0:
            # Cross-domain overdraft guard (W4 measured: leaching's sharp
            # AFm OH/SO4 redistribution drove the summed per-endmember feed
            # past the global holdings -> balance_endmember_negative abort,
            # the residual risk the E2 review named). If the summed feed
            # would overdraw an endmember, scale the offending domains'
            # WHOLE-CHANNEL feed by one scalar (mol, volume, endmembers and
            # elements together - every closure identity stays consistent)
            # in domain-id order, deterministic. Mode full is untouched;
            # the no-overdraw fast path does no arithmetic.
            avail = np.clip(trial.endmember_mol, 0.0, None)
            # only domains that will actually feed the backend claim the
            # budget: economy-DEFERRED domains never withdraw, and letting
            # them claim first starved the reacting face (review finding)
            act_rows = (np.ones(n_clusters, dtype=bool)
                        if equilibrate is None else equilibrate)
            fed_tot = np.where(act_rows[:, None],
                               np.clip(owned_em, 0.0, None),
                               0.0).sum(axis=0)
            if np.any(fed_tot > avail + 1e-30):
                if offered_vol is own_vol:
                    offered_vol = own_vol.copy()
                    offered_mol = owned_mol.copy()
                bad_ch = []
                for h in range(n_h):
                    sl = self._em_slice[self.hydrate_ids[h]]
                    if np.any(fed_tot[sl] > avail[sl] + 1e-30):
                        bad_ch.append(h)
                remaining = avail.copy()
                clamped = 0.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    for c in np.flatnonzero(act_rows):
                        for h in bad_ch:
                            sl = self._em_slice[self.hydrate_ids[h]]
                            need = np.clip(owned_em[c, sl], 0.0, None)
                            if not need.any():
                                continue
                            ratios = np.where(need > 0.0,
                                              remaining[sl] / np.where(
                                                  need > 0.0, need, 1.0),
                                              np.inf)
                            k = float(min(1.0, max(0.0, ratios.min())))
                            if k < 1.0:
                                clamped += float(need.sum() * (1.0 - k))
                                owned_em[c, sl] *= k
                                owned_elem[c, h, :] = (
                                    owned_em[c, sl]
                                    @ trial.endmember_elements[sl])
                                offered_mol[c, h] *= k
                                offered_vol[c, h] *= k
                                if f_ch is not None:
                                    # keep mode B archive bookkeeping on
                                    # the ACTUAL feed (review finding: the
                                    # stale values silently destroyed pool
                                    # memory under B+C)
                                    pool_fed[c, sl] *= k
                                    withheld_mol[c, h] = (
                                        owned_mol[c, h] - offered_mol[c, h])
                            remaining[sl] -= np.clip(owned_em[c, sl], 0.0,
                                                     None)
                if clamped > 0.0:
                    exchange_metrics["endmember_feed_clamped_mol"] = clamped
        owned_gel_c = (offered_vol * gel_eps).sum(axis=1)

        # ---- S stage (RT-S1b, Lie T->S->R): equilibrium surface sorption -
        # per wet reactor: offer = pore solution + previous sorbed
        # (snapshot re-offer, so shrinking sites desorb automatically);
        # the SAME delta array moves mass between inv_eff and the sorbed
        # store (balance_sorption is same-float by construction).
        sorption_bal = None
        sorb_new = None
        if self._sorb_op is not None and n_clusters > 0:
            if not snapshot:
                raise RuntimeError(
                    "sorption needs the snapshot backend (sites come from "
                    "the owned endmember split)")
            sites = (sorb_sites_mol if sorb_sites_mol is not None
                     else np.zeros(n_clusters))
            sorb_new = np.zeros_like(sorb_in)
            # S-stage wetness contract: a reactor is an aqueous phase only
            # when water dominates the solutes (mol ratio >= 10; real pore
            # solutions sit above ~50, remap/fold float dust sits near 1).
            # Below that there is nothing to sorb FROM - the store freezes
            # like the water<=0 branch, and the count is REPORTED, not
            # swallowed (measured: a 5.6e-17 mol water / 7.3e-17 mol S
            # dust cluster scaled to a ~7e4 mol/kg "solution" and broke
            # IPhreeqc's A(H2O) convergence).
            n_sorb_dry = 0
            # dust clause: an offer whose solute total sits at the ledger's
            # float-noise floor (same style as the overdraw guard below)
            # is not a chemical system either - water-rich micro-reactors
            # with noise-level contents broke the operator's exact oxide
            # decomposition (measured: O deficit -4e-16 at noise scale).
            inv_scale = float(np.abs(inv_eff).max()) if inv_eff.size else 0.0
            dust_floor = 1e-24 + 1e-12 * inv_scale
            for c in range(n_clusters):
                offer = inv_eff[c] + sorb_in[c]
                if sorption_reactor_dry(float(water_mol_c[c]), offer,
                                        dust_floor):
                    sorb_new[c] = sorb_in[c]     # dry reactor: frozen store
                    n_sorb_dry += int(water_mol_c[c] > 0.0)
                    continue
                res = self._sorb_op.sorb(offer, float(water_mol_c[c]),
                                         float(sites[c]))
                sorb_new[c] = res.sorbed_mol
            s_delta = sorb_new - sorb_in
            inv_eff = inv_eff - s_delta
            neg = inv_eff < 0.0
            if np.any(neg):
                worst = float(inv_eff[neg].min())
                scale = float(np.abs(inv_eff).max())
                if -worst > 1e-24 + 1e-12 * scale:
                    raise RuntimeError(
                        f"sorption overdrew the pore solution by {worst:.3e}"
                        f" mol - operator/config inconsistency (no clip)")
                inv_eff = np.where(neg, 0.0, inv_eff)
            sorption_bal = ledger.SorptionBalance(
                applied_inventory_delta=(-s_delta).sum(axis=0),
                applied_sorbed_delta=s_delta.sum(axis=0),
                abs_scale=float(np.abs(s_delta).sum()))
            exchange_metrics["sorbed_total_mol"] = float(
                np.abs(sorb_new).sum())
            exchange_metrics["sorbed_delta_mol"] = float(
                np.abs(s_delta).sum())
            exchange_metrics["sorption_sites_mol"] = float(sites.sum())
            exchange_metrics["sorption_pool_coverage"] = sorb_cov_frac
            exchange_metrics["sorption_dry_reactors"] = n_sorb_dry

        total_water_mol = float(water_mol_c.sum())
        for c in range(n_clusters):
            rel = {p: float(released[c, k]) for k, p in enumerate(KINETIC_PHASE_IDS)
                   if released[c, k] > 0.0}
            if equilibrate is not None and not equilibrate[c]:
                # economy deferral: element-exact, assemblage frozen, no
                # water consumed; supersaturation accumulates in the
                # inventory until the dirty trigger or age bound fires
                residual[c] = inv_eff[c] + released_elem[c]
                continue
            has_solids = snapshot and offered_vol[c].sum() > 0.0
            has_inventory = bool(np.any(inv_eff[c] != 0.0))
            if not rel and not has_solids and not has_inventory:
                residual[c] = inv_eff[c]
                continue
            water_c = float(water_mol_c[c])
            inv_flushed = False
            if dom_to_cl is None:
                trace_water = water_c < 1e-3 * total_water_mol
            else:
                # mode C rescope (PRD 4.6.2): against the MEAN wet-domain
                # water, not the global total — the literal rule would
                # classify every domain as trace at fine partitions and
                # stall hydration globally
                n_wet = int((water_mol_c > 0.0).sum())
                trace_water = water_c < 1e-3 * (total_water_mol
                                                / max(n_wet, 1))
            # a nearly-dry pocket fails identically at every release scale —
            # give it exactly one attempt instead of burning retries every step
            max_transient = 0 if trace_water else 4
            s = 1.0
            result = None
            transient_failures = 0
            failure_reason: Optional[str] = None
            solid_elem_c = owned_elem[c].sum(axis=0) if has_solids else None
            for _ in range(60):
                scaled = {p: v * s for p, v in rel.items()}
                try:
                    if snapshot:
                        result = self.backend.react(scaled, water_c, inv_eff[c],
                                                    solid_elements=solid_elem_c)
                    else:
                        result = self.backend.react(scaled, water_c, inv_eff[c])
                except backend_mod.BackendTransientError:
                    if (bath_cluster_domain is not None
                            and bath_cluster_domain[c]
                            and not inv_flushed
                            and np.any(inv_eff[c] != 0.0)
                            and float(np.abs(inv_eff[c]).sum())
                            < 1e-4 * water_c):
                        # W4 measured: drained-solution inventories (traces
                        # to ~1/100 of equilibrium, ratio-distorted) sit on
                        # a STOCHASTIC GEM/AIA knife edge, while the same
                        # call with the inventory exactly zero converges.
                        # The renewed bath takes the remainder: flush the
                        # domain's solutes to the boundary ledger (exact
                        # floats - closure holds) and retry once before
                        # the scale-down path.
                        trial.boundary_exchanged_elements = (
                            trial.boundary_exchanged_elements - inv_eff[c])
                        inv_eff[c] = 0.0
                        inv_flushed = True
                        exchange_metrics["bath_flushed_domains"] = (
                            exchange_metrics.get("bath_flushed_domains",
                                                 0.0) + 1.0)
                        continue
                    # nonconvergence may be release-size dependent — scale down a
                    # few times before giving up on this cluster
                    transient_failures += 1
                    if transient_failures > max_transient:
                        failure_reason = REJECT_BACKEND_FAILURE
                        break
                    s *= 0.5
                    continue
                if result.status == backend_mod.STATUS_OK:
                    gel_new = sum(_envelope_vox(pc) * gel_eps[h_index[pc.phase_id]]
                                  for pc in result.parcels)
                    gel_owned = owned_gel_c[c] if snapshot else 0.0
                    need = result.water_consumed_mol + (gel_new - gel_owned) / vm_w
                    if need <= water_c:
                        failure_reason = None
                        break
                    s *= min(0.5, water_c * (1.0 - 1e-9) / max(need, 1e-300))
                elif result.status == backend_mod.STATUS_INSUFFICIENT_WATER:
                    s *= 0.5
                else:
                    return None, StepReject(result.status), {}
            else:
                failure_reason = backend_mod.STATUS_INSUFFICIENT_WATER
            if failure_reason is not None:
                # A nearly-dry pocket (e.g. dissolved inventory outweighing its
                # trace water) can never equilibrate at ANY release scale. Its
                # release goes back to the solid as honest unmet, its inventory
                # and (under snapshot) its owned hydrates stay frozen in place;
                # a materially wet cluster failing this way rejects the trial.
                if trace_water:
                    scale_c[c] = 0.0
                    residual[c] = inv_eff[c]
                    continue
                if (dom_to_cl is not None and failure_reason
                        == backend_mod.STATUS_INSUFFICIENT_WATER):
                    # mode C (PRD 4.6.2): a DOMAIN whose equilibrium needs
                    # more water than the domain holds at every release
                    # scale is a local water-starved pocket - the snapshot
                    # delta is dt-independent, so rejecting could only
                    # abort the run (the space-filling argument). Full mode
                    # keeps the hard reject: there the whole cluster's
                    # water pooled, and failing it signals a defect.
                    exchange_metrics["water_frozen_domains"] = (
                        exchange_metrics.get("water_frozen_domains", 0.0)
                        + 1.0)
                    scale_c[c] = 0.0
                    residual[c] = inv_eff[c]
                    continue
                if (dom_to_cl is not None
                        and failure_reason == REJECT_BACKEND_FAILURE):
                    # RT-W3/W4 (PRD 4.6.3): a PARTITIONED domain can reach
                    # local compositions the pooled cluster never sees -
                    # bath-drained solutions (assemblage + near-pure water
                    # + dust) and late-age water-starved bands both sit on
                    # genuine GEM/AIA knife edges that no release scale or
                    # dt fixes (measured: a 1e-8 relative perturbation
                    # flips convergence). Freeze the domain exactly like
                    # the space-filling impossibility (release -> unmet,
                    # assemblage and inventory kept); later steps
                    # re-regularize it. FULL mode keeps the hard reject -
                    # there the failure signals a defect. Visible via
                    # nonconv_frozen_domains.
                    exchange_metrics["nonconv_frozen_domains"] = (
                        exchange_metrics.get("nonconv_frozen_domains", 0.0)
                        + 1.0)
                    scale_c[c] = 0.0
                    residual[c] = inv_eff[c]
                    continue
                return None, StepReject(failure_reason), {}
            if snapshot:
                # space-filling limit: a pocket whose equilibrium assemblage
                # wants more envelope growth than the pocket's entire pore
                # space (liquid + this step's vacated + its own re-dissolved
                # volume) can never place it — the snapshot delta is not
                # dt-scaled, so rejecting would abort the run. Freeze the
                # cluster exactly like a trace-water pocket: release returns
                # to the solid as unmet, assemblage and inventory stay.
                env_new = np.zeros(n_h)
                for pc in result.parcels:
                    env_new[h_index[pc.phase_id]] += _envelope_vox(pc)
                delta_c = env_new - offered_vol[c]
                member_c = recon == c
                grow_c = float(np.clip(delta_c, 0.0, None).sum())
                vac_tot_c = (s * float(np.where(member_c, vacated, 0.0).sum())
                             + float(np.clip(-delta_c, 0.0, None).sum()))
                if dom_to_cl is None:
                    cap_c = (float(np.where(member_c, trial.capillary_liquid,
                                            0.0).sum())
                             + vac_tot_c)
                else:
                    # mode C sequential capacity claim (PRD 4.6.2): growth
                    # places domain-locally but overflow resolves cluster-
                    # wide, so each solved domain claims against the CLUSTER
                    # capacity minus earlier claims (domain-id order —
                    # deterministic; one domain per cluster reduces to the
                    # legacy gate exactly)
                    cl_c = int(dom_to_cl[c])
                    cap_c = (float(np.where(recon_cl == cl_c,
                                            trial.capillary_liquid,
                                            0.0).sum())
                             + vac_tot_c - claimed_vol[cl_c])
                # 1e-9 relative margin: the placement layers recompute this
                # capacity through different float chains (give-back, remove
                # clamps, per-voxel mutation) — a near-exact fit must freeze
                # rather than gamble on ulp agreement, because a capacity
                # reject here is dt-independent and would abort the run
                # (rev.2 review finding)
                freeze = grow_c > cap_c * (1.0 - 1e-9)
                if not freeze:
                    # water-relocation limit (rev.2 review finding 5, now
                    # REPRODUCED): growth beyond this step's vacancies
                    # displaces liquid, and the displaced volume that
                    # chemistry does not consume must fit the cluster's own
                    # gas space — reconciliation deficit is exactly
                    # grow - vacancies - (chem + gel net) water volume, and a
                    # gas-free saturated pocket rejects balance_water at
                    # EVERY dt (snapshot deltas are not dt-scaled). Freeze
                    # instead: a pocket with nowhere to push its water stops
                    # hydrating, which is the same space-filling physics.
                    if dom_to_cl is None:
                        gas_c_vol = float(np.where(member_c,
                                                   trial.capillary_gas,
                                                   0.0).sum())
                    else:
                        cl_c = int(dom_to_cl[c])
                        gas_c_vol = (float(np.where(recon_cl == cl_c,
                                                    trial.capillary_gas,
                                                    0.0).sum())
                                     - claimed_gas[cl_c])
                    water_out_vol = (result.water_consumed_mol * vm_w
                                     + float((env_new * gel_eps).sum())
                                     - owned_gel_c[c])
                    deficit_est = grow_c - vac_tot_c - water_out_vol
                    freeze = deficit_est > gas_c_vol - 1e-12 * max(1.0, gas_c_vol)
                if freeze:
                    scale_c[c] = 0.0
                    residual[c] = inv_eff[c]
                    continue
                if dom_to_cl is not None:
                    cl_c = int(dom_to_cl[c])
                    claimed_vol[cl_c] += max(grow_c - vac_tot_c, 0.0)
                    claimed_gas[cl_c] += max(deficit_est, 0.0)
            solved[c] = True
            scale_c[c] = s
            chem_mol_c[c] = result.water_consumed_mol
            residual[c] = result.residual_inventory
            if spec_rows is not None and result.aqueous_species_mol:
                row = spec_rows[c]
                idx = self._np_dc_index
                for dc, m in result.aqueous_species_mol.items():
                    k = idx.get(dc)
                    if k is not None:      # solvent H2O@ is not transported
                        row[k] = m
            if (bath_active and bath_cluster_domain is not None
                    and bath_cluster_domain[c]
                    and np.any(result.injected_elements != 0.0)):
                # W4.1 measured (PRD 4.6.3): supply-limited drained front
                # domains re-earn the worker's redox seed on EVERY call; at
                # fine dt the per-call dust grows linearly with step count
                # and trips the injected cap (~1e-19 mol/call x 8640 steps,
                # abort at t=168.99 h) while sitting 13 decades below any
                # material scale. The anchor a bath-coupled domain receives
                # is PHYSICALLY the aerated bath's dissolved O2 - book it as
                # boundary influx (exact floats, closure identity has both
                # terms additive), keeping the injected gate's strict sealed
                # meaning: solver-fabricated mass, not bath re-supply.
                trial.boundary_exchanged_elements = (
                    trial.boundary_exchanged_elements
                    + result.injected_elements)
                exchange_metrics["bath_anchor_mol"] = (
                    exchange_metrics.get("bath_anchor_mol", 0.0)
                    + float(np.abs(result.injected_elements).sum()))
            else:
                injected_add += result.injected_elements
            if result.ph_status == "ok":
                cluster_ph[c] = result.ph
            for pc in result.parcels:
                hi = h_index[pc.phase_id]
                env = _envelope_vox(pc)
                parcel_env[c, hi] += env
                parcel_mol[c, hi] += pc.mol
                parcel_elem[c, hi] += pc.elements
                # E1: book the parcel's endmember split; None = single-
                # endmember phase (the channel is the endmember); an unknown
                # endmember name is a protocol violation, never dropped
                if pc.endmember_mol is None:
                    key = (pc.phase_id, pc.phase_id)
                    if key not in self._em_index:
                        # bundles name single-DC species differently from the
                        # phase (PC: Calcite->Cal etc.) and solid solutions
                        # have no self-named endmember — a None split there is
                        # a backend protocol violation, not bookable
                        raise RuntimeError(
                            f"backend omitted the endmember split for phase "
                            f"{pc.phase_id!r} whose endmember universe is "
                            f"{self._hydrate_endmembers[pc.phase_id]}")
                    parcel_em[c, self._em_index[key]] += pc.mol
                else:
                    for dc, m in pc.endmember_mol.items():
                        key = (pc.phase_id, dc)
                        if key not in self._em_index:
                            raise RuntimeError(
                                f"backend produced unknown endmember {dc!r} "
                                f"for phase {pc.phase_id!r}")
                        parcel_em[c, self._em_index[key]] += m
                parcel_rows.append((float(trial.time_h + dt_h), int(c), pc.phase_id,
                                    float(pc.mol),
                                    float(pc.skel_vol_cm3 / trial.vox_cm3), float(env)))
            gel_new = float((parcel_env[c] * gel_eps).sum())
            gel_mol_c[c] = (gel_new - (owned_gel_c[c] if snapshot else 0.0)) / vm_w

        # give scaled-back dissolution volume back to the solid and record it unmet
        if np.any(scale_c < 1.0):
            for c in np.flatnonzero(scale_c < 1.0):
                mask = site_mask & (site_cluster == c)
                for k, p in enumerate(KINETIC_PHASE_IDS):
                    give_back = dis.removed_vol[k] * np.where(mask, 1.0 - scale_c[c], 0.0)
                    chan = SOLID_PHASE_IDS.index(p)
                    trial.anhydrous_fraction[chan] += give_back
                    dis.removed_vol[k] -= give_back
            for k, p in enumerate(KINETIC_PHASE_IDS):
                vm = trial.vm_vox(reg, p)
                new_removed = float(dis.removed_vol[k].sum()) / vm
                dis.unmet_mol[k] += dis.removed_mol[k] - new_removed
                dis.removed_mol[k] = new_removed
            vacated = dis.removed_vol.sum(axis=0)
            site_mask = vacated > 0.0

        # ---- Tier 0 diagnostics (RT-P0a, PRD 4.6.4): NP effective
        # diffusivities from THIS step's accepted speciation, report-only.
        # Masked to edges whose BOTH endpoints equilibrated this step (other
        # rows are zeros here — the frozen state array is a P0b concern).
        if (spec_rows is not None and self._np_cfg.diagnostics_only
                and dom_to_cl is not None):
            npc = transport.np_effective_conductance(
                graph, spec_rows, graph.water, self._np_dw_vox2_h,
                self._np_z, self._np_nu)
            both = solved[graph.edge_a] & solved[graph.edge_b]
            exchange_metrics["np_edges_measured"] = float(both.sum())
            exchange_metrics["np_edges_stale"] = float((~both).sum())
            if self._np_cfg.phi_clamp_report:
                for key, v in npc.counts.items():
                    exchange_metrics[key] = float(v)
                exchange_metrics["np_charge_flux_rel_max"] = (
                    npc.charge_flux_rel_max)
            if both.any() and self._d0_vox2_h:
                ratio = npc.deff_edge[both] / self._d0_vox2_h
                ratio = ratio[ratio > 0.0]
                if ratio.size:
                    qs = np.percentile(ratio, (25.0, 50.0, 75.0))
                    exchange_metrics.update({
                        "np_deff_ratio_min": float(ratio.min()),
                        "np_deff_ratio_p25": float(qs[0]),
                        "np_deff_ratio_p50": float(qs[1]),
                        "np_deff_ratio_p75": float(qs[2]),
                        "np_deff_ratio_max": float(ratio.max()),
                    })

        # ---- spatial application -------------------------------------------
        backend_removal_vol = 0.0
        removed_total = 0.0
        if snapshot:
            # deltas vs owned volume; NEGATIVES FIRST — re-dissolved hydrate
            # frees pore capacity before growth is placed (v1 pattern)
            delta_env = np.where(solved[:, None], parcel_env - offered_vol, 0.0)
            freed = np.zeros_like(vacated)
            for c in np.flatnonzero(solved):
                member = recon == c
                for h in np.flatnonzero(delta_env[c] < 0.0):
                    request = -float(delta_env[c, h])
                    backend_removal_vol += request
                    removal_field, removed = morphology.remove(
                        trial.hydrate_fraction, h, request, member)
                    freed += removal_field
                    removed_total += removed
            vacated = vacated + freed
            site_mask = vacated > 0.0

            demand = np.zeros_like(trial.hydrate_fraction)
            for c in np.flatnonzero(solved):
                pos = np.flatnonzero(delta_env[c] > 0.0)
                if pos.size == 0:
                    continue
                member = recon == c
                # growth anchor cascade: freed/vacated space -> existing hydrate
                # surfaces -> pore liquid (deterministic, same-cluster only)
                w = np.where(member, vacated, 0.0)
                if float(w.sum()) <= 0.0:
                    w = np.where(member, trial.hydrate_fraction.sum(axis=0), 0.0)
                if float(w.sum()) <= 0.0:
                    w = np.where(member, trial.capillary_liquid, 0.0)
                wsum = float(w.sum())
                if wsum <= 0.0:
                    return None, StepReject(morphology.STATUS_CAPACITY), {}
                frac = w / wsum
                for h in pos:
                    demand[h] += delta_env[c, h] * frac
            requested_vol = float(demand.sum())
            backend_bulk_vol = float(np.clip(delta_env, 0.0, None).sum())
        else:
            # incremental: demand distributed over the cluster's dissolution
            # sites in proportion to locally dissolved volume
            demand = np.zeros_like(trial.hydrate_fraction)
            for c in np.flatnonzero(parcel_env.any(axis=1)):
                mask = site_mask & (site_cluster == c)
                wsum = float(vacated[mask].sum())
                if wsum <= 0.0:
                    return None, StepReject(REJECT_BACKEND_FAILURE), {}
                frac = np.where(mask, vacated, 0.0) / wsum
                for h in np.flatnonzero(parcel_env[c] > 0.0):
                    demand[h] += parcel_env[c, h] * frac
            requested_vol = float(demand.sum())
            backend_bulk_vol = float(parcel_env.sum())

        # total water feasibility (chemical + gel, both signed under snapshot)
        if np.any(chem_mol_c + gel_mol_c > water_mol_c + 1e-30):
            return None, StepReject(backend_mod.STATUS_INSUFFICIENT_WATER), {}

        outcome = morphology.place(trial.hydrate_fraction, trial.capillary_liquid,
                                   vacated, demand, recon_cl,
                                   recon_cl if snapshot else site_cluster_cl)
        if outcome.status != morphology.STATUS_OK:
            return None, StepReject(outcome.status), {}
        # water flows into leftover vacated/freed space (same connected liquid)
        trial.capillary_liquid += vacated

        # cluster water reconciliation: excess liquid volume becomes capillary gas.
        # The reference volume uses the same recon partition as the current volume
        # so sub-threshold liquid at site voxels cancels on both sides.
        if dom_to_cl is None:
            chem_cl, gel_cl = chem_mol_c, gel_mol_c
        else:
            # water is hydraulically cluster-wide (PRD 4.6.2): aggregate the
            # per-domain consumption to the parent clusters
            chem_cl = np.bincount(dom_to_cl, weights=chem_mol_c,
                                  minlength=n_cl)
            gel_cl = np.bincount(dom_to_cl, weights=gel_mol_c,
                                 minlength=n_cl)
        cur_vol_c = np.bincount(recon_cl[recon_cl >= 0],
                                weights=trial.capillary_liquid[recon_cl >= 0],
                                minlength=n_cl)
        prev_vol_recon_c = np.bincount(recon_cl[recon_cl >= 0],
                                       weights=prev_liquid[recon_cl >= 0],
                                       minlength=n_cl)
        target_vol_c = prev_vol_recon_c - (chem_cl + gel_cl) * vm_w
        diff_c = cur_vol_c - target_vol_c
        if n_cl > 0:
            # negative diff: water returned to solution (re-dissolution, solute
            # water re-emerging as solvent) refills capillary liquid from the
            # cluster's own gas space; without enough local gas the returned
            # volume has nowhere to appear
            gas_vol_c = np.bincount(recon_cl[recon_cl >= 0],
                                    weights=trial.capillary_gas[recon_cl >= 0],
                                    minlength=n_cl)
            deficit = np.clip(-diff_c, 0.0, None)
            tol_c = 1e-12 * (1.0 + np.abs(target_vol_c))
            if np.any((deficit > tol_c) & (deficit > gas_vol_c + tol_c)):
                return None, StepReject("balance_water"), {}
            with np.errstate(invalid="ignore", divide="ignore"):
                refill = np.where((deficit > tol_c) & (gas_vol_c > 0.0),
                                  deficit / gas_vol_c, 0.0)
                factor = np.where(cur_vol_c > 0.0,
                                  np.clip(diff_c, 0.0, None) / cur_vol_c, 0.0)
            refill_vox = np.where(recon_cl >= 0,
                                  refill[np.clip(recon_cl, 0, None)], 0.0)
            added_liq = trial.capillary_gas * refill_vox
            trial.capillary_gas -= added_liq
            trial.capillary_liquid += added_liq
            fac_vox = np.where(recon_cl >= 0,
                               factor[np.clip(recon_cl, 0, None)], 0.0)
            removed_liq = trial.capillary_liquid * fac_vox
            trial.capillary_liquid -= removed_liq
            trial.capillary_gas += removed_liq

        # authoritative mol ledger (exact scalar arithmetic; signed under snapshot:
        # exactly what was fed to the backend is subtracted, exactly what came out
        # is added — closure cannot open a gap regardless of share float dust)
        chem_total = float(chem_mol_c.sum())
        gel_total = float(gel_mol_c.sum())
        trial.phase_mol = trial.phase_mol - dis.removed_mol
        trial.unmet_mol = trial.unmet_mol + dis.unmet_mol
        if snapshot:
            sv = solved
            trial.hydrate_mol = trial.hydrate_mol + (
                parcel_mol[sv].sum(axis=0) - offered_mol[sv].sum(axis=0))
            trial.hydrate_env_vol_vox = trial.hydrate_env_vol_vox + (
                parcel_env[sv].sum(axis=0) - offered_vol[sv].sum(axis=0))
            trial.hydrate_elements_ch = trial.hydrate_elements_ch + (
                parcel_elem[sv].sum(axis=0) - owned_elem[sv].sum(axis=0))
            trial.endmember_mol = trial.endmember_mol + (
                parcel_em[sv].sum(axis=0) - owned_em[sv].sum(axis=0))
        else:
            trial.hydrate_mol = trial.hydrate_mol + parcel_mol.sum(axis=0)
            trial.hydrate_env_vol_vox = (trial.hydrate_env_vol_vox
                                         + parcel_env.sum(axis=0))
            trial.hydrate_elements_ch = (trial.hydrate_elements_ch
                                         + parcel_elem.sum(axis=0))
            trial.endmember_mol = trial.endmember_mol + parcel_em.sum(axis=0)
        trial.injected_elements = trial.injected_elements + injected_add
        trial.water_free_mol -= chem_total + gel_total
        trial.water_gel_mol += gel_total
        trial.water_bound_mol += chem_total

        # RT-W2b: refresh the economy snapshots on the pre-remap indexing
        if self._economy and dom_to_cl is not None:
            eq_inv_new = eq_inv_in.copy()
            eq_w_new = eq_w_in.copy()
            eq_age_new = np.where(eq_age_in >= 0, eq_age_in + 1, eq_age_in)
            eq_inv_new[solved] = residual[solved]
            eq_w_new[solved] = water_mol_c[solved]
            eq_age_new[solved] = 0
            frozen = equilibrate & ~solved & (scale_c == 0.0)
            exchange_metrics["n_frozen_domains"] = float(frozen.sum())

        # relabel + conservative inventory remap
        new_cl_labels, n_new_cl = transport.label_clusters(
            trial.capillary_liquid, axes_now)
        if dom_to_cl is None:
            new_labels, n_new = new_cl_labels, n_new_cl
        else:
            new_labels, n_new, _ = transport.label_domains(
                new_cl_labels, n_new_cl,
                self._domains_cfg.tiles(trial.grid_size))

        def _fold_dry_rows(rows_prev, result_rows, dry):
            """v4.0/RT mode C (PRD 4.6.2 fold): a dry DOMAIN whose parent
            cluster still has wet overlap redistributes its row over the new
            domains overlapping that cluster's previous liquid, weighted by
            liquid overlap. A domain whose whole cluster died keeps the
            legacy semantics (inventory: reject; pool: drop)."""
            hard = []
            fold_rows = []
            liq_min = np.minimum(prev_liquid, trial.capillary_liquid)
            for pd in dry:
                pc = int(dom_to_cl[pd])
                m = (cl_labels == pc) & (new_labels >= 0)
                if not m.any():
                    hard.append(pd)
                    continue
                dist = np.bincount(new_labels[m], weights=liq_min[m],
                                   minlength=n_new)
                tot = float(dist.sum())
                if tot <= 0.0:
                    hard.append(pd)
                    continue
                frac = dist / tot
                result_rows += rows_prev[pd][None, :] * frac[:, None]
                for nd in np.flatnonzero(dist > 0.0):
                    fold_rows.append((int(pd), int(nd), float(dist[nd])))
            return hard, fold_rows

        remap = transport.remap_inventories(labels, prev_liquid, new_labels,
                                            trial.capillary_liquid, residual, n_new)
        fold_events: List[tuple] = []
        if remap.dryout:
            if dom_to_cl is None:
                return None, StepReject(REJECT_CLUSTER_DRYOUT), {}
            hard, fold_events = _fold_dry_rows(residual, remap.inventory,
                                               remap.dryout)
            if hard:
                surrendered = (_dryout_surrender(residual, hard, dom_to_cl,
                                                 cl_labels, prev_liquid)
                               if bath_active else None)
                if surrendered is None:
                    return None, StepReject(REJECT_CLUSTER_DRYOUT), {}
                for pd in surrendered:
                    trial.boundary_exchanged_elements = (
                        trial.boundary_exchanged_elements - residual[pd])
                exchange_metrics["dryout_surrendered_domains"] = (
                    exchange_metrics.get("dryout_surrendered_domains", 0.0)
                    + float(len(surrendered)))
                exchange_metrics["dryout_surrendered_mol"] = (
                    exchange_metrics.get("dryout_surrendered_mol", 0.0)
                    + float(sum(np.abs(residual[pd]).sum()
                                for pd in surrendered)))
        trial.cluster_inventory = remap.inventory
        # E2: pools follow the assemblage — a solved cluster's pool IS its own
        # parcels (absolute replacement, non-compounding); frozen clusters
        # keep theirs. Remapped with the same overlaps as the inventory;
        # zero-overlap rows drop (pools are compositional memory — the global
        # ledgers carry conservation, so a stranded pool is not a dryout).
        pools_prev = em_pool_in.copy()
        if snapshot and bool(solved.any()):
            if f_ch is None:
                pools_prev[solved] = parcel_em[solved]
            else:
                # mode B (PRD 4.6.1): solved rows start from this step's
                # parcels; each rate-limited channel adds its withheld
                # archive — the un-fed pool memory, clipped to the physical
                # simplex (stale negative dust dies here exactly as it died
                # under absolute replacement — review finding) and capped so
                # the archive never exceeds the withheld owned mass. f -> 0
                # turns the pool into a deposition ledger. Channels at
                # f == 1.0 keep the absolute-replacement semantics.
                blended = parcel_em.copy()
                for h in np.flatnonzero(f_ch < 1.0):
                    sl = self._em_slice[self.hydrate_ids[h]]
                    resid = np.clip(em_pool_in[:, sl] - pool_fed[:, sl],
                                    0.0, None)
                    rsum = resid.sum(axis=1)
                    withheld = np.clip(withheld_mol[:, h], 0.0, None)
                    cap = np.where(
                        rsum > 0.0,
                        np.minimum(1.0, withheld / np.where(rsum > 0.0,
                                                            rsum, 1.0)),
                        0.0)
                    blended[:, sl] += resid * cap[:, None]
                pools_prev[solved] = blended[solved]
        pool_remap = transport.remap_inventories(
            labels, prev_liquid, new_labels, trial.capillary_liquid,
            pools_prev, n_new)
        if dom_to_cl is not None and pool_remap.dryout:
            # pools fold with the same weights; a dead cluster's pool drops
            # (legacy semantics — conservation lives in the global ledgers)
            _, pool_folds = _fold_dry_rows(pools_prev, pool_remap.inventory,
                                           pool_remap.dryout)
            fold_events.extend(pool_folds)
        trial.cluster_endmember_mol = pool_remap.inventory
        if np_active:
            # frozen speciation: solved domains take THIS step's accepted
            # speciation, deferred domains keep their last one (domain_eq_age
            # semantics); then remap onto the new labels with the same
            # liquid-overlap weights. A COEFFICIENT CACHE, not a ledger:
            # dry folds keep rows usable but are NOT booked as remap events,
            # and hard-dry rows simply drop (conservation lives elsewhere).
            spec_base = spec_in.copy()
            if spec_rows is not None:
                spec_base[solved] = spec_rows[solved]
            spec_remap = transport.remap_inventories(
                labels, prev_liquid, new_labels, trial.capillary_liquid,
                spec_base, n_new)
            if dom_to_cl is not None and spec_remap.dryout:
                _fold_dry_rows(spec_base, spec_remap.inventory,
                               spec_remap.dryout)
            trial.domain_species_mol = spec_remap.inventory
        if sorb_new is not None:
            # the sorbed store is a LEDGER (unlike the speciation cache):
            # it remaps with the inventory weights, folds on domain dryout
            # with the same recorded events, and a hard dryout - sorbed
            # mass with no wet successor anywhere - rejects the step like
            # the solution inventory does (mass must not vanish)
            sorb_remap = transport.remap_inventories(
                labels, prev_liquid, new_labels, trial.capillary_liquid,
                sorb_new, n_new)
            sorb_folds = []
            if sorb_remap.dryout:
                if dom_to_cl is None:
                    return None, StepReject(REJECT_CLUSTER_DRYOUT), {}
                hard_s, sorb_folds = _fold_dry_rows(sorb_new,
                                                    sorb_remap.inventory,
                                                    sorb_remap.dryout)
                if hard_s:
                    return None, StepReject(REJECT_CLUSTER_DRYOUT), {}
            fold_events.extend(sorb_folds)
            trial.domain_sorbed_mol = sorb_remap.inventory
        # RT-W2b: economy snapshots survive relabeling only through pure
        # 1:1 transfers (one source, one target); merges, splits and folds
        # invalidate the record — merging equilibration snapshots is not
        # meaningful, and a forced re-equilibration is always safe
        if self._economy and dom_to_cl is not None:
            src_cnt = np.zeros(n_clusters, dtype=np.int64)
            tgt_cnt = np.zeros(n_new, dtype=np.int64)
            for pa, pb, _ov in remap.events:
                src_cnt[pa] += 1
                tgt_cnt[pb] += 1
            inv2 = np.zeros((n_new, len(ELEMENT_IDS)))
            w2 = np.zeros(n_new)
            age2 = np.full(n_new, -1, dtype=np.int64)
            for pa, pb, _ov in remap.events:
                if src_cnt[pa] == 1 and tgt_cnt[pb] == 1:
                    inv2[pb] = eq_inv_new[pa]
                    w2[pb] = eq_w_new[pa]
                    age2[pb] = eq_age_new[pa]
            for _pd, nd, _w in fold_events:
                age2[nd] = -1
            trial.domain_eq_inventory = inv2
            trial.domain_eq_water = w2
            trial.domain_eq_age = age2
        # cluster_id persists CLUSTER labels in every mode — the domain
        # partition is a derived function of the liquid field, never stored
        trial.cluster_id = new_cl_labels
        for prev_c, new_c, ov in (*remap.events, *fold_events):
            trial.remap_events["time_h"].append(trial.time_h + dt_h)
            trial.remap_events["prev"].append(prev_c)
            trial.remap_events["new"].append(new_c)
            trial.remap_events["overlap_vox"].append(ov)
        if dom_to_cl is not None and parcel_rows:
            # mode C: one aggregate row per channel per step instead of one
            # per (domain, channel) — at fine partitions per-domain rows
            # would grow the event table by orders of magnitude (PRD 2.3).
            # cluster = -1 marks the aggregate. Mode-C runs have no legacy
            # parcel goldens; deterministic via sorted channel order.
            agg: Dict[str, List[float]] = {}
            for t, _c, hname, mol, skel, env in parcel_rows:
                a = agg.setdefault(hname, [t, 0.0, 0.0, 0.0])
                a[1] += mol
                a[2] += skel
                a[3] += env
            parcel_rows = [(a[0], -1, hname, a[1], a[2], a[3])
                           for hname, a in sorted(agg.items())]
        for row in parcel_rows:
            for key, val in zip(("time_h", "cluster", "hydrate", "mol",
                                 "skel_vol_vox", "bulk_vol_vox"), row):
                trial.parcels[key].append(val)

        trial.time_h += dt_h

        placement = ledger.PlacementBalance(
            backend_bulk_vol_vox=backend_bulk_vol,
            requested_bulk_vol_vox=requested_vol,
            placed_bulk_vol_vox=outcome.placed_vol_vox,
            backend_removal_vol_vox=backend_removal_vol,
            removed_vol_vox=removed_total)
        partition_check = None
        if dom_to_cl is not None:
            partition_check = ledger.DomainPartition(
                labels=cl_labels, domain_id=labels,
                domain_to_cluster=dom_to_cl)
        report = ledger.check_all(trial, reg, placement,
                                  exchange=exchange_bal,
                                  partition=partition_check,
                                  sorption=sorption_bal)
        if not report.ok:
            return None, StepReject(report.violations[0]), report.metrics
        metrics = dict(report.metrics)
        metrics.update(exchange_metrics)
        if cluster_ph:
            metrics["cluster_ph"] = {int(k): float(v) for k, v in cluster_ph.items()}
            # liquid share per pH-carrying cluster: lets the sanity band judge
            # the MAIN solution and merely count nearly-dry pocket outliers
            total_liq = float(liq_vol_c.sum())
            if dom_to_cl is None:
                share = liq_vol_c
            else:
                # mode C: judge each reading by its PARENT CLUSTER's liquid
                # share — per-domain shares collapse below the 1% band
                # threshold at fine partitions and silently disabled the
                # pH plausibility gate (review finding)
                cl_liq = np.bincount(dom_to_cl, weights=liq_vol_c,
                                     minlength=n_cl)
                share = cl_liq[dom_to_cl]
            metrics["cluster_liq_frac"] = {
                int(k): (float(share[int(k)]) / total_liq
                         if total_liq > 0.0 else 0.0)
                for k in cluster_ph}
        return trial, None, metrics

    # -------------------------------------------------------------------- run
    def run(self, state: Optional[SimulationState] = None,
            out_dir: Optional[str] = None,
            audit_hook: Optional[Callable[[dict], None]] = None
            ) -> Tuple[SimulationState, dict]:
        """Run through the configured output schedule.

        ``audit_hook`` is an optional, read-only qualification observer.  It is
        deliberately outside :class:`SimulationState`: enabling a paper audit
        must not change the numerical trajectory or checkpoint format.
        """
        state = state or self.initial_state()
        if tuple(state.hydrate_ids) != self.hydrate_ids:
            raise RuntimeError(
                "checkpoint hydrate channels do not match the backend's channel "
                f"order - bundle changed between runs? state: {state.hydrate_ids} "
                f"backend: {self.hydrate_ids}")
        if (self._np_cfg is not None and not self._np_cfg.diagnostics_only
                and tuple(state.aq_species_ids)
                != tuple(self.backend.aq_species_ids)):
            raise RuntimeError(
                "checkpoint aqueous species order does not match the "
                "backend's - bundle changed between runs, or species "
                "transport was toggled mid-run (no silent fallback): "
                f"state {state.aq_species_ids[:5]}... backend "
                f"{tuple(self.backend.aq_species_ids)[:5]}...")
        expect_em = tuple((h, dc) for h in self.hydrate_ids
                          for dc in self._hydrate_endmembers[h])
        if tuple(state.endmember_ids) != expect_em:
            raise RuntimeError(
                "checkpoint endmember universe does not match the backend's - "
                "bundle changed between runs? (E1: endmember ledgers are "
                "positional and never remapped)")
        # names matching is not enough: a bundle whose DCH stoichiometry rows
        # changed under unchanged DC names must fail FAST here, not drift into
        # a mid-run balance_endmember_elements abort (review finding)
        if self._endmember_elements is not None and len(expect_em):
            expect_rows = np.array([self._endmember_elements[dc]
                                    for _, dc in expect_em])
            if not np.array_equal(expect_rows, state.endmember_elements):
                raise RuntimeError(
                    "checkpoint endmember element rows do not match the "
                    "backend's DCH stoichiometry - bundle content changed "
                    "between runs (no silent fallback)")
        sched = self.config.schedule
        outputs = [t for t in sched.output_times_h if t > state.time_h + 1e-12]
        summary: Dict = {
            "config_hash": state.config_hash,
            "code_version": code_version(),
            "backend_id": state.backend_id,
            "outputs": [],
        }
        if getattr(self, "_np_cfg", None) is not None:
            # Tier 0 provenance (PRD 4.6.4): which dw data produced this
            # run's species diffusivities, and which species fell to the
            # declared default — audit trail, spec §2.2 report duty
            summary["species_transport"] = dict(self._np_provenance)
        if audit_hook is not None:
            audit_hook({
                "event": "run_start",
                "time_h": float(state.time_h),
                "dt_h": float(state.dt_h),
                "accept_count": int(state.accept_count),
                "reject_counts": dict(state.reject_counts),
                "dense_hash": state.dense_hash(),
                "full_hash": state.full_hash(),
            })
        last_metrics: dict = {}
        for t_out in outputs:
            while state.time_h < t_out - 1e-12:
                dt_cap = sched.dt_cap_at(state.time_h)
                cruise = min(state.dt_h, dt_cap)
                window_end = sched.next_window_end_after(state.time_h)
                dt_try = min(cruise, t_out - state.time_h)
                if window_end is not None:
                    dt_try = min(dt_try, window_end - state.time_h)
                retries = 0
                while True:
                    dense_before = (state.dense_hash()
                                    if audit_hook is not None else None)
                    full_before = (state.full_hash()
                                   if audit_hook is not None else None)
                    injected_before = state.injected_elements.copy()
                    trial, reject, metrics = self.try_step(state, dt_try)
                    if trial is not None:
                        break
                    state.reject_counts[reject.reason] = (
                        state.reject_counts.get(reject.reason, 0) + 1)
                    retries += 1
                    if audit_hook is not None:
                        audit_hook({
                            "event": "trial_rejected",
                            "time_h": float(state.time_h),
                            "dt_attempt_h": float(dt_try),
                            "retry": int(retries),
                            "reason": reject.reason,
                            "metrics": dict(metrics),
                            "committed_dense_hash_before": dense_before,
                            "committed_dense_hash_after": state.dense_hash(),
                            "committed_full_hash_before_bookkeeping": full_before,
                            "committed_full_hash_after_bookkeeping": state.full_hash(),
                        })
                    if retries > sched.max_retries:
                        raise EngineError(
                            f"step at t={state.time_h} h rejected {retries} times "
                            f"(last reason {reject.reason!r})", reject.reason)
                    # halve, but never below dt_min (a boundary-alignment sliver
                    # already below dt_min just retries at its own size)
                    dt_try = max(dt_try / 2.0, min(sched.dt_min_h, dt_try))
                trial.accept_count = state.accept_count + 1
                # grow cruise dt; a sliver step clamped by the output boundary
                # must not collapse an otherwise healthy step size
                next_cap = sched.dt_cap_at(trial.time_h)
                if sched.dt_windows is not None and retries == 0:
                    # A declared piecewise schedule is authoritative. Output
                    # boundary slivers do not change its next prescribed cap.
                    trial.dt_h = next_cap
                elif retries == 0:
                    trial.dt_h = min(max(dt_try * 2.0, cruise), next_cap)
                else:
                    trial.dt_h = min(dt_try * 2.0, next_cap)
                state = trial
                last_metrics = metrics
                if audit_hook is not None:
                    audit_hook({
                        "event": "step_accepted",
                        "time_start_h": float(state.time_h - dt_try),
                        "time_end_h": float(state.time_h),
                        "dt_accepted_h": float(dt_try),
                        "scheduled_dt_cap_h": float(dt_cap),
                        "retries": int(retries),
                        "metrics": dict(metrics),
                        "injected_delta_mol": (
                            state.injected_elements - injected_before).tolist(),
                        "dense_hash": state.dense_hash(),
                        "full_hash": state.full_hash(),
                    })
            summary["outputs"].append(self._snapshot_row(state, last_metrics))
            if out_dir is not None:
                k_global = self.config.schedule.output_times_h.index(t_out)
                ckpt = storage.save_checkpoint(
                    state, out_dir, f"ckpt_{k_global:03d}")
                if audit_hook is not None:
                    audit_hook({
                        "event": "checkpoint_written",
                        "time_h": float(state.time_h),
                        "path": str(ckpt),
                        "dense_hash": state.dense_hash(),
                        "full_hash": state.full_hash(),
                    })
        summary["final"] = self._snapshot_row(state, last_metrics)
        summary["sanity_band"] = analysis.sanity_band(summary["outputs"],
                                                      self.config)
        return state, summary

    def _snapshot_row(self, state: SimulationState, metrics: dict) -> dict:
        row = analysis.state_row(state, self.registry, self._gel_eps)
        row["ledger_metrics"] = dict(metrics)
        return row
