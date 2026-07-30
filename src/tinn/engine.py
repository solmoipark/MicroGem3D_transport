"""Transactional orchestrator: trial -> kinetics -> dissolution -> backend ->
morphology -> ledger checks -> atomic commit or full rollback with dt halving.

The engine is the only mutator of SimulationState. Rejection reasons are stable
identifiers: placement_capacity, insufficient_water, cluster_dryout,
backend_failure, balance_* (from ledger checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import analysis, backend as backend_mod
from . import dissolution, ledger, morphology, storage, transport
from .config import TinnConfig
from .geometry import initialize_rve
from .kinetics import KineticsModel, make_kinetics
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                       Registry, SOLID_PHASE_IDS, default_registry)
from .state import SimulationState, code_version

REJECT_BACKEND_FAILURE = "backend_failure"
REJECT_CLUSTER_DRYOUT = "cluster_dryout"


class EngineError(RuntimeError):
    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class StepReject:
    reason: str


def _neighbor_best_label(labels: np.ndarray, liquid: np.ndarray) -> np.ndarray:
    """Label of the neighboring cluster with the largest liquid contact — the
    cluster that actually supplies the dissolution weight (deterministic:
    argmax over the fixed axis order breaks ties). Liquid is clamped to >= 0:
    placement float dust can leave ~-1e-19 in a voxel, and a labeled neighbor
    must never be disqualified by noise (it would stall the ring propagation
    for coated sites and misreport cluster_dryout)."""
    labs = []
    liqs = []
    for ax, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        labs.append(np.roll(labels, shift, axis=ax))
        liqs.append(np.roll(liquid, shift, axis=ax))
    labs = np.stack(labs)
    liqs = np.where(labs >= 0, np.maximum(np.stack(liqs), 0.0), -1.0)
    best = np.argmax(liqs, axis=0)
    best_lab = np.take_along_axis(labs, best[None], axis=0)[0]
    best_liq = np.take_along_axis(liqs, best[None], axis=0)[0]
    return np.where(best_liq >= 0.0, best_lab, np.int64(-1))


class Engine:
    def __init__(self, config: TinnConfig, registry: Optional[Registry] = None,
                 reaction_backend: Optional[backend_mod.ReactionBackend] = None,
                 kinetics: Optional[KineticsModel] = None):
        self.config = config
        self.registry = registry or default_registry()
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
            self.backend = GemsBackend(worker, config.temperature_K)
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

    # ------------------------------------------------------------------ setup
    def initial_state(self) -> SimulationState:
        rve = initialize_rve(self.config, self.registry)
        return SimulationState.from_geometry(
            self.config, self.registry, rve, self.backend.backend_id,
            hydrate_ids=self.hydrate_ids,
            hydrate_endmembers=self._hydrate_endmembers,
            endmember_elements=self._endmember_elements)

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

        prev_liquid = trial.capillary_liquid.copy()
        labels, n_clusters = transport.label_clusters(prev_liquid)

        dis = dissolution.dissolve(trial, reg, dn)
        vacated = dis.removed_vol.sum(axis=0)

        # site -> cluster attribution (own label, else wettest neighboring cluster)
        site_cluster = np.where(labels >= 0, labels,
                                _neighbor_best_label(labels, prev_liquid))
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
                nxt = np.where(relay, _neighbor_best_label(deep, prev_liquid),
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

        if state.cluster_inventory.shape[0] == n_clusters:
            inv_in = state.cluster_inventory
        elif state.cluster_inventory.shape[0] == 0:
            inv_in = np.zeros((n_clusters, len(ELEMENT_IDS)))
        else:
            raise RuntimeError(
                f"cluster inventory has {state.cluster_inventory.shape[0]} rows but "
                f"{n_clusters} clusters were labeled - state is corrupted (no fallback)")
        # E2: per-cluster endmember pools ride the same labeling contract
        if state.cluster_endmember_mol.shape[0] == n_clusters \
                and state.cluster_endmember_mol.shape[1] == self._n_em:
            em_pool_in = state.cluster_endmember_mol
        elif state.cluster_endmember_mol.shape[0] == 0:
            em_pool_in = np.zeros((n_clusters, self._n_em))
        else:
            raise RuntimeError(
                f"cluster endmember pool has shape "
                f"{state.cluster_endmember_mol.shape} but "
                f"({n_clusters}, {self._n_em}) was expected - state is "
                f"corrupted (no fallback)")

        # ---- reaction phase (mode-dependent) --------------------------------
        # incremental (stoichiometric): parcels are NEW precipitates appended.
        # snapshot (gems3k, PRD v2.2): each cluster's owned hydrates + solution
        # inventory + released elements + free water are re-equilibrated as ONE
        # system; the result replaces the cluster's assemblage (re-dissolution,
        # CH consumption and phase rearrangement emerge from equilibrium).
        recon = np.where(labels >= 0, labels, site_cluster)
        snapshot = self.backend.mode == "snapshot"

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
        own_vol = np.zeros((n_clusters, n_h))
        owned_elem = np.zeros((n_clusters, n_h, len(ELEMENT_IDS)))
        owned_mol = np.zeros((n_clusters, n_h))
        owned_em = np.zeros((n_clusters, self._n_em))
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
                # E2 (PRD 4.5 v3.0): the AMOUNT stays volume-share derived,
                # but the endmember RATIOS come from the cluster's OWN pool —
                # the average-composition approximation is gone wherever a
                # pool exists. New/rewetted clusters without a material pool
                # fall back to the global channel ratio (self-heals next
                # step: solved pools are replaced by their own parcels).
                sl = self._em_slice[self.hydrate_ids[h]]
                g = trial.endmember_mol[sl]
                gsum = float(g.sum())
                g_ratio = g / gsum if gsum > 0.0 else np.zeros(sl.stop - sl.start)
                pool = em_pool_in[:, sl]
                psum = pool.sum(axis=1)
                amounts = owned_mol[:, h]
                use_pool = psum > np.maximum(1e-9 * np.abs(amounts), 1e-300)
                safe = np.where(use_pool, psum, 1.0)
                ratios = np.where(use_pool[:, None], pool / safe[:, None],
                                  g_ratio[None, :])
                owned_em[:, sl] = amounts[:, None] * ratios
                # fed elements follow the fed COMPOSITION exactly, so the
                # global element/endmember ledgers stay mutually consistent
                # by construction ("subtract what was fed")
                owned_elem[:, h, :] = owned_em[:, sl] @ trial.endmember_elements[sl]
        owned_gel_c = (own_vol * gel_eps).sum(axis=1)

        total_water_mol = float(water_mol_c.sum())
        for c in range(n_clusters):
            rel = {p: float(released[c, k]) for k, p in enumerate(KINETIC_PHASE_IDS)
                   if released[c, k] > 0.0}
            has_solids = snapshot and own_vol[c].sum() > 0.0
            has_inventory = bool(np.any(inv_in[c] != 0.0))
            if not rel and not has_solids and not has_inventory:
                residual[c] = inv_in[c]
                continue
            water_c = float(water_mol_c[c])
            trace_water = water_c < 1e-3 * total_water_mol
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
                        result = self.backend.react(scaled, water_c, inv_in[c],
                                                    solid_elements=solid_elem_c)
                    else:
                        result = self.backend.react(scaled, water_c, inv_in[c])
                except backend_mod.BackendTransientError:
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
                    residual[c] = inv_in[c]
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
                delta_c = env_new - own_vol[c]
                member_c = recon == c
                grow_c = float(np.clip(delta_c, 0.0, None).sum())
                vac_tot_c = (s * float(np.where(member_c, vacated, 0.0).sum())
                             + float(np.clip(-delta_c, 0.0, None).sum()))
                cap_c = (float(np.where(member_c, trial.capillary_liquid, 0.0).sum())
                         + vac_tot_c)
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
                    gas_c_vol = float(np.where(member_c, trial.capillary_gas,
                                               0.0).sum())
                    water_out_vol = (result.water_consumed_mol * vm_w
                                     + float((env_new * gel_eps).sum())
                                     - owned_gel_c[c])
                    deficit_est = grow_c - vac_tot_c - water_out_vol
                    freeze = deficit_est > gas_c_vol - 1e-12 * max(1.0, gas_c_vol)
                if freeze:
                    scale_c[c] = 0.0
                    residual[c] = inv_in[c]
                    continue
            solved[c] = True
            scale_c[c] = s
            chem_mol_c[c] = result.water_consumed_mol
            residual[c] = result.residual_inventory
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

        # ---- spatial application -------------------------------------------
        backend_removal_vol = 0.0
        removed_total = 0.0
        if snapshot:
            # deltas vs owned volume; NEGATIVES FIRST — re-dissolved hydrate
            # frees pore capacity before growth is placed (v1 pattern)
            delta_env = np.where(solved[:, None], parcel_env - own_vol, 0.0)
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
                                   vacated, demand, recon,
                                   recon if snapshot else site_cluster)
        if outcome.status != morphology.STATUS_OK:
            return None, StepReject(outcome.status), {}
        # water flows into leftover vacated/freed space (same connected liquid)
        trial.capillary_liquid += vacated

        # cluster water reconciliation: excess liquid volume becomes capillary gas.
        # The reference volume uses the same recon partition as the current volume
        # so sub-threshold liquid at site voxels cancels on both sides.
        cur_vol_c = np.bincount(recon[recon >= 0],
                                weights=trial.capillary_liquid[recon >= 0],
                                minlength=n_clusters)
        prev_vol_recon_c = np.bincount(recon[recon >= 0],
                                       weights=prev_liquid[recon >= 0],
                                       minlength=n_clusters)
        target_vol_c = prev_vol_recon_c - (chem_mol_c + gel_mol_c) * vm_w
        diff_c = cur_vol_c - target_vol_c
        if n_clusters > 0:
            # negative diff: water returned to solution (re-dissolution, solute
            # water re-emerging as solvent) refills capillary liquid from the
            # cluster's own gas space; without enough local gas the returned
            # volume has nowhere to appear
            gas_vol_c = np.bincount(recon[recon >= 0],
                                    weights=trial.capillary_gas[recon >= 0],
                                    minlength=n_clusters)
            deficit = np.clip(-diff_c, 0.0, None)
            tol_c = 1e-12 * (1.0 + np.abs(target_vol_c))
            if np.any((deficit > tol_c) & (deficit > gas_vol_c + tol_c)):
                return None, StepReject("balance_water"), {}
            with np.errstate(invalid="ignore", divide="ignore"):
                refill = np.where((deficit > tol_c) & (gas_vol_c > 0.0),
                                  deficit / gas_vol_c, 0.0)
                factor = np.where(cur_vol_c > 0.0,
                                  np.clip(diff_c, 0.0, None) / cur_vol_c, 0.0)
            refill_vox = np.where(recon >= 0, refill[np.clip(recon, 0, None)], 0.0)
            added_liq = trial.capillary_gas * refill_vox
            trial.capillary_gas -= added_liq
            trial.capillary_liquid += added_liq
            fac_vox = np.where(recon >= 0, factor[np.clip(recon, 0, None)], 0.0)
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
                parcel_mol[sv].sum(axis=0) - owned_mol[sv].sum(axis=0))
            trial.hydrate_env_vol_vox = trial.hydrate_env_vol_vox + (
                parcel_env[sv].sum(axis=0) - own_vol[sv].sum(axis=0))
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

        # relabel + conservative inventory remap
        new_labels, n_new = transport.label_clusters(trial.capillary_liquid)
        remap = transport.remap_inventories(labels, prev_liquid, new_labels,
                                            trial.capillary_liquid, residual, n_new)
        if remap.dryout:
            return None, StepReject(REJECT_CLUSTER_DRYOUT), {}
        trial.cluster_inventory = remap.inventory
        # E2: pools follow the assemblage — a solved cluster's pool IS its own
        # parcels (absolute replacement, non-compounding); frozen clusters
        # keep theirs. Remapped with the same overlaps as the inventory;
        # zero-overlap rows drop (pools are compositional memory — the global
        # ledgers carry conservation, so a stranded pool is not a dryout).
        pools_prev = em_pool_in.copy()
        if snapshot and bool(solved.any()):
            pools_prev[solved] = parcel_em[solved]
        pool_remap = transport.remap_inventories(
            labels, prev_liquid, new_labels, trial.capillary_liquid,
            pools_prev, n_new)
        trial.cluster_endmember_mol = pool_remap.inventory
        trial.cluster_id = new_labels
        for prev_c, new_c, ov in remap.events:
            trial.remap_events["time_h"].append(trial.time_h + dt_h)
            trial.remap_events["prev"].append(prev_c)
            trial.remap_events["new"].append(new_c)
            trial.remap_events["overlap_vox"].append(ov)
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
        report = ledger.check_all(trial, reg, placement)
        if not report.ok:
            return None, StepReject(report.violations[0]), report.metrics
        metrics = dict(report.metrics)
        if cluster_ph:
            metrics["cluster_ph"] = {int(k): float(v) for k, v in cluster_ph.items()}
            # liquid share per pH-carrying cluster: lets the sanity band judge
            # the MAIN solution and merely count nearly-dry pocket outliers
            total_liq = float(liq_vol_c.sum())
            metrics["cluster_liq_frac"] = {
                int(k): (float(liq_vol_c[int(k)]) / total_liq
                         if total_liq > 0.0 else 0.0)
                for k in cluster_ph}
        return trial, None, metrics

    # -------------------------------------------------------------------- run
    def run(self, state: Optional[SimulationState] = None,
            out_dir: Optional[str] = None) -> Tuple[SimulationState, dict]:
        state = state or self.initial_state()
        if tuple(state.hydrate_ids) != self.hydrate_ids:
            raise RuntimeError(
                "checkpoint hydrate channels do not match the backend's channel "
                f"order - bundle changed between runs? state: {state.hydrate_ids} "
                f"backend: {self.hydrate_ids}")
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
        last_metrics: dict = {}
        for t_out in outputs:
            while state.time_h < t_out - 1e-12:
                cruise = state.dt_h
                dt_try = min(cruise, t_out - state.time_h)
                retries = 0
                while True:
                    trial, reject, metrics = self.try_step(state, dt_try)
                    if trial is not None:
                        break
                    state.reject_counts[reject.reason] = (
                        state.reject_counts.get(reject.reason, 0) + 1)
                    retries += 1
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
                if retries == 0:
                    trial.dt_h = min(max(dt_try * 2.0, cruise), sched.dt_initial_h)
                else:
                    trial.dt_h = min(dt_try * 2.0, sched.dt_initial_h)
                state = trial
                last_metrics = metrics
            summary["outputs"].append(self._snapshot_row(state, last_metrics))
            if out_dir is not None:
                k_global = self.config.schedule.output_times_h.index(t_out)
                storage.save_checkpoint(state, out_dir, f"ckpt_{k_global:03d}")
        summary["final"] = self._snapshot_row(state, last_metrics)
        summary["sanity_band"] = analysis.sanity_band(summary["outputs"],
                                                      self.config)
        return state, summary

    def _snapshot_row(self, state: SimulationState, metrics: dict) -> dict:
        row = analysis.state_row(state, self.registry, self._gel_eps)
        row["ledger_metrics"] = dict(metrics)
        return row
