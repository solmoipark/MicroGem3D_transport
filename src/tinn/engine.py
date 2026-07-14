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

from . import backend as backend_mod
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
    argmax over the fixed axis order breaks ties)."""
    labs = []
    liqs = []
    for ax, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        labs.append(np.roll(labels, shift, axis=ax))
        liqs.append(np.roll(liquid, shift, axis=ax))
    labs = np.stack(labs)
    liqs = np.where(labs >= 0, np.stack(liqs), -1.0)
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
            raise NotImplementedError("gems3k backend arrives with milestone M4")
        self.kinetics = kinetics or make_kinetics(config)

    # ------------------------------------------------------------------ setup
    def initial_state(self) -> SimulationState:
        rve = initialize_rve(self.config, self.registry)
        return SimulationState.from_geometry(self.config, self.registry, rve,
                                             self.backend.backend_id)

    # ------------------------------------------------------------------- step
    def try_step(self, state: SimulationState, dt_h: float
                 ) -> Tuple[Optional[SimulationState], Optional[StepReject], dict]:
        """One trial step of size dt_h. Returns (trial, None, metrics) on success
        or (None, StepReject, metrics) on rejection; `state` is never mutated."""
        trial = state.clone()
        reg = self.registry
        vm_w = trial.vm_vox(reg, "H2O")
        env_vm = np.array([trial.vm_vox(reg, h, envelope=True) for h in HYDRATE_PHASE_IDS])
        skel_vm = np.array([reg.get(h).skeleton_molar_volume_cm3 / trial.vox_cm3
                            for h in HYDRATE_PHASE_IDS])
        gel_eps = np.array([reg.get(h).gel_porosity for h in HYDRATE_PHASE_IDS])

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
        if np.any(site_mask & (site_cluster < 0)):
            return None, StepReject(REJECT_CLUSTER_DRYOUT), {}

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
                f"{n_clusters} clusters were labeled — state is corrupted (no fallback)")

        # Backend negotiation per cluster: a water-starved cluster (e.g. a sealed
        # wet pocket) dissolves only what its own water can react — the release is
        # scaled down deterministically and the shortfall is returned to the solid
        # and recorded as unmet, never hidden and never a whole-run abort.
        chem_mol_c = np.zeros(n_clusters)
        residual = np.zeros((n_clusters, len(ELEMENT_IDS)))
        parcel_mol = np.zeros((n_clusters, len(HYDRATE_PHASE_IDS)))
        parcel_rows: List[tuple] = []
        scale_c = np.ones(n_clusters)
        gel_per_mol = env_vm * gel_eps  # gel water volume per mol of each hydrate
        for c in range(n_clusters):
            rel = {p: float(released[c, k]) for k, p in enumerate(KINETIC_PHASE_IDS)
                   if released[c, k] > 0.0}
            if not rel:
                residual[c] = inv_in[c]
                continue
            water_c = float(water_mol_c[c])
            s = 1.0
            result = None
            for _ in range(60):
                scaled = {p: v * s for p, v in rel.items()}
                try:
                    result = self.backend.react(scaled, water_c, inv_in[c])
                except Exception:
                    return None, StepReject(REJECT_BACKEND_FAILURE), {}
                if result.status == backend_mod.STATUS_OK:
                    pm = np.zeros(len(HYDRATE_PHASE_IDS))
                    for hid, mol in result.parcels:
                        pm[HYDRATE_PHASE_IDS.index(hid)] += mol
                    need = result.water_consumed_mol + float((pm * gel_per_mol).sum()) / vm_w
                    if need <= water_c:
                        break
                    s *= min(0.5, water_c * (1.0 - 1e-9) / need)
                elif result.status == backend_mod.STATUS_INSUFFICIENT_WATER:
                    s *= 0.5
                else:
                    return None, StepReject(result.status), {}
            else:
                return None, StepReject(backend_mod.STATUS_INSUFFICIENT_WATER), {}
            scale_c[c] = s
            chem_mol_c[c] = result.water_consumed_mol
            residual[c] = result.residual_inventory
            for hid, mol in result.parcels:
                hi = HYDRATE_PHASE_IDS.index(hid)
                parcel_mol[c, hi] += mol
                parcel_rows.append((float(trial.time_h + dt_h), int(c), hid, float(mol),
                                    float(mol * skel_vm[hi]), float(mol * env_vm[hi])))
        hydrate_add = parcel_mol.sum(axis=0)
        backend_bulk_vol = float((parcel_mol * env_vm).sum())

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

        # per-voxel bulk envelope demand: the backend's parcels (whatever chemistry
        # produced them) are distributed over the cluster's dissolution sites in
        # proportion to locally dissolved volume — the engine stays backend-agnostic
        demand = np.zeros_like(trial.hydrate_fraction)
        for c in np.flatnonzero(parcel_mol.any(axis=1)):
            mask = site_mask & (site_cluster == c)
            wsum = float(vacated[mask].sum())
            if wsum <= 0.0:
                return None, StepReject(REJECT_BACKEND_FAILURE), {}
            frac = np.where(mask, vacated, 0.0) / wsum
            for h in np.flatnonzero(parcel_mol[c] > 0.0):
                demand[h] += (parcel_mol[c, h] * env_vm[h]) * frac
        requested_vol = float(demand.sum())

        # gel water per cluster + total water feasibility (chemical + gel)
        gel_mol_c = (parcel_mol * (env_vm * gel_eps)).sum(axis=1) / vm_w
        if np.any(chem_mol_c + gel_mol_c > water_mol_c + 1e-30):
            return None, StepReject(backend_mod.STATUS_INSUFFICIENT_WATER), {}

        recon = np.where(labels >= 0, labels, site_cluster)
        outcome = morphology.place(trial.hydrate_fraction, trial.capillary_liquid,
                                   vacated, demand, recon, site_cluster)
        if outcome.status != morphology.STATUS_OK:
            return None, StepReject(outcome.status), {}
        # water flows into leftover vacated space (same connected liquid)
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
        if np.any(diff_c < -1e-12 * (1.0 + np.abs(target_vol_c))):
            return None, StepReject("balance_water"), {}
        if n_clusters > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                factor = np.where(cur_vol_c > 0.0,
                                  np.clip(diff_c, 0.0, None) / cur_vol_c, 0.0)
            fac_vox = np.where(recon >= 0, factor[np.clip(recon, 0, None)], 0.0)
            removed_liq = trial.capillary_liquid * fac_vox
            trial.capillary_liquid -= removed_liq
            trial.capillary_gas += removed_liq

        # authoritative mol ledger (exact scalar arithmetic)
        chem_total = float(chem_mol_c.sum())
        gel_total = float(gel_mol_c.sum())
        trial.phase_mol = trial.phase_mol - dis.removed_mol
        trial.unmet_mol = trial.unmet_mol + dis.unmet_mol
        trial.hydrate_mol = trial.hydrate_mol + hydrate_add
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
            placed_bulk_vol_vox=outcome.placed_vol_vox)
        report = ledger.check_all(trial, reg, placement)
        if not report.ok:
            return None, StepReject(report.violations[0]), report.metrics
        return trial, None, report.metrics

    # -------------------------------------------------------------------- run
    def run(self, state: Optional[SimulationState] = None,
            out_dir: Optional[str] = None) -> Tuple[SimulationState, dict]:
        state = state or self.initial_state()
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
        return state, summary

    def _snapshot_row(self, state: SimulationState, metrics: dict) -> dict:
        reg = self.registry
        alpha = state.alpha()
        cap_por = float((state.capillary_liquid + state.capillary_gas).mean())
        gel_por_vol = sum(float(state.hydrate_fraction[i].sum()) * reg.get(h).gel_porosity
                          for i, h in enumerate(HYDRATE_PHASE_IDS))
        n_vox = state.capillary_liquid.size
        reacted_mass_g = float(np.sum(
            (state.initial_phase_mol - state.phase_mol)
            * np.array([reg.get(p).molar_mass_g_mol for p in KINETIC_PHASE_IDS])))
        gas_cm3 = float(state.capillary_gas.sum()) * state.vox_cm3
        return {
            "time_h": state.time_h,
            "alpha": {p: float(alpha[i]) for i, p in enumerate(KINETIC_PHASE_IDS)},
            "phase_mol": {p: float(state.phase_mol[i])
                          for i, p in enumerate(KINETIC_PHASE_IDS)},
            "hydrate_mol": {h: float(state.hydrate_mol[i])
                            for i, h in enumerate(HYDRATE_PHASE_IDS)},
            "unmet_mol": {p: float(state.unmet_mol[i])
                          for i, p in enumerate(KINETIC_PHASE_IDS)},
            "water_mol": {"free": state.water_free_mol, "gel": state.water_gel_mol,
                          "bound": state.water_bound_mol},
            "porosity_capillary": cap_por,
            "porosity_total": cap_por + gel_por_vol / n_vox,
            "chem_shrinkage_ml_per_g_reacted": (gas_cm3 / reacted_mass_g
                                                if reacted_mass_g > 0 else 0.0),
            "accept_count": state.accept_count,
            "reject_counts": dict(state.reject_counts),
            "ledger_metrics": dict(metrics),
        }
