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
                       Registry, default_registry)
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


def _neighbor_min_label(labels: np.ndarray) -> np.ndarray:
    big = np.iinfo(np.int64).max
    m = np.full(labels.shape, big, dtype=np.int64)
    lab = np.where(labels >= 0, labels, big)
    for ax, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        m = np.minimum(m, np.roll(lab, shift, axis=ax))
    return np.where(m == big, np.int64(-1), m)


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
        # rule coefficient matrix C[k, h]: mol hydrate h per mol kinetic phase k
        self._coeff = np.zeros((len(KINETIC_PHASE_IDS), len(HYDRATE_PHASE_IDS)))
        self._water_mol = np.zeros(len(KINETIC_PHASE_IDS))
        rules = config.chemistry.stoichiometric_rules or {}
        for k, p in enumerate(KINETIC_PHASE_IDS):
            if p in rules:
                self._water_mol[k] = rules[p].water_mol
                for hid, coeff in rules[p].products.items():
                    self._coeff[k, HYDRATE_PHASE_IDS.index(hid)] = coeff

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

        # kinetic targets: reach alpha(t+dt), catching up any previous deficit
        alpha_target = self.kinetics.alpha_at(trial.time_h + dt_h)
        dissolved = trial.initial_phase_mol - trial.phase_mol
        dn = np.clip(trial.initial_phase_mol * alpha_target - dissolved,
                     0.0, trial.phase_mol)

        prev_liquid = trial.capillary_liquid.copy()
        labels, n_clusters = transport.label_clusters(prev_liquid)

        dis = dissolution.dissolve(trial, reg, dn)
        vacated = dis.removed_vol.sum(axis=0)

        # site -> cluster attribution (own label, else smallest neighboring label)
        site_cluster = np.where(labels >= 0, labels, _neighbor_min_label(labels))
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

        inv_in = (state.cluster_inventory if state.cluster_inventory.shape[0] == n_clusters
                  else np.zeros((n_clusters, len(ELEMENT_IDS))))

        chem_mol_c = np.zeros(n_clusters)
        residual = np.zeros((n_clusters, len(ELEMENT_IDS)))
        parcel_rows: List[tuple] = []
        hydrate_add = np.zeros(len(HYDRATE_PHASE_IDS))
        backend_bulk_vol = 0.0
        for c in range(n_clusters):
            rel = {p: float(released[c, k]) for k, p in enumerate(KINETIC_PHASE_IDS)
                   if released[c, k] > 0.0}
            if not rel:
                residual[c] = inv_in[c]
                continue
            try:
                result = self.backend.react(rel, float(water_mol_c[c]), inv_in[c])
            except Exception:
                return None, StepReject(REJECT_BACKEND_FAILURE), {}
            if result.status != backend_mod.STATUS_OK:
                return None, StepReject(result.status), {}
            chem_mol_c[c] = result.water_consumed_mol
            residual[c] = result.residual_inventory
            for hid, mol in result.parcels:
                hi = HYDRATE_PHASE_IDS.index(hid)
                hydrate_add[hi] += mol
                backend_bulk_vol += mol * env_vm[hi]
                parcel_rows.append((float(trial.time_h + dt_h), int(c), hid, float(mol),
                                    float(mol * skel_vm[hi]), float(mol * env_vm[hi])))

        # per-voxel bulk envelope demand from local dissolution (same linear rules)
        demand = np.zeros_like(trial.hydrate_fraction)
        for k in range(len(KINETIC_PHASE_IDS)):
            vm = trial.vm_vox(reg, KINETIC_PHASE_IDS[k])
            mol_vox = dis.removed_vol[k] / vm
            for h in range(len(HYDRATE_PHASE_IDS)):
                if self._coeff[k, h] > 0.0:
                    demand[h] += mol_vox * (self._coeff[k, h] * env_vm[h])
        requested_vol = float(demand.sum())

        # gel water per cluster + total water feasibility (chemical + gel)
        gel_vol_vox = (demand * gel_eps[:, None, None, None]).sum(axis=0)
        gel_mol_c = np.bincount(site_cluster[site_mask],
                                weights=gel_vol_vox[site_mask],
                                minlength=n_clusters) / vm_w
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
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(cur_vol_c > 0.0, np.clip(diff_c, 0.0, None) / cur_vol_c, 0.0)
        fac_vox = np.where(recon >= 0, factor[np.clip(recon, 0, None)], 0.0)
        removed_liq = trial.capillary_liquid * fac_vox
        trial.capillary_liquid -= removed_liq
        trial.capillary_gas += removed_liq

        # authoritative mol ledger (exact scalar arithmetic)
        chem_total = float(chem_mol_c.sum())
        gel_total = float(gel_mol_c.sum())
        trial.phase_mol = trial.phase_mol - dis.removed_mol
        trial.unmet_mol = dis.unmet_mol.copy()
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
        for k_out, t_out in enumerate(outputs):
            while state.time_h < t_out - 1e-12:
                dt = min(state.dt_h, t_out - state.time_h)
                trial, reject, metrics = self.try_step(state, dt)
                if trial is not None:
                    trial.accept_count = state.accept_count + 1
                    trial.dt_h = min(state.dt_h * 2.0, sched.dt_initial_h)
                    state = trial
                    last_metrics = metrics
                else:
                    state.reject_counts[reject.reason] = (
                        state.reject_counts.get(reject.reason, 0) + 1)
                    state.dt_h = dt / 2.0
                    if state.dt_h < sched.dt_min_h:
                        raise EngineError(
                            f"dt underflow at t={state.time_h} h after reject "
                            f"{reject.reason!r}", reject.reason)
            summary["outputs"].append(self._snapshot_row(state, last_metrics))
            if out_dir is not None:
                storage.save_checkpoint(state, out_dir, f"ckpt_{k_out:03d}")
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
