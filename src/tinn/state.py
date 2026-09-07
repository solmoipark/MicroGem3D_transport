"""SimulationState: dense arrays + tables + authoritative mol ledger (phase-vector native).

Mol quantities are authoritative; dense volume arrays are the spatial view and are
cross-checked against the ledger every step (ledger.py). The unassigned "inert"
solid has no invented chemistry: it is tracked as a constant volume invariant and
never enters the element ledger or the backend.
"""

from __future__ import annotations

import copy
import hashlib
import subprocess
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import TinnConfig
from .geometry import RVEInit
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, INERT_PHASE_ID,
                       KINETIC_PHASE_IDS, Registry, element_vector)

_DENSE_FIELDS = ("anhydrous_fraction", "hydrate_fraction", "capillary_liquid",
                 "capillary_gas", "particle_id", "cluster_id")

_code_version_cache: str | None = None


def code_version() -> str:
    global _code_version_cache
    if _code_version_cache is None:
        try:
            root = Path(__file__).resolve().parents[2]
            _code_version_cache = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, check=True).stdout.strip()
        except Exception:
            _code_version_cache = "unknown"
    return _code_version_cache


def formula_elements(formula: Dict[str, float]) -> np.ndarray:
    return element_vector(formula, 1.0)


@dataclass
class SimulationState:
    config: TinnConfig
    # run-scoped hydrate channel order: registry HYDRATE_PHASE_IDS for the
    # stoichiometric backend, the bundle's solid phases for gems3k
    hydrate_ids: tuple
    # --- dense fields, (z, y, x), float64 / int64 ---
    anhydrous_fraction: np.ndarray   # (5, N, N, N), channels = SOLID_PHASE_IDS
    hydrate_fraction: np.ndarray     # (H, N, N, N) bulk envelope incl. gel pores
    capillary_liquid: np.ndarray     # (N, N, N)
    capillary_gas: np.ndarray        # (N, N, N) fixed-RVE residual (chem. shrinkage)
    particle_id: np.ndarray          # (N, N, N) int64, initial placement label
    cluster_id: np.ndarray           # (N, N, N) int64, snapshot-local, -1 = none
    # --- tables ---
    particles: Dict[str, np.ndarray]
    subgrid_bins: Dict[str, np.ndarray]
    parcels: Dict[str, List]         # immutable rows appended on commit
    remap_events: Dict[str, List]
    cluster_inventory: np.ndarray    # (K, n_elements) mol, dissolved-solution ledger
    # E2 (PRD 2.3/4.5 v3.0): per-cluster endmember pools (K, M) — the
    # COMPOSITION memory of each cluster's owned hydrates. Amounts stay
    # volume-share derived; these rows only set the endmember RATIOS fed back
    # to re-equilibration, are replaced by the cluster's own parcels on every
    # solved step (non-compounding), and are remapped by the same liquid
    # overlap as cluster_inventory. Zero-overlap rows drop back to the global
    # average (conservation lives in the global ledgers, not here).
    cluster_endmember_mol: np.ndarray  # (K, M)
    # --- ledger / header (mol is authoritative) ---
    time_h: float
    dt_h: float
    phase_mol: np.ndarray            # (4,) remaining anhydrous mol
    initial_phase_mol: np.ndarray    # (4,)
    unmet_mol: np.ndarray            # (4,) current deficit vs kinetic target
    hydrate_mol: np.ndarray          # (H,) over hydrate_ids
    hydrate_env_vol_vox: np.ndarray  # (H,) authoritative placed bulk-envelope volume
    hydrate_elements_ch: np.ndarray  # (H, E) per-channel element pools of the
                                     # CURRENT hydrate holdings (signed updates
                                     # under full re-equilibration)
    # --- E1 endmember ledger (PRD 2.3 rev.3; FORMAT_VERSION 3) ---
    endmember_ids: tuple             # run-scoped ((hydrate_id, dc_name), ...)
    endmember_mol: np.ndarray        # (M,) CURRENT holdings per endmember (signed)
    endmember_elements: np.ndarray   # (M, E) element row per endmember — static
                                     # per run, from the backend's own source
                                     # (bundle DCH / registry), stored so
                                     # checkpoint re-checks need no re-probe
    boundary_exchanged_elements: np.ndarray  # (E,) RESERVED for RT-W3 boundary
                                     # reservoirs (always zeros until then);
                                     # closure reads initial+injected+boundary
    injected_elements: np.ndarray    # (E,) backend-injected seeds/solver floors
    water_free_mol: float
    water_gel_mol: float
    water_bound_mol: float
    initial_water_mol: float
    inert_volume_vox: float          # constant invariant (no invented chemistry)
    initial_elements: np.ndarray     # (n_elements,)
    accept_count: int
    reject_counts: Dict[str, int]
    rng_state: dict
    config_hash: str
    backend_id: str
    boundary_water_mol: float = 0.0  # v4.0/RT (FORMAT_VERSION 4): net solvent
                                     # exchanged through boundary reservoirs
                                     # (RT-W3; zero until then). The water
                                     # identity reads initial + boundary.
                                     # Trails the field list so pre-RT
                                     # construction sites stay valid.
    # v4.0/RT-W2b GEM-call economy snapshots (PRD 4.6.2): the last
    # equilibrated inventory/water per domain and the steps since. Rows
    # follow the cluster_inventory contract (n_domains or 0 = no records);
    # age -1 = no valid record (forced equilibration). Persisted because the
    # post-restart call pattern - hence the trajectory - depends on them.
    # Empty (rows 0) whenever the economy is off, so W2a-era checkpoints and
    # hashes are unchanged by their existence.
    domain_eq_inventory: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(ELEMENT_IDS))))
    domain_eq_water: np.ndarray = field(default_factory=lambda: np.zeros(0))
    domain_eq_age: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Tier 0 / RT-P0b (FORMAT_VERSION 5, PRD 4.6.4): frozen aqueous
    # speciation per domain — a COEFFICIENT CACHE for the NP conductances,
    # not a conservation ledger (it never enters §6.1 closure). Rows follow
    # the cluster_inventory contract; (0, 0) whenever species transport is
    # off, so pre-Tier-0 trajectories and hashes gain nothing but the
    # format break itself. aq_species_ids is the run-scoped column order
    # (bundle DCH aqueous DCs, solvent excluded), header-owned like
    # endmember_ids.
    aq_species_ids: tuple = ()
    domain_species_mol: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    # RESERVED for RT-S1a (Tier 1 sorption): per-domain sorbed element
    # inventory. Zero rows until the sorption operator lands — reserved in
    # the SAME format break (the v4 boundary_water_mol precedent) so Tier 1
    # does not cost a second anchor re-pin.
    domain_sorbed_mol: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(ELEMENT_IDS))))
    # RT-D1 (review 2026-09-07): identity of the external chemistry the
    # run was computed with - content sha256 of the GEMS bundle and the
    # PHREEQC database plus engine versions. Header metadata only (not a
    # hash input: it identifies the environment, it is not physics);
    # a restart under a different environment is refused by the engine.
    chemistry_env: dict = field(default_factory=dict)

    # --- derived helpers ---
    @property
    def grid_size(self) -> int:
        return self.capillary_liquid.shape[0]

    @property
    def vox_cm3(self) -> float:
        return self.config.rve.voxel_size_um ** 3 * 1e-12

    def vm_vox(self, registry: Registry, phase_id: str, envelope: bool = False) -> float:
        """Molar volume in voxel-volume units."""
        e = registry.get(phase_id)
        vm = e.envelope_molar_volume_cm3 if envelope else e.molar_volume_cm3
        return vm / self.vox_cm3

    def alpha(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            a = 1.0 - self.phase_mol / self.initial_phase_mol
        return np.where(self.initial_phase_mol > 0.0, a, 0.0)

    def dense_hash(self) -> str:
        h = hashlib.sha256()
        for name in _DENSE_FIELDS:
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return h.hexdigest()

    def full_hash(self) -> str:
        """Equality oracle for restart equivalence: dense arrays, ledger, counters,
        RNG, and all tables — nothing persisted may diverge unnoticed."""
        import json as _json
        h = hashlib.sha256(self.dense_hash().encode())
        for arr in (self.phase_mol, self.initial_phase_mol, self.unmet_mol,
                    self.hydrate_mol, self.hydrate_env_vol_vox,
                    self.hydrate_elements_ch, self.endmember_mol,
                    self.endmember_elements, self.boundary_exchanged_elements,
                    self.injected_elements,
                    self.initial_elements, self.cluster_inventory,
                    self.cluster_endmember_mol, self.domain_eq_inventory,
                    self.domain_eq_water, self.domain_eq_age,
                    self.domain_species_mol, self.domain_sorbed_mol):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr(self.hydrate_ids).encode())
        h.update(repr(self.endmember_ids).encode())
        h.update(repr(self.aq_species_ids).encode())
        for x in (self.time_h, self.dt_h, self.water_free_mol, self.water_gel_mol,
                  self.water_bound_mol, self.initial_water_mol,
                  self.boundary_water_mol,
                  self.inert_volume_vox, self.accept_count):
            h.update(repr(x).encode())
        h.update(repr(sorted(self.reject_counts.items())).encode())
        for table in (self.particles, self.subgrid_bins):
            for key in sorted(table):
                arr = np.ascontiguousarray(table[key])
                h.update(key.encode())
                h.update(str(arr.dtype).encode())
                h.update(str(arr.shape).encode())
                h.update(arr.tobytes())
        h.update(_json.dumps(self.parcels, sort_keys=True).encode())
        h.update(_json.dumps(self.remap_events, sort_keys=True).encode())
        h.update(_json.dumps(self.rng_state, sort_keys=True).encode())
        return h.hexdigest()

    def clone(self) -> "SimulationState":
        """Deep copy for trial stepping (copy-on-write commit unit)."""
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                kw[f.name] = v.copy()
            elif isinstance(v, dict):
                kw[f.name] = {k: (val.copy() if isinstance(val, np.ndarray)
                                  else copy.deepcopy(val)) for k, val in v.items()}
            else:
                kw[f.name] = v  # scalars, str, config (immutable use)
        return SimulationState(**kw)

    @classmethod
    def from_geometry(cls, config: TinnConfig, registry: Registry,
                      rve: RVEInit, backend_id: str,
                      hydrate_ids: tuple = HYDRATE_PHASE_IDS,
                      hydrate_endmembers: Optional[Dict[str, tuple]] = None,
                      endmember_elements: Optional[Dict[str, np.ndarray]] = None
                      ) -> "SimulationState":
        n = config.rve.grid_size
        vox_cm3 = config.rve.voxel_size_um ** 3 * 1e-12
        phase_mol = np.zeros(len(KINETIC_PHASE_IDS))
        for i, p in enumerate(KINETIC_PHASE_IDS):
            vol_vox = float(rve.anhydrous_fraction[rve.phase_ids.index(p)].sum())
            phase_mol[i] = vol_vox * vox_cm3 / registry.get(p).molar_volume_cm3
        inert_vol = float(rve.anhydrous_fraction[rve.phase_ids.index(INERT_PHASE_ID)].sum())
        water_mol = float(rve.capillary_liquid.sum()) * vox_cm3 / registry.get("H2O").molar_volume_cm3

        initial_elements = np.zeros(len(ELEMENT_IDS))
        for i, p in enumerate(KINETIC_PHASE_IDS):
            initial_elements += formula_elements(registry.get(p).formula) * phase_mol[i]
        initial_elements += formula_elements(registry.get("H2O").formula) * water_mol

        # E1 endmember index: run-scoped, backend-declared; a backend without
        # endmember metadata (legacy fake in a test) maps each channel to
        # itself with its registry formula — same rule as the stoich backend
        if hydrate_endmembers is None:
            hydrate_endmembers = {h: (h,) for h in hydrate_ids}
        if endmember_elements is None:
            multi = [h for h in hydrate_ids
                     if tuple(hydrate_endmembers[h]) != (h,)]
            if multi:
                raise ValueError(
                    f"backend declares non-trivial endmember universes for "
                    f"{multi} but no endmember element rows — the registry "
                    f"formula fallback only covers channel==endmember "
                    f"(no guessing)")
            endmember_elements = {
                h: formula_elements(registry.get(h).formula)
                for h in hydrate_ids}
        em_ids = tuple((h, dc) for h in hydrate_ids
                       for dc in hydrate_endmembers[h])
        em_formulas = np.array([endmember_elements[dc] for _, dc in em_ids]) \
            if em_ids else np.zeros((0, len(ELEMENT_IDS)))

        rng = np.random.Generator(np.random.PCG64(config.rve.seed))
        return cls(
            config=config,
            hydrate_ids=tuple(hydrate_ids),
            anhydrous_fraction=rve.anhydrous_fraction.copy(),
            hydrate_fraction=np.zeros((len(hydrate_ids), n, n, n)),
            capillary_liquid=rve.capillary_liquid.copy(),
            capillary_gas=rve.capillary_gas.copy(),
            particle_id=rve.particle_id.copy(),
            cluster_id=np.full((n, n, n), -1, dtype=np.int64),
            particles={k: v.copy() for k, v in rve.particles.items()},
            subgrid_bins={k: v.copy() for k, v in rve.subgrid_bins.items()},
            parcels={"time_h": [], "cluster": [], "hydrate": [], "mol": [],
                     "skel_vol_vox": [], "bulk_vol_vox": []},
            remap_events={"time_h": [], "prev": [], "new": [], "overlap_vox": []},
            cluster_inventory=np.zeros((0, len(ELEMENT_IDS))),
            cluster_endmember_mol=np.zeros((0, len(em_ids))),
            time_h=0.0,
            dt_h=config.schedule.dt_initial_h,
            phase_mol=phase_mol,
            initial_phase_mol=phase_mol.copy(),
            unmet_mol=np.zeros(len(KINETIC_PHASE_IDS)),
            hydrate_mol=np.zeros(len(hydrate_ids)),
            hydrate_env_vol_vox=np.zeros(len(hydrate_ids)),
            hydrate_elements_ch=np.zeros((len(hydrate_ids), len(ELEMENT_IDS))),
            endmember_ids=em_ids,
            endmember_mol=np.zeros(len(em_ids)),
            endmember_elements=em_formulas,
            boundary_exchanged_elements=np.zeros(len(ELEMENT_IDS)),
            injected_elements=np.zeros(len(ELEMENT_IDS)),
            water_free_mol=water_mol,
            water_gel_mol=0.0,
            water_bound_mol=0.0,
            initial_water_mol=water_mol,
            boundary_water_mol=0.0,
            inert_volume_vox=inert_vol,
            initial_elements=initial_elements,
            accept_count=0,
            reject_counts={},
            rng_state=rng.bit_generator.state,
            config_hash=config.config_hash(),
            backend_id=backend_id,
        )
