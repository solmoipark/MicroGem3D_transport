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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import TinnConfig
from .geometry import RVEInit
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, INERT_PHASE_ID,
                       KINETIC_PHASE_IDS, Registry)

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
    v = np.zeros(len(ELEMENT_IDS))
    for el, count in formula.items():
        v[ELEMENT_IDS.index(el)] += count
    return v


@dataclass
class SimulationState:
    config: TinnConfig
    # --- dense fields, (z, y, x), float64 / int64 ---
    anhydrous_fraction: np.ndarray   # (5, N, N, N), channels = SOLID_PHASE_IDS
    hydrate_fraction: np.ndarray     # (4, N, N, N) bulk envelope incl. gel pores
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
    # --- ledger / header (mol is authoritative) ---
    time_h: float
    dt_h: float
    phase_mol: np.ndarray            # (4,) remaining anhydrous mol
    initial_phase_mol: np.ndarray    # (4,)
    unmet_mol: np.ndarray            # (4,) current deficit vs kinetic target
    hydrate_mol: np.ndarray          # (4,) channels = HYDRATE_PHASE_IDS
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
        h = hashlib.sha256(self.dense_hash().encode())
        for arr in (self.phase_mol, self.initial_phase_mol, self.unmet_mol,
                    self.hydrate_mol, self.initial_elements, self.cluster_inventory):
            h.update(np.ascontiguousarray(arr).tobytes())
        for x in (self.time_h, self.dt_h, self.water_free_mol, self.water_gel_mol,
                  self.water_bound_mol, self.initial_water_mol,
                  self.inert_volume_vox, self.accept_count):
            h.update(repr(x).encode())
        h.update(repr(sorted(self.reject_counts.items())).encode())
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
                      rve: RVEInit, backend_id: str) -> "SimulationState":
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

        rng = np.random.Generator(np.random.PCG64(config.rve.seed))
        return cls(
            config=config,
            anhydrous_fraction=rve.anhydrous_fraction.copy(),
            hydrate_fraction=np.zeros((len(HYDRATE_PHASE_IDS), n, n, n)),
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
            time_h=0.0,
            dt_h=config.schedule.dt_initial_h,
            phase_mol=phase_mol,
            initial_phase_mol=phase_mol.copy(),
            unmet_mol=np.zeros(len(KINETIC_PHASE_IDS)),
            hydrate_mol=np.zeros(len(HYDRATE_PHASE_IDS)),
            water_free_mol=water_mol,
            water_gel_mol=0.0,
            water_bound_mol=0.0,
            initial_water_mol=water_mol,
            inert_volume_vox=inert_vol,
            initial_elements=initial_elements,
            accept_count=0,
            reject_counts={},
            rng_state=rng.bit_generator.state,
            config_hash=config.config_hash(),
            backend_id=backend_id,
        )
