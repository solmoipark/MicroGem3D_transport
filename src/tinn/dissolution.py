"""Per-phase dissolution allocation over liquid-accessible sites (PRD §4.2).

Weights are the voxel's wetted-face count (own wetness + number of 6-neighbor
periodic voxels holding cluster liquid) — a geometric surface-area measure,
relative only, never an absolute rate. Exhausted sites are re-allocated
iteratively; anything unachievable is returned as unmet mol, never hidden.
The weight needs no particle attribution, so it is well-defined for all three
solid tiers including smeared subgrid volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import KINETIC_PHASE_IDS, Registry, SOLID_PHASE_IDS
from .state import SimulationState
from .transport import LIQ_EPS  # single threshold shared with cluster labeling

_AXES = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def neighbor_liquid_sum(liquid: np.ndarray) -> np.ndarray:
    s = np.zeros_like(liquid)
    for ax, shift in _AXES:
        s += np.roll(liquid, shift, axis=ax)
    return s


@dataclass
class DissolutionResult:
    removed_vol: np.ndarray   # (4, N, N, N) volume removed per kinetic phase
    removed_mol: np.ndarray   # (4,)
    unmet_mol: np.ndarray     # (4,)


def dissolve(trial: SimulationState, registry: Registry,
             dn_target_mol: np.ndarray) -> DissolutionResult:
    """Remove dn_target_mol per kinetic phase from trial.anhydrous_fraction in place."""
    n = trial.grid_size
    # wetted-face count from cluster liquid only (same threshold as labeling),
    # so every site is guaranteed an adjacent labeled cluster
    wet = (trial.capillary_liquid > LIQ_EPS).astype(np.float64)
    weight = wet + neighbor_liquid_sum(wet)
    removed_vol = np.zeros((len(KINETIC_PHASE_IDS), n, n, n))
    removed_mol = np.zeros(len(KINETIC_PHASE_IDS))
    unmet_mol = np.zeros(len(KINETIC_PHASE_IDS))

    for k, p in enumerate(KINETIC_PHASE_IDS):
        vm = trial.vm_vox(registry, p)
        target_vol = dn_target_mol[k] * vm
        if target_vol <= 0.0:
            continue
        chan = SOLID_PHASE_IDS.index(p)
        avail = trial.anhydrous_fraction[chan]
        sites = (avail > 0.0) & (weight > 0.0)
        alloc = np.zeros_like(avail)
        remaining = target_vol
        for _ in range(64):
            idx = sites & (avail - alloc > 0.0)
            if remaining <= target_vol * 1e-15 or not idx.any():
                break
            w = np.where(idx, weight, 0.0)
            share = remaining * w / w.sum()
            take = np.minimum(share, avail - alloc)
            alloc += take
            remaining -= float(take.sum())
        trial.anhydrous_fraction[chan] -= alloc
        removed_vol[k] = alloc
        removed_mol[k] = float(alloc.sum()) / vm
        unmet_mol[k] = max(0.0, remaining) / vm

    return DissolutionResult(removed_vol, removed_mol, unmet_mol)
