"""Per-phase dissolution allocation over water-accessible sites (PRD §4.2).

Weights are the voxel's conductive-face count: own/6-neighbor faces holding
cluster liquid count 1 each, faces holding hydrate count GEL_FACE_WEIGHT each
(gel pore water is the ion conduit through a coating — P&K's diffusion stage
already encodes the shell resistance empirically, so a hard geometric gate on
liquid contact would double-count it; PRD §4.2, v2.2 rev.2). The weight is a
geometric surface-area measure, relative only, never an absolute rate.
Exhausted sites are re-allocated iteratively; anything unachievable is
returned as unmet mol, never hidden. The weight needs no particle
attribution, so it is well-defined for all three solid tiers including
smeared subgrid volume. Particle-interior voxels (neither liquid nor hydrate
anywhere on their faces) stay inaccessible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import KINETIC_PHASE_IDS, Registry, SOLID_PHASE_IDS
from .state import SimulationState
from .transport import LIQ_EPS  # single threshold shared with cluster labeling

_AXES = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))

# relative conductivity of a hydrate-bearing face vs a liquid-bearing face:
# the representative C-S-H gel porosity (the water fraction of the conduit)
GEL_FACE_WEIGHT = 0.28


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


def conductive_weight(trial: SimulationState) -> np.ndarray:
    """Per-voxel conductive-face count (the accessibility measure this module
    allocates by). Own/neighbor faces holding cluster liquid count 1, faces
    holding hydrate count GEL_FACE_WEIGHT. Zero means the voxel is sealed
    inside a particle and cannot dissolve at all.

    Public because the engine needs the SAME measure to decide how much of a
    solubility-controlled carrier is water-accessible this step (PRD 4.2
    v3.0/E3): those phases have no rate law, so their offer is "everything the
    water can currently reach" and the equilibrium decides what stays solid."""
    wet = (trial.capillary_liquid > LIQ_EPS).astype(np.float64)
    gel = (trial.hydrate_fraction.sum(axis=0) > LIQ_EPS).astype(np.float64)
    return (wet + neighbor_liquid_sum(wet)
            + GEL_FACE_WEIGHT * (gel + neighbor_liquid_sum(gel)))


def accessible_mol(trial: SimulationState, registry: Registry,
                   phase_ids) -> np.ndarray:
    """Mol of each named phase that water can currently reach, in
    KINETIC_PHASE_IDS positions (zero elsewhere)."""
    weight = conductive_weight(trial)
    out = np.zeros(len(KINETIC_PHASE_IDS))
    for p in phase_ids:
        k = KINETIC_PHASE_IDS.index(p)
        chan = SOLID_PHASE_IDS.index(p)
        avail = trial.anhydrous_fraction[chan]
        vol = float(np.where(weight > 0.0, avail, 0.0).sum())
        if vol > 0.0:
            out[k] = vol / trial.vm_vox(registry, p)
    return out


def dissolve(trial: SimulationState, registry: Registry,
             dn_target_mol: np.ndarray) -> DissolutionResult:
    """Remove dn_target_mol per kinetic phase from trial.anhydrous_fraction in place."""
    n = trial.grid_size
    # conductive-face count: cluster liquid (same threshold as labeling) plus
    # hydrate gel faces at GEL_FACE_WEIGHT — a coated site reaches its cluster
    # through the gel, so attribution may need multi-hop (engine handles it)
    weight = conductive_weight(trial)
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
