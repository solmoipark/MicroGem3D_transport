"""Hydrate bulk-envelope placement: inner (vacated dissolution space) first, then
outer (displacing capillary liquid) over periodic 6-neighbor BFS shells around each
source voxel. Volume is never dropped or teleported — anything unplaceable within
the shell radius is a placement_capacity reject (PRD §2.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

STATUS_OK = "ok"
STATUS_CAPACITY = "placement_capacity"
_MAX_SHELL = 3  # Manhattan radius searched around a source voxel


def _shell_offsets() -> List[Tuple[int, int, int, int]]:
    offs = []
    r = _MAX_SHELL
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = abs(dz) + abs(dy) + abs(dx)
                if 0 < d <= r:
                    offs.append((d, dz, dy, dx))
    offs.sort()
    return offs

_OFFSETS = _shell_offsets()


@dataclass
class PlacementOutcome:
    status: str
    placed_vol_vox: float
    unplaced_vol_vox: float
    liquid_displaced_vol_vox: float


def place(hydrate_fraction: np.ndarray, capillary_liquid: np.ndarray,
          vacated: np.ndarray, demand_vol: np.ndarray,
          cell_cluster: np.ndarray, source_cluster: np.ndarray) -> PlacementOutcome:
    """Place demand_vol (H, N, N, N) bulk-envelope volume, mutating hydrate_fraction,
    capillary_liquid, and vacated in place. Capacity (vacated space or displaceable
    liquid) may only be taken from cells of the source's own connected cluster —
    displaced water must have somewhere to go, and cross-cluster vacancy use would
    break per-cluster water reconciliation. Leftover vacated space is refilled with
    liquid by the caller, which owns the reconciliation."""
    n = capillary_liquid.shape[0]
    n_h = demand_vol.shape[0]
    total_demand = demand_vol.sum(axis=0)
    sources = np.flatnonzero(total_demand.ravel() > 0.0)
    placed = 0.0
    displaced = 0.0
    unplaced = 0.0

    for flat in sources:
        z, y, x = np.unravel_index(flat, (n, n, n))
        c_src = source_cluster[z, y, x]
        remaining = demand_vol[:, z, y, x].copy()
        rem_sum = float(remaining.sum())

        def _cells(z=int(z), y=int(y), x=int(x)):
            # shells are generated lazily: most sources finish in their own
            # voxel and never pay for the 63-cell neighborhood
            yield z, y, x
            for _, dz, dy, dx in _OFFSETS:
                yield (z + dz) % n, (y + dy) % n, (x + dx) % n

        for (cz, cy, cx) in _cells():
            if rem_sum <= 0.0:
                break
            if cell_cluster[cz, cy, cx] != c_src:
                continue
            cap = vacated[cz, cy, cx] + capillary_liquid[cz, cy, cx]
            if cap <= 0.0:
                continue
            if cap >= rem_sum - 1e-15 * max(1.0, rem_sum):
                # final cell: place the exact remainder so no float dust is
                # ever dropped (a ~1e-16 capacity overdraw stays within the
                # voxel identity tolerance)
                take = rem_sum
                part = remaining.copy()
                remaining[:] = 0.0
                rem_sum = 0.0
            else:
                take = cap
                part = remaining * (take / rem_sum)
                remaining -= part
                rem_sum = float(remaining.sum())
            for h in range(n_h):
                hydrate_fraction[h, cz, cy, cx] += part[h]
            use_vac = min(take, vacated[cz, cy, cx])
            vacated[cz, cy, cx] -= use_vac
            from_liquid = take - use_vac
            capillary_liquid[cz, cy, cx] -= from_liquid
            displaced += from_liquid
            placed += take
        # anything left is a genuine capacity shortfall — volume is never dropped
        unplaced += rem_sum

    status = STATUS_OK if unplaced == 0.0 else STATUS_CAPACITY
    return PlacementOutcome(status=status, placed_vol_vox=placed,
                            unplaced_vol_vox=unplaced,
                            liquid_displaced_vol_vox=displaced)
