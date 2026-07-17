"""Hydrate bulk-envelope placement: inner (vacated dissolution space) first, then
outer (displacing capillary liquid) over periodic 6-neighbor BFS shells around each
source voxel. Volume is never dropped or teleported — anything unplaceable within
the shell radius overflows into the source cluster's remaining pore capacity
(through-solution precipitation within the same connected liquid); only a
cluster-wide shortfall is a placement_capacity reject (PRD §2.2).
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


def remove(hydrate_fraction: np.ndarray, channel: int, request_vol: float,
           member_mask: np.ndarray) -> tuple:
    """Remove request_vol of one hydrate channel over the member voxels,
    proportional to local volume with a deterministic flat-index residual
    correction so removed == request exactly (v1 reversible pattern).
    Returns (removal_field (N,N,N), removed_vol). Over-request is a hard error
    — volume is never fabricated."""
    field = hydrate_fraction[channel]
    local = np.where(member_mask, field, 0.0)
    available = float(local.sum())
    if request_vol > available * (1.0 + 1e-12) + 1e-30:
        raise ValueError(
            f"removal request {request_vol!r} exceeds available {available!r} "
            f"on channel {channel}")
    # dust-sized over-request (bincount vs pairwise summation) clamps to the
    # available volume — otherwise the proportional ratio exceeds 1 and the
    # residual pass would take negative amounts / drive voxels negative
    request_vol = min(request_vol, available)
    if request_vol <= 0.0 or available <= 0.0:
        return np.zeros_like(field), 0.0
    removal = local * (request_vol / available)
    # exact-residual pass: fix float dust deterministically in flat-index order
    residual = request_vol - float(removal.sum())
    if residual != 0.0:
        flat = removal.ravel()
        loc = local.ravel()
        for j in np.flatnonzero(loc > 0.0):
            room = loc[j] - flat[j]
            take = min(residual, room) if residual > 0.0 else max(residual, -flat[j])
            flat[j] += take
            residual -= take
            if residual == 0.0:
                break
    hydrate_fraction[channel] -= removal
    return removal, float(removal.sum())


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
    leftovers: dict = {}

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
        # anything left overflows the shell radius — retried cluster-wide below
        if rem_sum > 0.0:
            acc = leftovers.setdefault(int(c_src), np.zeros(n_h))
            acc += remaining

    # overflow pass: a dense late-age neighborhood can lock all capacity out of
    # the shell radius while the cluster still has ample pore space. The excess
    # precipitates through solution anywhere in the SAME connected liquid,
    # spread in proportion to local remaining capacity (exact-residual
    # corrected). Only a cluster-wide shortfall is a capacity reject.
    for c_src in sorted(leftovers):
        remaining = leftovers[c_src]
        rem_sum = float(remaining.sum())
        member = cell_cluster == c_src
        # placement float dust can leave ~-1e-16 liquid in a voxel — clipping
        # keeps such voxels out of the capacity pool so the proportional take
        # never goes negative there (rev.2 review finding)
        cap_field = np.where(member,
                             np.clip(vacated + capillary_liquid, 0.0, None), 0.0)
        total_cap = float(cap_field.sum())
        # dust tolerance mirrors remove(): a near-exact fit (engine freeze
        # gate and this gate disagree only at ulp level) proceeds with a
        # sanctioned dust overdraw instead of aborting the run
        if total_cap <= 0.0 or rem_sum > total_cap * (1.0 + 1e-12) + 1e-30:
            unplaced += rem_sum  # genuine shortfall — volume is never dropped
            continue
        take_field = cap_field * (rem_sum / total_cap)
        residual = rem_sum - float(take_field.sum())
        if residual != 0.0:
            # dust-scale by construction (float summation error of the ratio
            # multiply): book it at the largest-capacity voxel, deterministic,
            # same sanctioned-overdraw rule as the main loop's final cell
            flat_t = take_field.ravel()
            flat_t[int(np.argmax(cap_field.ravel()))] += residual
        frac_h = remaining / rem_sum
        for h in range(n_h):
            hydrate_fraction[h] += frac_h[h] * take_field
        use_vac = np.minimum(take_field, vacated)
        vacated -= use_vac
        from_liquid = take_field - use_vac
        capillary_liquid -= from_liquid
        displaced += float(from_liquid.sum())
        placed += rem_sum

    status = STATUS_OK if unplaced == 0.0 else STATUS_CAPACITY
    return PlacementOutcome(status=status, placed_vol_vox=placed,
                            unplaced_vol_vox=unplaced,
                            liquid_displaced_vol_vox=displaced)
