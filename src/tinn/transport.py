"""Liquid cluster labeling (6-neighbor, periodic) + conservative overlap remapping.

Labeling is deterministic: each cluster gets consecutive ids ordered by its
smallest flat voxel index. Remapping distributes previous cluster inventories to
new clusters proportionally to physical liquid overlap volume (merge and split
both supported); a cluster holding inventory with zero overlap is a dryout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

LIQ_EPS = 1e-9
_AXES = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))


def label_clusters(liquid: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return (labels int64 with -1 for dry voxels, cluster count)."""
    mask = liquid > LIQ_EPS
    n_vox = liquid.size
    labels = np.where(mask, np.arange(n_vox, dtype=np.int64).reshape(liquid.shape),
                      np.int64(n_vox))
    # min-label flooding advances >= 1 voxel per sweep along the longest geodesic,
    # which is bounded by the voxel count — never bail out on a legal topology
    for _ in range(n_vox + 1):
        prev = labels
        m = labels
        for ax, shift in _AXES:
            r = np.roll(labels, shift, axis=ax)
            m = np.minimum(m, np.where(mask, r, n_vox))
        labels = np.where(mask, np.minimum(labels, m), np.int64(n_vox))
        if np.array_equal(labels, prev):
            break
    else:
        raise RuntimeError("cluster labeling did not converge")
    roots = np.unique(labels[mask]) if mask.any() else np.empty(0, dtype=np.int64)
    remap = np.full(n_vox + 1, -1, dtype=np.int64)
    remap[roots] = np.arange(len(roots), dtype=np.int64)
    return remap[labels], len(roots)


@dataclass
class RemapResult:
    inventory: np.ndarray            # (K_new, E)
    events: List[Tuple[int, int, float]]  # (prev, new, overlap_vox)
    dryout: List[int]                # prev cluster ids with inventory but no overlap


def remap_inventories(prev_labels: np.ndarray, prev_liquid: np.ndarray,
                      new_labels: np.ndarray, new_liquid: np.ndarray,
                      prev_inventory: np.ndarray, n_new: int) -> RemapResult:
    n_prev = prev_inventory.shape[0]
    n_elem = prev_inventory.shape[1] if prev_inventory.ndim == 2 else 0
    inventory = np.zeros((n_new, n_elem))
    both = (prev_labels >= 0) & (new_labels >= 0)
    w = np.minimum(prev_liquid, new_liquid)[both]
    a = prev_labels[both]
    b = new_labels[both]
    keep = w > 0.0
    a, b, w = a[keep], b[keep], w[keep]
    events: List[Tuple[int, int, float]] = []
    dryout: List[int] = []
    if n_prev == 0:
        return RemapResult(inventory, events, dryout)

    keys = a * np.int64(n_new) + b
    order = np.argsort(keys, kind="stable")
    keys, w = keys[order], w[order]
    uniq, start = np.unique(keys, return_index=True)
    sums = np.add.reduceat(w, start) if len(w) else np.empty(0)
    ov_a = (uniq // n_new).astype(np.int64)
    ov_b = (uniq % n_new).astype(np.int64)

    row_total = np.zeros(n_prev)
    np.add.at(row_total, ov_a, sums)
    for pa, pb, s in zip(ov_a, ov_b, sums):
        frac = s / row_total[pa]
        inventory[pb] += prev_inventory[pa] * frac
        events.append((int(pa), int(pb), float(s)))
    for pa in range(n_prev):
        if row_total[pa] == 0.0 and np.any(prev_inventory[pa] != 0.0):
            dryout.append(pa)
    return RemapResult(inventory, events, dryout)
