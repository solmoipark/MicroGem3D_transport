"""Liquid cluster labeling (6-neighbor, periodic) + conservative overlap
remapping + the v4.0/RT domain machinery (PRD 2.1/4.6.2): sub-cluster
equilibration domains from a static tiling, the shared cell-conductance rule,
and the implicit (backward-Euler) solute exchange operator on the domain
graph in flux form.

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
# transport-frozen dust threshold: a domain whose liquid volume is below this
# fraction of the mean wet-domain volume gets no edges (mirrors the engine's
# trace-water chemistry freeze and bounds the BE matrix contrast)
DUST_WATER_REL = 1e-6


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


# --------------------------------------------------------------- v4.0/RT ---

def conductance_field(capillary_liquid: np.ndarray,
                      hydrate_fraction: np.ndarray,
                      gel_eps: np.ndarray,
                      gel_rel_diffusivity: float) -> np.ndarray:
    """Shared cell-conductance rule (PRD 1.3/4.6.2): capillary liquid
    conducts at 1, gel-bearing hydrate volume at gel_rel_diffusivity.
    NO floor here - the report's NETWORK_FLOOR is a CG-conditioning and
    reporting device that stays analysis-side; carried into transport
    physics it would leak solutes through sealed solid at 1e-8."""
    g = capillary_liquid.astype(np.float64).copy()
    for i in range(hydrate_fraction.shape[0]):
        if gel_eps[i] > 0.0:
            g += gel_rel_diffusivity * hydrate_fraction[i]
    return g


def label_domains(labels: np.ndarray, n_clusters: int, tile_vox: int
                  ) -> Tuple[np.ndarray, int, np.ndarray]:
    """Equilibration domains = connected cluster INTERSECTED with a static
    axis-aligned tiling (PRD 4.6.2). tile_vox == grid size returns the
    cluster labels themselves - the literal legacy identity the 1-domain
    bit-compare gate relies on. Ids are consecutive, ordered by
    (cluster id, tile index): deterministic, restart-stable, periodic-aware
    (tiles wrap because the grid size is a multiple of tile_vox).

    Returns (domain_id with -1 dry, n_domains, domain_to_cluster)."""
    n = labels.shape[0]
    if tile_vox == n:
        return labels, n_clusters, np.arange(n_clusters, dtype=np.int64)
    if n % tile_vox != 0:
        raise ValueError(f"tile_vox {tile_vox} does not divide grid {n}")
    t = n // tile_vox
    az = np.arange(n, dtype=np.int64) // tile_vox
    tile = (az[:, None, None] * t + az[None, :, None]) * t + az[None, None, :]
    n_tiles = t * t * t
    wet = labels >= 0
    key = np.where(wet, labels * n_tiles + tile, np.int64(-1))
    uniq = np.unique(key[wet]) if wet.any() else np.empty(0, dtype=np.int64)
    domain_id = np.full(labels.shape, -1, dtype=np.int64)
    if uniq.size:
        domain_id[wet] = np.searchsorted(uniq, key[wet])
    return domain_id, int(uniq.size), (uniq // n_tiles).astype(np.int64)


@dataclass
class DomainGraph:
    """Face-adjacency graph of wet domains. edge_g carries the summed
    harmonic face conductances of each interface; water is the liquid
    volume per domain (vox^3). Dust domains (water below DUST_WATER_REL of
    the mean wet-domain water) carry no edges - transport-frozen."""
    n_domains: int
    edge_a: np.ndarray          # (E,) int64, edge_a < edge_b
    edge_b: np.ndarray          # (E,) int64
    edge_g: np.ndarray          # (E,) float64
    water: np.ndarray           # (D,) float64
    dust: np.ndarray            # (D,) bool


def build_domain_graph(domain_id: np.ndarray, n_domains: int,
                       g: np.ndarray, liquid: np.ndarray) -> DomainGraph:
    """Edges between face-adjacent wet voxels of DIFFERENT domains (always
    domains of the same cluster, by construction of the 6-neighbor flood).
    Face conductance is the harmonic mean 2ab/(a+b) - the same rule as the
    report's diffusivity network - summed over the interface. Fully
    periodic (wrap faces included); deterministic edge order."""
    water = np.zeros(n_domains)
    wet = domain_id >= 0
    if wet.any():
        np.add.at(water, domain_id[wet], liquid[wet])
    wp = water[water > 0.0]
    dust_floor = DUST_WATER_REL * float(wp.mean()) if wp.size else 0.0
    dust = water < dust_floor
    keys = []
    conds = []
    for ax in range(3):
        da = domain_id
        db = np.roll(domain_id, -1, axis=ax)
        ga = g
        gb = np.roll(g, -1, axis=ax)
        m = (da >= 0) & (db >= 0) & (da != db)
        if not m.any():
            continue
        a = da[m]
        b = db[m]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keep = ~(dust[lo] | dust[hi])
        if not keep.any():
            continue
        lo, hi = lo[keep], hi[keep]
        fa = ga[m][keep]
        fb = gb[m][keep]
        s = fa + fb
        face = np.where(s > 0.0, 2.0 * fa * fb / np.where(s > 0.0, s, 1.0),
                        0.0)
        keys.append(lo * np.int64(n_domains) + hi)
        conds.append(face)
    if keys:
        key = np.concatenate(keys)
        cond = np.concatenate(conds)
        order = np.argsort(key, kind="stable")
        key, cond = key[order], cond[order]
        uniq, start = np.unique(key, return_index=True)
        sums = np.add.reduceat(cond, start)
        edge_a = (uniq // n_domains).astype(np.int64)
        edge_b = (uniq % n_domains).astype(np.int64)
    else:
        edge_a = np.empty(0, dtype=np.int64)
        edge_b = np.empty(0, dtype=np.int64)
        sums = np.empty(0)
    return DomainGraph(n_domains=n_domains, edge_a=edge_a, edge_b=edge_b,
                       edge_g=sums, water=water, dust=dust)


@dataclass
class ExchangeResult:
    delta: np.ndarray            # (D, E) antisymmetric inventory change
    status: str                  # "ok" | "not_converged"
    cg_iterations: int
    max_edge_flux_mol: float
    repair_rel: float            # largest relative negative-dust repair


def exchange_be(graph: DomainGraph, inventory: np.ndarray, dt_h: float,
                d0_vox2_h: float, pitch_vox: float,
                tol: float = 1e-12, max_iter: int = 10000) -> ExchangeResult:
    """One backward-Euler diffusion step on the domain graph, in FLUX FORM
    (PRD 4.6.2): solve (diag(W) + dt L) c = n per element column with
    Jacobi-preconditioned CG (fixed operation order - deterministic), then
    apply per-edge fluxes F = dt T (c_a - c_b) antisymmetrically, so the
    same float leaves one domain and enters the other and element closure
    is exact by construction. The M-matrix guarantees positivity up to CG
    dust; a deterministic repair pass (morphology.remove pattern) clips it
    and charges the largest entry, asserting the magnitude stays dust."""
    n_dom, n_elem = inventory.shape
    delta = np.zeros_like(inventory)
    if graph.edge_a.size == 0 or dt_h <= 0.0:
        return ExchangeResult(delta, "ok", 0, 0.0, 0.0)
    ea, eb = graph.edge_a, graph.edge_b
    t_e = d0_vox2_h * graph.edge_g / pitch_vox      # vox^3 / h
    # only edge-connected domains participate; the diagonal floors at a tiny
    # positive value on participating rows to stay invertible
    act = np.zeros(n_dom, dtype=bool)
    act[ea] = True
    act[eb] = True
    w_act = np.where(act, np.maximum(graph.water, 1e-300), 1.0)
    deg = np.zeros(n_dom)
    np.add.at(deg, ea, t_e)
    np.add.at(deg, eb, t_e)
    diag = w_act + dt_h * deg
    b = inventory * act[:, None]

    def matvec(x):
        y = w_act[:, None] * x
        d = x[ea] - x[eb]
        contrib = dt_h * t_e[:, None] * d
        np.add.at(y, ea, contrib)
        np.add.at(y, eb, -contrib)
        return y

    x = b / diag[:, None]
    r = b - matvec(x)
    z = r / diag[:, None]
    p = z.copy()
    rz = (r * z).sum(axis=0)
    bnorm = np.maximum(np.abs(b).sum(axis=0), 1e-300)
    iters = 0
    for iters in range(1, max_iter + 1):
        if np.all(np.abs(r).sum(axis=0) <= tol * bnorm):
            break
        ap = matvec(p)
        pap = (p * ap).sum(axis=0)
        alpha = np.where(pap > 0.0, rz / np.where(pap > 0.0, pap, 1.0), 0.0)
        x = x + alpha[None, :] * p
        r = r - alpha[None, :] * ap
        z = r / diag[:, None]
        rz_new = (r * z).sum(axis=0)
        beta = np.where(rz > 0.0, rz_new / np.where(rz > 0.0, rz, 1.0), 0.0)
        p = z + beta[None, :] * p
        rz = rz_new
    else:
        return ExchangeResult(delta, "not_converged", iters, 0.0, 0.0)

    flux = dt_h * t_e[:, None] * (x[ea] - x[eb])    # (n_edges, n_elem)
    np.add.at(delta, ea, -flux)
    np.add.at(delta, eb, flux)
    max_flux = float(np.abs(flux).max()) if flux.size else 0.0

    # deterministic negative-dust repair: clip, charge the largest entry
    new = inventory + delta
    repair_rel = 0.0
    for e in range(n_elem):
        col = new[:, e]
        neg = col < 0.0
        if not neg.any():
            continue
        shortfall = float(col[neg].sum())           # < 0
        scale = float(np.abs(inventory[:, e]).max())
        if scale > 0.0:
            repair_rel = max(repair_rel, -shortfall / scale)
        col[neg] = 0.0
        top = int(np.argmax(col))
        col[top] = max(col[top] + shortfall, 0.0)
        delta[:, e] = col - inventory[:, e]
    if repair_rel > 1e-11:
        raise RuntimeError(
            f"exchange_be negative repair {repair_rel:.3e} exceeds dust - "
            f"logic error, not absorbable")
    return ExchangeResult(delta, "ok", iters, max_flux, repair_rel)
