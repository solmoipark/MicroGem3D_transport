"""Periodic RVE initialization: particle placement, 3-tier solid partition, initial saturation.

Tiers (radius rv in voxel units, rv = d / 2h):
  resolved   rv >  RV_FRACTIONAL_MAX : rasterized onto multiple voxels (exact-volume boundary)
  fractional RV_SUBGRID_MAX <= rv <= RV_FRACTIONAL_MAX : single-voxel partial occupancy
  subgrid    rv <  RV_SUBGRID_MAX : bin-level volume spread proportional to free capacity

All clinker particles share the recipe's phase composition (uniform multiphase clinker),
so per-phase dense arrays are total occupancy times fixed phase volume weights.
Every deviation from targets (sampling residue, unplaced particles, raster deficit) is
reported, never hidden.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import TinnConfig
from .registry import INERT_PHASE_ID, Registry, SOLID_PHASE_IDS

RV_SUBGRID_MAX = 0.25  # d < h/2
RV_FRACTIONAL_MAX = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)  # sphere volume <= 1 voxel
_HALF_DIAG = math.sqrt(3.0) / 2.0
_CAPACITY_MARGIN = 1e-9
_MAX_PLACEMENT_TRIES = 500
_SUBSAMPLES = 4

TIER_RESOLVED, TIER_FRACTIONAL, TIER_SUBGRID = 0, 1, 2


class GeometryError(RuntimeError):
    pass


@dataclass
class RVEInit:
    phase_ids: Tuple[str, ...]
    anhydrous_fraction: np.ndarray  # (P, N, N, N) float64, z,y,x order
    capillary_liquid: np.ndarray    # (N, N, N) float64
    capillary_gas: np.ndarray       # (N, N, N) float64
    particle_id: np.ndarray         # (N, N, N) int64, -1 = none
    particles: Dict[str, np.ndarray]
    subgrid_bins: Dict[str, np.ndarray]
    report: Dict[str, float]

    def dense_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.anhydrous_fraction, self.capillary_liquid,
                    self.capillary_gas, self.particle_id):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()


def _sphere_volume(rv: float) -> float:
    return (4.0 / 3.0) * math.pi * rv ** 3


def _waterfill(v: np.ndarray, target: float, cap: float) -> np.ndarray:
    """Scale boundary fractions v so their sum equals target, each capped at cap."""
    w = np.zeros_like(v)
    free = v > 0.0
    for _ in range(64):
        remaining = target - w[~free].sum()
        if remaining <= 0.0 or not free.any():
            break
        s = remaining / v[free].sum()
        cand = s * v[free]
        over = cand > cap
        if not over.any():
            w[free] = cand
            return w
        idx = np.flatnonzero(free)[over]
        w[idx] = cap
        free[idx] = False
    return w


def _rasterize_sphere(center: np.ndarray, rv: float, n: int
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Periodic sphere rasterization with exact analytic volume on the boundary shell.

    Returns (flat wrapped voxel indices, fractions, volume deficit)."""
    lo = np.floor(center - rv - _HALF_DIAG).astype(np.int64)
    hi = np.floor(center + rv + _HALF_DIAG).astype(np.int64) + 1
    if np.any(hi - lo > n):
        raise GeometryError(f"particle with rv={rv} voxels does not fit the periodic RVE")
    axes = [np.arange(lo[a], hi[a]) for a in range(3)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    idx = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    dist = np.linalg.norm(idx + 0.5 - center, axis=1)

    frac = np.zeros(len(idx))
    frac[dist <= rv - _HALF_DIAG] = 1.0
    bmask = (dist > rv - _HALF_DIAG) & (dist < rv + _HALF_DIAG)
    if bmask.any():
        off = (np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES
        oz, oy, ox = np.meshgrid(off, off, off, indexing="ij")
        offs = np.stack([oz, oy, ox], axis=-1).reshape(-1, 3)
        pts = idx[bmask][:, None, :] + offs[None, :, :]
        inside = ((pts - center) ** 2).sum(axis=2) <= rv * rv
        raw = inside.mean(axis=1)
        boundary_target = _sphere_volume(rv) - float((frac == 1.0).sum())
        filled = _waterfill(raw, boundary_target, cap=1.0 - _CAPACITY_MARGIN)
        frac[bmask] = filled
        deficit = boundary_target - float(filled.sum())
    else:
        deficit = _sphere_volume(rv) - float(frac.sum())

    keep = frac > 0.0
    idx, frac = idx[keep], frac[keep]
    wrapped = ((idx[:, 0] % n) * n + (idx[:, 1] % n)) * n + (idx[:, 2] % n)
    return wrapped, frac, deficit


def initialize_rve(config: TinnConfig, registry: Registry) -> RVEInit:
    n = config.rve.grid_size
    h = config.rve.voxel_size_um
    rng = np.random.Generator(np.random.PCG64(config.rve.seed))

    # --- targets (per 1 g binder; only ratios matter) ---
    masses = dict(config.binder.mass_fractions)
    masses[INERT_PHASE_ID] = config.binder.unassigned
    vol_g = {p: masses.get(p, 0.0) / registry.get(p).density_g_cm3 for p in SOLID_PHASE_IDS}
    v_solid = sum(vol_g.values())
    rho_w = registry.get("H2O").density_g_cm3
    v_water = config.w_c / rho_w
    phi_target = v_solid / (v_solid + v_water)
    phase_weights = np.array([vol_g[p] / v_solid for p in SOLID_PHASE_IDS])

    n_vox = n ** 3
    v_solid_target = phi_target * n_vox  # in voxel-volume units

    # --- sample particle sizes per PSD bin (coarse -> fine, carrying residual) ---
    diameters_rv = []
    subgrid_rows = []  # (d_lo_um, d_hi_um, volume_vox, number_est)
    carry = 0.0
    for b in reversed(config.psd.bins):
        target = v_solid_target * b.volume_fraction + carry
        if target <= 0.0:
            carry = target
            continue
        rv_lo, rv_hi = b.d_lo_um / (2 * h), b.d_hi_um / (2 * h)
        if rv_hi <= RV_SUBGRID_MAX:
            d_mean = math.sqrt(b.d_lo_um * b.d_hi_um)
            v_mean = _sphere_volume(d_mean / (2 * h))
            subgrid_rows.append((b.d_lo_um, b.d_hi_um, target, target / v_mean))
            carry = 0.0
            continue
        acc = 0.0
        while True:
            rv = math.exp(rng.uniform(math.log(rv_lo), math.log(rv_hi)))
            v = _sphere_volume(rv)
            if acc + v - target > target - acc:  # stopping is closer than overshooting
                break
            if rv < RV_SUBGRID_MAX:
                subgrid_rows.append((2 * rv * h, 2 * rv * h, v, 1.0))
            else:
                diameters_rv.append(rv)
            acc += v
        carry = target - acc
    sampling_residual = carry

    order = np.argsort(np.asarray(diameters_rv))[::-1]
    rvs = np.asarray(diameters_rv)[order] if diameters_rv else np.empty(0)

    # --- placement ---
    occ = np.zeros((n, n, n))
    particle_id = np.full((n, n, n), -1, dtype=np.int64)
    pid_best = np.zeros((n, n, n))
    occ_flat = occ.ravel()
    pid_flat = particle_id.ravel()
    best_flat = pid_best.ravel()

    n_p = len(rvs)
    p_tier = np.zeros(n_p, dtype=np.int8)
    p_volume = np.zeros(n_p)
    p_center = np.zeros((n_p, 3))
    p_placed = np.zeros(n_p, dtype=bool)
    unplaced_volume = 0.0
    raster_deficit = 0.0

    for i, rv in enumerate(rvs):
        v_full = _sphere_volume(rv)
        if rv <= RV_FRACTIONAL_MAX:
            p_tier[i] = TIER_FRACTIONAL
            # rejection sampling is O(1) expected while most voxels have
            # capacity; the exhaustive scan only runs as a rare fallback
            k = -1
            limit = 1.0 - _CAPACITY_MARGIN - v_full
            for _ in range(200):
                j = int(rng.integers(occ_flat.size))
                if occ_flat[j] <= limit:
                    k = j
                    break
            if k < 0:
                cand = np.flatnonzero(occ_flat <= limit)
                if cand.size == 0:
                    unplaced_volume += v_full
                    continue
                k = int(cand[rng.integers(cand.size)])
            occ_flat[k] += v_full
            if v_full > best_flat[k]:
                best_flat[k] = v_full
                pid_flat[k] = i
            p_volume[i] = v_full
            p_center[i] = np.array(np.unravel_index(k, (n, n, n))) + 0.5
            p_placed[i] = True
        else:
            p_tier[i] = TIER_RESOLVED
            for _ in range(_MAX_PLACEMENT_TRIES):
                center = rng.uniform(0.0, n, 3)
                widx, frac, deficit = _rasterize_sphere(center, rv, n)
                cur = occ_flat[widx]
                interior = frac == 1.0
                ok = np.all(cur[interior] == 0.0) and np.all(
                    cur[~interior] + frac[~interior] <= 1.0 - _CAPACITY_MARGIN)
                if not ok:
                    continue
                occ_flat[widx] += frac
                better = frac > best_flat[widx]
                best_flat[widx[better]] = frac[better]
                pid_flat[widx[better]] = i
                p_volume[i] = float(frac.sum())
                p_center[i] = center
                p_placed[i] = True
                raster_deficit += deficit
                break
            else:
                unplaced_volume += v_full

    # --- subgrid spread proportional to free capacity ---
    v_subgrid = sum(r[2] for r in subgrid_rows)
    if v_subgrid > 0.0:
        avail = 1.0 - occ
        total_avail = float(avail.sum())
        if v_subgrid > total_avail:
            raise GeometryError("subgrid solid volume exceeds available pore capacity")
        occ += (v_subgrid / total_avail) * avail

    if float(occ.max()) > 1.0:
        raise GeometryError(f"voxel occupancy exceeded 1: {occ.max()}")

    # --- fields: full initial saturation, capillary_gas fixed residual = 0 ---
    anhydrous = phase_weights[:, None, None, None] * occ[None, :, :, :]
    capillary_liquid = 1.0 - occ
    capillary_gas = np.zeros_like(occ)

    # --- achieved vs target (report, never hide) ---
    phi_achieved = float(occ.sum()) / n_vox
    rho = np.array([registry.get(p).density_g_cm3 for p in SOLID_PHASE_IDS])
    mass_solid = float((anhydrous.sum(axis=(1, 2, 3)) * rho).sum())
    mass_water = float(capillary_liquid.sum()) * rho_w
    w_c_achieved = mass_water / mass_solid

    report = {
        "grid_size": n,
        "voxel_size_um": h,
        "seed": config.rve.seed,
        "solid_fraction_target": phi_target,
        "solid_fraction_achieved": phi_achieved,
        "solid_fraction_rel_error": abs(phi_achieved - phi_target) / phi_target,
        "w_c_target": config.w_c,
        "w_c_achieved": w_c_achieved,
        "w_c_rel_error": abs(w_c_achieved - config.w_c) / config.w_c,
        "sampling_residual_volume_vox": sampling_residual,
        "unplaced_volume_vox": unplaced_volume,
        "raster_deficit_volume_vox": raster_deficit,
        "n_particles_resolved": int((p_tier[p_placed] == TIER_RESOLVED).sum()),
        "n_particles_fractional": int((p_tier[p_placed] == TIER_FRACTIONAL).sum()),
        "n_subgrid_bins": len(subgrid_rows),
    }

    particles = {
        "id": np.arange(n_p, dtype=np.int64),
        "tier": p_tier,
        "diameter_um": 2.0 * rvs * h,
        "volume_vox": p_volume,
        "center_zyx": p_center,
        "placed": p_placed,
    }
    subgrid_bins = {
        "d_lo_um": np.array([r[0] for r in subgrid_rows]),
        "d_hi_um": np.array([r[1] for r in subgrid_rows]),
        "volume_vox": np.array([r[2] for r in subgrid_rows]),
        "number_est": np.array([r[3] for r in subgrid_rows]),
    }

    return RVEInit(
        phase_ids=SOLID_PHASE_IDS,
        anhydrous_fraction=anhydrous,
        capillary_liquid=capillary_liquid,
        capillary_gas=capillary_gas,
        particle_id=particle_id,
        particles=particles,
        subgrid_bins=subgrid_bins,
        report=report,
    )
