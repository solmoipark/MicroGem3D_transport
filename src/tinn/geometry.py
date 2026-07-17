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
from .registry import (INERT_PHASE_ID, Registry, SCM_PHASE_IDS,
                       SOLID_PHASE_IDS)

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


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) rotation from a normalized random quaternion (seeded,
    deterministic). Drawn ONLY for shaped materials so shape-free configs make
    byte-identical RNG draws to the legacy path."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rasterize_ellipsoid(center: np.ndarray, rv: float, ratios: np.ndarray,
                         rot: np.ndarray, n: int
                         ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Periodic rotated-ellipsoid rasterization (PRD 1.2 rev.2). `ratios` are
    volume-normalized semi-axis ratios (product 1), so the ellipsoid volume
    equals the volume-equivalent sphere's; same exact-boundary bookkeeping as
    _rasterize_sphere. Interior/exterior classification uses the conservative
    level-set bound |grad v| <= 1/s_min: anything uncertain goes through the
    subsampled boundary path, which is exact in expectation."""
    s = rv * ratios
    r_max = float(s.max())
    s_min = float(s.min())
    lo = np.floor(center - r_max - _HALF_DIAG).astype(np.int64)
    hi = np.floor(center + r_max + _HALF_DIAG).astype(np.int64) + 1
    if np.any(hi - lo > n):
        raise GeometryError(
            f"particle with major semi-axis {r_max} voxels does not fit the "
            f"periodic RVE")
    axes = [np.arange(lo[a], hi[a]) for a in range(3)]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    idx = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    # particle-frame coordinates y = R^T x, written as a broadcast product —
    # np.matmul dispatches to delay-loaded BLAS, which faults in some conda
    # numpy builds on Windows (0xc06d007f); a 3x3 rotation never needs BLAS
    x = idx + 0.5 - center
    y = (x[:, :, None] * rot[None, :, :]).sum(axis=1)
    v = np.sqrt(((y / s) ** 2).sum(axis=1))

    frac = np.zeros(len(idx))
    margin = _HALF_DIAG / s_min
    frac[v <= 1.0 - margin] = 1.0
    bmask = (v > 1.0 - margin) & (v < 1.0 + margin)
    if bmask.any():
        off = (np.arange(_SUBSAMPLES) + 0.5) / _SUBSAMPLES
        oz, oy, ox = np.meshgrid(off, off, off, indexing="ij")
        offs = np.stack([oz, oy, ox], axis=-1).reshape(-1, 3)
        pts = idx[bmask][:, None, :] + offs[None, :, :]
        xb = pts - center
        yb = (xb[:, :, :, None] * rot[None, None, :, :]).sum(axis=2)
        inside = ((yb / s) ** 2).sum(axis=2) <= 1.0
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

    # --- per-material populations (PRD v2.2): "clinker" carries the 4 clinker
    # phases + the inert residual with uniform composition; each SCM with mass
    # is its own pure-glass population with its own PSD. A single-material
    # (SCM-free) config makes byte-identical RNG draws to the legacy path.
    materials = []  # (name, phase_weights (P,), volume_target_vox, psd)
    clinker_vol = sum(vol_g[p] for p in SOLID_PHASE_IDS if p not in SCM_PHASE_IDS)
    psd_map = config.material_psd or {}
    if clinker_vol > 0.0:  # an all-SCM binder has no clinker population
        clinker_w = np.array([vol_g[p] / clinker_vol if p not in SCM_PHASE_IDS
                              else 0.0 for p in SOLID_PHASE_IDS])
        materials.append(("clinker", clinker_w,
                          v_solid_target * (clinker_vol / v_solid),
                          psd_map.get("clinker", config.psd)))
    for sid in SCM_PHASE_IDS:
        if vol_g.get(sid, 0.0) > 0.0:
            w = np.zeros(len(SOLID_PHASE_IDS))
            w[SOLID_PHASE_IDS.index(sid)] = 1.0
            materials.append((sid, w, v_solid_target * (vol_g[sid] / v_solid),
                              psd_map.get(sid, config.psd)))
    # per-material shape (PRD 1.2 rev.2): normalized semi-axis ratios, or None
    # for spheres — shape-free materials draw no extra RNG numbers
    shape_map = config.material_shape or {}
    mat_shape = [np.asarray(shape_map[name].normalized_axes())
                 if name in shape_map else None
                 for name, _, _, _ in materials]

    # --- sample particle sizes per material, per PSD bin (coarse -> fine,
    # carrying residual within the material) ---
    diameters_rv = []
    diameter_mat = []
    subgrid_rows = []  # (material_idx, d_lo_um, d_hi_um, volume_vox, number_est)
    sampling_residual = 0.0
    for m_idx, (_, _, v_target_m, psd) in enumerate(materials):
        carry = 0.0
        for b in reversed(psd.bins):
            target = v_target_m * b.volume_fraction + carry
            if target <= 0.0:
                carry = target
                continue
            rv_lo, rv_hi = b.d_lo_um / (2 * h), b.d_hi_um / (2 * h)
            if rv_hi <= RV_SUBGRID_MAX:
                d_mean = math.sqrt(b.d_lo_um * b.d_hi_um)
                v_mean = _sphere_volume(d_mean / (2 * h))
                subgrid_rows.append((m_idx, b.d_lo_um, b.d_hi_um, target,
                                     target / v_mean))
                carry = 0.0
                continue
            acc = 0.0
            while True:
                rv = math.exp(rng.uniform(math.log(rv_lo), math.log(rv_hi)))
                v = _sphere_volume(rv)
                if acc + v - target > target - acc:  # stopping is closer
                    break
                if rv < RV_SUBGRID_MAX:
                    subgrid_rows.append((m_idx, 2 * rv * h, 2 * rv * h, v, 1.0))
                else:
                    diameters_rv.append(rv)
                    diameter_mat.append(m_idx)
                acc += v
            carry = target - acc
        sampling_residual += carry

    rvs_all = np.asarray(diameters_rv)
    mats_all = np.asarray(diameter_mat, dtype=np.int8)
    if len(diameters_rv):
        # primary: size descending; ties: material order then sampling order
        order = np.lexsort((np.arange(len(rvs_all)), mats_all, -rvs_all))
        rvs = rvs_all[order]
        p_material = mats_all[order]
    else:
        rvs = np.empty(0)
        p_material = np.empty(0, dtype=np.int8)

    # --- placement ---
    occ = np.zeros((n, n, n))
    occ_m = np.zeros((len(materials), n, n, n))
    occ_m_flat = occ_m.reshape(len(materials), -1)
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
            occ_m_flat[p_material[i], k] += v_full
            if v_full > best_flat[k]:
                best_flat[k] = v_full
                pid_flat[k] = i
            p_volume[i] = v_full
            p_center[i] = np.array(np.unravel_index(k, (n, n, n))) + 0.5
            p_placed[i] = True
        else:
            p_tier[i] = TIER_RESOLVED
            shape = mat_shape[p_material[i]]
            for _ in range(_MAX_PLACEMENT_TRIES):
                center = rng.uniform(0.0, n, 3)
                if shape is None:
                    widx, frac, deficit = _rasterize_sphere(center, rv, n)
                else:
                    rot = _random_rotation(rng)
                    widx, frac, deficit = _rasterize_ellipsoid(
                        center, rv, shape, rot, n)
                cur = occ_flat[widx]
                interior = frac == 1.0
                ok = np.all(cur[interior] == 0.0) and np.all(
                    cur[~interior] + frac[~interior] <= 1.0 - _CAPACITY_MARGIN)
                if not ok:
                    continue
                occ_flat[widx] += frac
                occ_m_flat[p_material[i], widx] += frac
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

    # --- subgrid spread proportional to free capacity, per material in order ---
    for m_idx in range(len(materials)):
        v_subgrid = sum(r[3] for r in subgrid_rows if r[0] == m_idx)
        if v_subgrid > 0.0:
            avail = 1.0 - occ
            total_avail = float(avail.sum())
            if v_subgrid > total_avail:
                raise GeometryError(
                    "subgrid solid volume exceeds available pore capacity")
            add = (v_subgrid / total_avail) * avail
            occ += add
            occ_m[m_idx] += add

    if float(occ.max()) > 1.0:
        raise GeometryError(f"voxel occupancy exceeded 1: {occ.max()}")

    # --- fields: full initial saturation, capillary_gas fixed residual = 0 ---
    anhydrous = np.zeros((len(SOLID_PHASE_IDS), n, n, n))
    for m_idx, (_, weights, _, _) in enumerate(materials):
        anhydrous += weights[:, None, None, None] * occ_m[m_idx][None, :, :, :]
    capillary_liquid = 1.0 - occ
    capillary_gas = np.zeros_like(occ)

    # --- achieved vs target (report, never hide) ---
    phi_achieved = float(occ.sum()) / n_vox
    rho = np.array([registry.get(p).density_g_cm3 for p in SOLID_PHASE_IDS])
    mass_solid = float((anhydrous.sum(axis=(1, 2, 3)) * rho).sum())
    mass_water = float(capillary_liquid.sum()) * rho_w
    w_c_achieved = mass_water / mass_solid

    mat_report = {}
    for m_idx, (name, weights, v_target_m, psd) in enumerate(materials):
        row = {
            "volume_target_vox": v_target_m,
            "volume_achieved_vox": float(occ_m[m_idx].sum()),
            "rel_error": (abs(float(occ_m[m_idx].sum()) - v_target_m)
                          / v_target_m if v_target_m > 0 else 0.0),
        }
        # sphere-based specific-surface estimate from the PSD (diagnostic
        # only, PRD 1.2 rev.2): lets a measured PSD be cross-checked against
        # the measured Blaine fed to kinetics
        vol_m = sum(vol_g[p] for i, p in enumerate(SOLID_PHASE_IDS)
                    if weights[i] > 0.0)
        mass_m = sum(masses.get(p, 0.0) for i, p in enumerate(SOLID_PHASE_IDS)
                     if weights[i] > 0.0)
        if vol_m > 0.0 and mass_m > 0.0:
            rho_kg_m3 = (mass_m / vol_m) * 1000.0
            row["ssa_est_m2_kg"] = sum(
                b.volume_fraction * 6.0
                / (rho_kg_m3 * math.sqrt(b.d_lo_um * b.d_hi_um) * 1e-6)
                for b in psd.bins)
        # measured-PSD exclusions (never hidden): grid truncation + the mass
        # the measured input carries outside its represented window
        trunc_key = name if name in psd_map else "__shared__"
        trunc = config._psd_truncation.get(trunc_key, 0.0)
        if trunc > 0.0:
            row["psd_truncated_volume_fraction"] = trunc
        excl = psd.measured_window_excluded()
        if excl > 0.0:
            row["psd_window_excluded"] = excl
        mat_report[name] = row

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
        "materials": mat_report,
    }

    particles = {
        "id": np.arange(n_p, dtype=np.int64),
        "tier": p_tier,
        "material": p_material.copy(),
        "diameter_um": 2.0 * rvs * h,
        "volume_vox": p_volume,
        "center_zyx": p_center,
        "placed": p_placed,
    }
    subgrid_bins = {
        "material": np.array([r[0] for r in subgrid_rows], dtype=np.int8),
        "d_lo_um": np.array([r[1] for r in subgrid_rows]),
        "d_hi_um": np.array([r[2] for r in subgrid_rows]),
        "volume_vox": np.array([r[3] for r in subgrid_rows]),
        "number_est": np.array([r[4] for r in subgrid_rows]),
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
