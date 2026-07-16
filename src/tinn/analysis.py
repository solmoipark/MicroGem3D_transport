"""Read-only analysis over run outputs (PRD §3 M5): porosity time series, liquid
percolation, phase fractions, central-slice PNGs, and the §6.3 sanity-band
judgment — plus `report(run_dir)` which regenerates §6.1 ledger checks from the
checkpoints themselves.

This module owns the single implementation of the §6.3 band table and the
gel-porosity policy; the engine imports both (no second copy that can drift).
The PNG writer is dependency-free (stdlib zlib/struct only, PRD §5).
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ledger
from .config import TinnConfig
from .registry import (CLINKER_PHASE_IDS, KINETIC_PHASE_IDS, Registry,
                       default_registry)
from .state import SimulationState
from .storage import load_checkpoint
from .transport import LIQ_EPS

_AXIS_NAMES = ("z", "y", "x")


# ------------------------------------------------------------------ policies

def gel_porosity_vector(config: TinnConfig, hydrate_ids: Sequence[str],
                        registry: Registry) -> np.ndarray:
    """Gel porosity per hydrate channel: registry values for the stoichiometric
    backend, the declared config map for gems3k (absent phases are crystalline,
    0.0 — documented policy). Bulk envelope = skeleton / (1 - eps)."""
    if config.chemistry.backend == "stoichiometric":
        return np.asarray([registry.get(h).gel_porosity for h in hydrate_ids])
    gel_map = config.chemistry.gems_gel_porosity or {}
    return np.asarray([gel_map.get(h, 0.0) for h in hydrate_ids])


# ------------------------------------------------------- per-state quantities

def state_row(state: SimulationState, registry: Registry,
              gel_eps: np.ndarray) -> Dict:
    """One summary row for a state (shared by engine summaries and reports)."""
    alpha = state.alpha()
    cap_por = float((state.capillary_liquid + state.capillary_gas).mean())
    gel_por_vol = float((state.hydrate_env_vol_vox * gel_eps).sum())
    n_vox = state.capillary_liquid.size
    reacted_mass_g = float(np.sum(
        (state.initial_phase_mol - state.phase_mol)
        * np.array([registry.get(p).molar_mass_g_mol for p in KINETIC_PHASE_IDS])))
    gas_cm3 = float(state.capillary_gas.sum()) * state.vox_cm3
    return {
        "time_h": state.time_h,
        "alpha": {p: float(alpha[i]) for i, p in enumerate(KINETIC_PHASE_IDS)},
        "phase_mol": {p: float(state.phase_mol[i])
                      for i, p in enumerate(KINETIC_PHASE_IDS)},
        "hydrate_mol": {h: float(state.hydrate_mol[i])
                        for i, h in enumerate(state.hydrate_ids)
                        if state.hydrate_mol[i] > 0.0},
        "unmet_mol": {p: float(state.unmet_mol[i])
                      for i, p in enumerate(KINETIC_PHASE_IDS)},
        "water_mol": {"free": state.water_free_mol, "gel": state.water_gel_mol,
                      "bound": state.water_bound_mol},
        "porosity_capillary": cap_por,
        "porosity_total": cap_por + gel_por_vol / n_vox,
        "chem_shrinkage_ml_per_g_reacted": (gas_cm3 / reacted_mass_g
                                            if reacted_mass_g > 0 else 0.0),
        "accept_count": state.accept_count,
        "reject_counts": dict(state.reject_counts),
    }


def phase_volume_fractions(state: SimulationState) -> Dict[str, float]:
    """Volume fraction of the RVE per solid channel (anhydrous and hydrates)."""
    n_vox = state.capillary_liquid.size
    out: Dict[str, float] = {}
    from .registry import SOLID_PHASE_IDS
    for i, p in enumerate(SOLID_PHASE_IDS):
        v = float(state.anhydrous_fraction[i].sum()) / n_vox
        if v > 0.0:
            out[p] = v
    for i, h in enumerate(state.hydrate_ids):
        v = float(state.hydrate_fraction[i].sum()) / n_vox
        if v > 0.0:
            out[h] = v
    return out


# --------------------------------------------------------------- percolation

def _label_nonperiodic(mask: np.ndarray) -> np.ndarray:
    """Min-label flooding with 6-connectivity and NO periodic wrap."""
    n_vox = mask.size
    labels = np.where(mask, np.arange(n_vox, dtype=np.int64).reshape(mask.shape),
                      np.int64(n_vox))
    big = np.int64(n_vox)
    for _ in range(n_vox + 1):
        prev = labels
        m = labels
        for ax in range(3):
            fwd = np.full_like(labels, big)
            bwd = np.full_like(labels, big)
            sl_to = [slice(None)] * 3
            sl_from = [slice(None)] * 3
            sl_to[ax] = slice(1, None)
            sl_from[ax] = slice(None, -1)
            fwd[tuple(sl_to)] = labels[tuple(sl_from)]
            bwd[tuple(sl_from)] = labels[tuple(sl_to)]
            m = np.minimum(m, np.where(mask, fwd, big))
            m = np.minimum(m, np.where(mask, bwd, big))
        labels = np.where(mask, np.minimum(labels, m), big)
        if np.array_equal(labels, prev):
            break
    return np.where(mask, labels, np.int64(-1))


def liquid_percolation(capillary_liquid: np.ndarray) -> Dict[str, bool]:
    """Does a connected liquid cluster span the box face-to-face per axis?
    (Non-periodic spanning criterion.) Returns {"z","y","x","any"}."""
    mask = capillary_liquid > LIQ_EPS
    labels = _label_nonperiodic(mask)
    result: Dict[str, bool] = {}
    spans_any = False
    for ax, name in enumerate(_AXIS_NAMES):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = 0
        hi[ax] = -1
        front = set(np.unique(labels[tuple(lo)]))
        back = set(np.unique(labels[tuple(hi)]))
        front.discard(-1)
        back.discard(-1)
        spans = bool(front & back)
        result[name] = spans
        spans_any = spans_any or spans
    result["any"] = spans_any
    return result


# ------------------------------------------------- pore structure (PRD v2.2)

# a voxel counts as pore when its capillary (liquid+gas) fraction reaches this
# majority level — documented mask policy for all pore-structure metrics
PORE_MASK_LEVEL = 0.5


def _edt_sq_1d(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb–Huttenlocher squared-distance lower envelope along the last
    axis of a 2-D stack (rows independent). f is the squared-distance seed."""
    rows, n = f.shape
    out = np.empty_like(f)
    v = np.empty(n, dtype=np.int64)
    z = np.empty(n + 1)
    for r in range(rows):
        fr = f[r]
        k = 0
        v[0] = 0
        z[0] = -np.inf
        z[1] = np.inf
        for q in range(1, n):
            while True:
                p_ = v[k]
                s_ = ((fr[q] + q * q) - (fr[p_] + p_ * p_)) / (2 * q - 2 * p_)
                if s_ <= z[k]:
                    k -= 1
                else:
                    break
            k += 1
            v[k] = q
            z[k] = s_
            z[k + 1] = np.inf
        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            p_ = v[k]
            out[r, q] = (q - p_) * (q - p_) + fr[p_]
    return out


def periodic_edt_um(pore_mask: np.ndarray, voxel_um: float) -> np.ndarray:
    """Exact Euclidean distance (um) from each pore voxel to the nearest solid,
    on the periodic box: each axis pass triples that axis and keeps the center
    third (exact for the separable squared EDT)."""
    big = 1e18
    d2 = np.where(pore_mask, big, 0.0).astype(np.float64)
    n = pore_mask.shape[0]
    for ax in range(3):
        moved = np.moveaxis(d2, ax, -1)
        shape = moved.shape
        flat = moved.reshape(-1, n)
        tripled = np.concatenate([flat, flat, flat], axis=1)
        tr = _edt_sq_1d(tripled)
        flat = tr[:, n:2 * n]
        d2 = np.moveaxis(flat.reshape(shape), -1, ax)
    return np.sqrt(np.clip(d2, 0.0, None)) * voxel_um


def pore_size_distribution(state: SimulationState) -> Dict:
    """Local pore diameters, VOLUME-weighted (PRD 1.3 rev.2): voxels in the
    resolved pore mask carry their EDT diameter and their actual capillary
    volume; partially-filled voxels below the mask level — invisible to a
    binary analysis — enter as a sub-voxel tail with the slab-aperture
    approximation d = f*h (pore volume fraction f flattened against the local
    solid interface). Nothing of the capillary volume is dropped from the
    distribution any more."""
    cap = state.capillary_liquid + state.capillary_gas
    mask = cap >= PORE_MASK_LEVEL
    n_pore = int(mask.sum())
    total_vol = float(cap.sum())
    if total_vol <= 0.0:
        return {"edges_um": [], "volume_fraction_per_bin": [],
                "mean_diameter_um": 0.0, "pore_voxels": 0,
                "pore_volume_vox": 0.0, "subvoxel_volume_fraction": 0.0}
    h = state.config.rve.voxel_size_um
    if n_pore > 0:
        dist = periodic_edt_um(mask, h)
        diam_res = 2.0 * dist[mask]
        w_res = cap[mask]
    else:
        diam_res = np.zeros(0)
        w_res = np.zeros(0)
    sub = (~mask) & (cap > 0.0)
    diam_sub = cap[sub] * h          # slab aperture, always < h/2
    w_sub = cap[sub]
    diam = np.concatenate([diam_res, diam_sub])
    w = np.concatenate([w_res, w_sub])
    edges = [h * m for m in (0.25, 0.5, 1, 2, 4, 8, 16, 32)]
    hist, _ = np.histogram(diam, bins=[0.0] + edges, weights=w)
    return {
        "edges_um": edges,
        "volume_fraction_per_bin": (hist / total_vol).tolist(),
        "mean_diameter_um": float((diam * w).sum() / total_vol),
        "pore_voxels": n_pore,
        "pore_volume_vox": total_vol,
        "subvoxel_volume_fraction": float(w_sub.sum() / total_vol),
    }


def porosity_split(state: SimulationState) -> Dict[str, float]:
    """Capillary porosity split into CONNECTED (in a face-to-face spanning pore
    cluster) and ISOLATED parts. Sub-mask capillary volume counts as isolated;
    connected + isolated == porosity_capillary exactly by construction."""
    cap = state.capillary_liquid + state.capillary_gas
    n_vox = cap.size
    mask = cap >= PORE_MASK_LEVEL
    labels = _label_nonperiodic(mask)
    spanning = set()
    for ax in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax] = 0
        hi[ax] = -1
        front = set(np.unique(labels[tuple(lo)]))
        back = set(np.unique(labels[tuple(hi)]))
        front.discard(-1)
        back.discard(-1)
        spanning |= front & back
    if spanning:
        conn_mask = np.isin(labels, sorted(spanning))
        connected = float(cap[conn_mask].sum()) / n_vox
    else:
        connected = 0.0
    total = float(cap.sum()) / n_vox
    return {"connected": connected, "isolated": total - connected}


def permeability_kozeny_carman(phi_connected: float, d_char_um: float,
                               kc_constant_m2: Optional[float] = None) -> Dict:
    """k = C * phi^3 / (1-phi)^2 with C = d_char^2/180 unless overridden.
    Outside (0,1) the value is NaN + not_available — never fabricated."""
    formula = "k = C * phi_conn^3 / (1 - phi_conn)^2"
    if not (0.0 < phi_connected < 1.0) or (kc_constant_m2 is None
                                           and d_char_um <= 0.0):
        return {"k_m2": float("nan"), "status": "not_available",
                "C_m2": float("nan"), "formula": formula}
    c = kc_constant_m2 if kc_constant_m2 is not None else (d_char_um * 1e-6) ** 2 / 180.0
    k = c * phi_connected ** 3 / (1.0 - phi_connected) ** 2
    return {"k_m2": k, "status": "ok", "C_m2": c, "formula": formula,
            "phi_connected": phi_connected}


# C-S-H (incl. its gel water) relative ion diffusivity vs bulk solution —
# Garboczi & Bentz calibration against steady-state chloride diffusion;
# report-time override via `tinn report --gel-rel-diffusivity`
GEL_REL_DIFFUSIVITY = 0.0025
# uniform background conductance: keeps the CG system nonsingular and bounds
# its contrast; also the resolution floor of the reported D_rel values
NETWORK_FLOOR = 1e-8


# throat-correction knob (PRD 1.3 rev.2): face conductance interpolates
# between the Wiener bounds G = H^(1-beta*w) * A^(beta*w), applied only where
# a cell is genuinely partial (sub-voxel throat) so the resolved-continuum
# limit stays exactly harmonic. LADDER CALIBRATION RESULT (FA30, 32/64/128^3):
# beta=0.6 flattens the 1d ladder (6.0% -> 1.2% spread) but NOT 28d
# (35% -> 25%) — the late-age grid dependence is topological (throat paths
# absent from the coarse 6-neighbor graph), out of reach for any face rule
# within the Wiener bounds. Default therefore stays 0 (classical harmonic);
# the knob remains for sensitivity studies, and grid refinement remains the
# honest route to transport accuracy.
FACE_MIXING_BETA = 0.0


def relative_diffusivity_network(state: SimulationState, gel_eps: np.ndarray,
                                 gel_rel_diffusivity: float = GEL_REL_DIFFUSIVITY,
                                 face_mixing_beta: Optional[float] = None,
                                 tol: float = 1e-10, max_iter: int = 50000) -> Dict:
    """Effective relative diffusivity D_eff/D0 per axis from a face-conductance
    network over the FRACTIONAL fields (PRD 1.3 rev.2) — the two-scale
    composite view: capillary liquid conducts at 1, gel-bearing hydrate volume
    at gel_rel_diffusivity, everything else at the background floor. Unlike the
    binary 0.5-mask percolation, sub-mask liquid slivers and gel water keep
    conducting, so late-age transport does not artificially cut off at the
    voxel resolution. Harmonic-mean face conductances; Dirichlet inlet/outlet
    on the solve axis, periodic transverse; Jacobi-preconditioned CG
    (deterministic: fixed operation order, fixed tolerance)."""
    if face_mixing_beta is None:
        face_mixing_beta = FACE_MIXING_BETA
    g = state.capillary_liquid.astype(np.float64).copy()
    for i in range(state.hydrate_fraction.shape[0]):
        if gel_eps[i] > 0.0:
            g += gel_rel_diffusivity * state.hydrate_fraction[i]
    g = np.maximum(np.clip(g, 0.0, None), NETWORK_FLOOR)
    # partiality of the capillary filling: 1 wherever the voxel is genuinely
    # partial (a sub-voxel throat), 0 for resolved full/empty voxels — the
    # smooth 4f(1-f) variant was tested on the ladder and under-corrects the
    # asymmetric chokes (full pore vs nearly-closed neighbor) that matter
    f = np.clip(state.capillary_liquid, 0.0, 1.0)
    part = ((f > 1e-6) & (f < 1.0 - 1e-6)).astype(np.float64)
    n = g.shape[0]
    beta = float(face_mixing_beta)

    def face(a, b, pa_, pb_):
        h = 2.0 * a * b / (a + b)
        if beta == 0.0:
            return h
        w = beta * np.maximum(pa_, pb_)
        arith = 0.5 * (a + b)
        return h ** (1.0 - w) * arith ** w

    axes: Dict[str, float] = {}
    iters: Dict[str, int] = {}
    status = "ok"
    for ax, name in enumerate(("z", "y", "x")):
        ga = np.ascontiguousarray(np.moveaxis(g, ax, 0))
        pa = np.ascontiguousarray(np.moveaxis(part, ax, 0))
        gz = face(ga[:-1], ga[1:], pa[:-1], pa[1:])
        gy = face(ga, np.roll(ga, -1, axis=1), pa, np.roll(pa, -1, axis=1))
        gx = face(ga, np.roll(ga, -1, axis=2), pa, np.roll(pa, -1, axis=2))
        gin = 2.0 * ga[0]
        gout = 2.0 * ga[-1]
        diag = np.zeros_like(ga)
        diag[:-1] += gz
        diag[1:] += gz
        diag += gy + np.roll(gy, 1, axis=1)
        diag += gx + np.roll(gx, 1, axis=2)
        diag[0] += gin
        diag[-1] += gout

        def apply_a(phi):
            out = diag * phi
            out[:-1] -= gz * phi[1:]
            out[1:] -= gz * phi[:-1]
            out -= gy * np.roll(phi, -1, axis=1)
            out -= np.roll(gy, 1, axis=1) * np.roll(phi, 1, axis=1)
            out -= gx * np.roll(phi, -1, axis=2)
            out -= np.roll(gx, 1, axis=2) * np.roll(phi, 1, axis=2)
            return out

        b = np.zeros_like(ga)
        b[0] = gin  # Dirichlet phi=1 upstream, phi=0 downstream
        # linear-ramp start: exact for a homogeneous medium
        phi = np.broadcast_to(((n - 0.5 - np.arange(n)) / n)[:, None, None],
                              ga.shape).copy()
        r = b - apply_a(phi)
        z = r / diag
        p = z.copy()
        rz = float((r * z).sum())
        b_norm = float(np.sqrt((b * b).sum()))
        it = 0
        converged = float(np.sqrt((r * r).sum())) <= tol * b_norm
        while not converged and it < max_iter:
            it += 1
            ap = apply_a(p)
            alpha = rz / float((p * ap).sum())
            phi += alpha * p
            r -= alpha * ap
            if float(np.sqrt((r * r).sum())) <= tol * b_norm:
                converged = True
                break
            z = r / diag
            rz_new = float((r * z).sum())
            p = z + (rz_new / rz) * p
            rz = rz_new
        if not converged:
            status = "not_converged"
        flux = float((gin * (1.0 - phi[0])).sum())
        axes[name] = flux / n
        iters[name] = it
    return {"relative_diffusivity": axes,
            "mean": float(np.mean(list(axes.values()))),
            "gel_rel_diffusivity": gel_rel_diffusivity,
            "face_mixing_beta": beta,
            "background_floor": NETWORK_FLOOR,
            "cg_iterations": iters, "status": status}


# ------------------------------------------------------------- §6.3 judgment

def sanity_band(rows: List[dict], config: TinnConfig) -> dict:
    """PRD §6.3: non-blocking physical-plausibility bands, pass/warn only."""
    checks: List[dict] = []

    def add(name: str, ok: bool, value) -> None:
        checks.append({"check": name, "status": "pass" if ok else "warn",
                       "value": value})

    # the PRD 6.3 alpha bands are for TOTAL CLINKER alpha (the 4 P&K phases);
    # SCM reaction degrees follow their own slower schedules and stay out
    w = {p: config.binder.mass_fractions.get(p, 0.0) for p in CLINKER_PHASE_IDS}
    tot_w = sum(w.values())
    bands = {24.0: (0.25, 0.55), 168.0: (0.50, 0.80), 672.0: (0.65, 0.90)}
    for row in rows:
        t = row["time_h"]
        for band_t, (lo, hi) in bands.items():
            # tolerant match: accumulated step times can land one ulp off
            if tot_w > 0.0 and abs(t - band_t) <= 1e-6 * band_t:
                total = sum(row["alpha"].get(p, 0.0) * w[p]
                            for p in CLINKER_PHASE_IDS) / tot_w
                add(f"total_clinker_alpha@{band_t:g}h", lo <= total <= hi, total)
    if rows:
        add("alpha_order_C3S_ge_C2S",
            all(r["alpha"]["C3S"] >= r["alpha"]["C2S"] - 1e-12 for r in rows), None)
        ch_name = ("Portlandite" if any("Portlandite" in r["hydrate_mol"]
                                        for r in rows) else "CH")
        ch = [r["hydrate_mol"].get(ch_name, 0.0) for r in rows]
        # re-dissolution is legal physics under full re-equilibration (PRD
        # v2.2): pozzolanic blends consume CH late. The band only asks that CH
        # EXISTS from 1 d onward while clinker remains.
        late = [(r, v) for r, v in zip(rows, ch) if r["time_h"] >= 24.0
                and any(r["alpha"].get(p2, 0.0) < 1.0 for p2 in ("C3S", "C2S"))]
        add("CH_present_after_1d",
            all(v > 0.0 for _, v in late) if late else True,
            ch[-1])
        por = [r["porosity_capillary"] for r in rows]
        add("capillary_porosity_monotone_decrease",
            all(b <= a + 1e-12 for a, b in zip(por, por[1:])), por[-1])
        sh = rows[-1]["chem_shrinkage_ml_per_g_reacted"]
        add("chem_shrinkage_ml_per_g", 0.03 <= sh <= 0.08, sh)
        phs = [v for r in rows
               for v in r.get("ledger_metrics", {}).get("cluster_ph", {}).values()]
        if phs:
            add("cluster_ph_band_12.4_13.9",
                all(12.4 <= p <= 13.9 for p in phs), [min(phs), max(phs)])
    return {
        "note": ("physical plausibility bands (PRD 6.3), pass/warn only - "
                 "this is not scientific validation"),
        "checks": checks,
    }


# --------------------------------------------------------------- PNG writing

# convex-combination colors per volume channel (fractions sum to 1 per voxel)
_COL_ANHYDROUS = np.array([90, 90, 90], dtype=np.float64)    # clinker: gray
_COL_SCM = np.array([30, 110, 130], dtype=np.float64)        # reactive SCM: teal
_COL_INERT = np.array([70, 150, 70], dtype=np.float64)       # unassigned filler: green
_COL_HYDRATE = np.array([215, 150, 60], dtype=np.float64)    # hydrates: orange
_COL_LIQUID = np.array([40, 90, 220], dtype=np.float64)      # capillary water: blue
_COL_GAS = np.array([235, 235, 235], dtype=np.float64)       # shrinkage gas: near-white


def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal deterministic RGB8 PNG writer (stdlib only)."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    payload = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
               + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    Path(path).write_bytes(payload)


def central_slice_rgb(state: SimulationState) -> np.ndarray:
    """Central z-slice as an RGB8 image: clinker gray, unassigned inert filler
    green, hydrates orange, liquid blue, gas near-white (convex combination)."""
    from .registry import CLINKER_PHASE_IDS as _CLK
    from .registry import SCM_PHASE_IDS as _SCM
    from .registry import SOLID_PHASE_IDS
    z = state.grid_size // 2
    clk = [SOLID_PHASE_IDS.index(p) for p in _CLK]
    scm = [SOLID_PHASE_IDS.index(p) for p in _SCM]
    other = [i for i in range(len(SOLID_PHASE_IDS)) if i not in clk + scm]
    anh = state.anhydrous_fraction[clk, z].sum(axis=0)
    scm_f = state.anhydrous_fraction[scm, z].sum(axis=0)
    inert = state.anhydrous_fraction[other, z].sum(axis=0)
    hyd = state.hydrate_fraction[:, z].sum(axis=0)
    liq = state.capillary_liquid[z]
    gas = state.capillary_gas[z]
    img = (anh[..., None] * _COL_ANHYDROUS + scm_f[..., None] * _COL_SCM
           + inert[..., None] * _COL_INERT + hyd[..., None] * _COL_HYDRATE
           + liq[..., None] * _COL_LIQUID + gas[..., None] * _COL_GAS)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# -------------------------------------------------------------------- report

def report(run_dir: str, out_dir: Optional[str] = None,
           registry: Optional[Registry] = None,
           kc_constant_m2: Optional[float] = None,
           gel_rel_diffusivity: float = GEL_REL_DIFFUSIVITY,
           face_mixing_beta: Optional[float] = None) -> Dict:
    """Regenerate the full report from a run directory's checkpoints: per-output
    rows, §6.1 ledger re-checks, percolation, phase fractions, §6.3 band, and a
    central-slice PNG per checkpoint. Writes report.json + PNGs to out_dir
    (default: the run directory)."""
    reg = registry or default_registry()
    run = Path(run_dir)
    ckpts = sorted(p for p in run.glob("ckpt_*") if p.is_dir())
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_* checkpoints under {run}")
    out = Path(out_dir) if out_dir else run
    out.mkdir(parents=True, exist_ok=True)

    # per-cluster pH lives only in run summaries (a solver diagnostic, not
    # checkpointed state) — merge it in when the run wrote one
    summary_ph: Dict[float, dict] = {}
    summary_path = run / "summary.json"
    if summary_path.is_file():
        try:
            for row in json.loads(summary_path.read_text(encoding="utf-8"))["outputs"]:
                ph = row.get("ledger_metrics", {}).get("cluster_ph")
                if ph:
                    summary_ph[float(row["time_h"])] = ph
        except Exception:
            pass  # a foreign/corrupt summary never blocks a checkpoint report

    rows: List[dict] = []
    config = None
    backend_id = None
    for ck in ckpts:
        state = load_checkpoint(str(ck), reg)
        config = state.config
        backend_id = state.backend_id
        gel_eps = gel_porosity_vector(config, state.hydrate_ids, reg)
        row = state_row(state, reg, gel_eps)
        check = ledger.check_all(state, reg)
        row["ledger_metrics"] = dict(check.metrics)
        row["ledger_violations"] = list(check.violations)
        if row["time_h"] in summary_ph:
            row["ledger_metrics"]["cluster_ph"] = summary_ph[row["time_h"]]
        row["percolation"] = liquid_percolation(state.capillary_liquid)
        row["phase_volume_fractions"] = phase_volume_fractions(state)
        split = porosity_split(state)
        row["porosity_connected"] = split["connected"]
        row["porosity_isolated"] = split["isolated"]
        psd_row = pore_size_distribution(state)
        row["pore_size_distribution"] = psd_row
        row["permeability"] = permeability_kozeny_carman(
            split["connected"], psd_row["mean_diameter_um"], kc_constant_m2)
        row["diffusivity_network"] = relative_diffusivity_network(
            state, gel_eps, gel_rel_diffusivity,
            face_mixing_beta=face_mixing_beta)
        png_name = f"slice_{ck.name}.png"
        write_png(str(out / png_name), central_slice_rgb(state))
        row["slice_png"] = png_name
        rows.append(row)
    # checkpoint NAMES sort lexicographically (ckpt_1000 < ckpt_999) — time is
    # the authority for series order
    rows.sort(key=lambda r: r["time_h"])

    result = {
        "run_dir": str(run),
        "config_hash": config.config_hash(),
        "backend": backend_id,
        "outputs": rows,
        "sanity_band": sanity_band(rows, config),
    }
    (out / "report.json").write_text(json.dumps(result, indent=1),
                                     encoding="utf-8")
    return result
