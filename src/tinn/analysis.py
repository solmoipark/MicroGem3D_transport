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
from .registry import (CLINKER_PHASE_IDS, ELEMENT_IDS, KINETIC_PHASE_IDS,
                       Registry, SALT_PHASE_IDS, SCM_PHASE_IDS,
                       default_registry, registry_for)
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
    # chemical shrinkage is reported per gram of REACTED BINDER. A salt
    # carrier merely dissolves — it is not a hydration reaction — and E3 made
    # it a kinetic phase, so leaving it in this denominator would dilute the
    # ratio by the carrier dose (measured: 22 % of "reacted" mass at 12 h,
    # enough to flip the PRD 6.3 verdict). Carriers are excluded; the gas
    # volume in the numerator still counts everything (E3 review finding).
    reacted_mass_g = float(np.sum(
        [(state.initial_phase_mol[i] - state.phase_mol[i])
         * registry.get(p).molar_mass_g_mol
         for i, p in enumerate(KINETIC_PHASE_IDS) if p not in SALT_PHASE_IDS]))
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
        "solid_solution_composition": solid_solution_composition(state),
        "porosity_capillary": cap_por,
        "porosity_total": cap_por + gel_por_vol / n_vox,
        "chem_shrinkage_ml_per_g_reacted": (gas_cm3 / reacted_mass_g
                                            if reacted_mass_g > 0 else 0.0),
        "accept_count": state.accept_count,
        "reject_counts": dict(state.reject_counts),
    }


def solid_solution_composition(state: SimulationState) -> Dict[str, Dict]:
    """Derived observables from the E1 endmember ledger (PRD 2.3 rev.3) —
    diagnostics only, the mol ledger stays authoritative. For every channel
    with more than one endmember and material holdings: the endmember mol
    split plus Ca/Si, H/Si (silicate gels) and Al/(Al+Fe) (Al-Fe solid
    solutions) where the denominators carry mass."""
    out: Dict[str, Dict] = {}
    if not len(state.endmember_ids):
        return out
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    by_channel: Dict[str, List[int]] = {}
    for j, (h, _dc) in enumerate(state.endmember_ids):
        by_channel.setdefault(h, []).append(j)
    # dust floor: fully-redissolved channels legally keep signed float dust
    # (~eps x turnover) in the ledger; ratios computed from dust are noise,
    # so channels below a relative floor of the global holdings are skipped
    global_abs = float(np.abs(state.endmember_mol).sum())
    floor = 1e-9 * global_abs
    for h, idxs in by_channel.items():
        if len(idxs) < 2:
            continue
        mols = state.endmember_mol[idxs]
        total = float(mols.sum())
        if total <= floor:
            continue
        elems = (mols[:, None] * state.endmember_elements[idxs]).sum(axis=0)
        row: Dict[str, object] = {
            "endmember_mol": {state.endmember_ids[j][1]: float(state.endmember_mol[j])
                              for j in idxs if state.endmember_mol[j] != 0.0}}
        si = float(elems[el["Si"]])
        if si > 0.0:
            row["ca_si"] = float(elems[el["Ca"]]) / si
            row["h_si"] = float(elems[el["H"]]) / si
        al, fe = float(elems[el["Al"]]), float(elems[el["Fe"]])
        if al + fe > 0.0:
            row["al_over_al_fe"] = al / (al + fe)
        out[h] = row
    return out


_OXIDE_M = {"SO3": 80.06, "Na2O": 61.979, "K2O": 94.196}


def _oxides_from_mass(mass_g: Dict[str, float], registry: Registry,
                      basis_g: float) -> Dict[str, float]:
    out = {"so3_pct": 0.0, "na2o_pct": 0.0, "k2o_pct": 0.0}
    if basis_g <= 0.0:
        out["na2o_eq_pct"] = 0.0
        return out
    for pid, grams in mass_g.items():
        if grams <= 0.0:
            continue
        entry = registry.get(pid)
        mm = entry.molar_mass_g_mol
        if not mm:
            continue
        mol_per_g = grams / mm / basis_g   # mol of formula unit per g of binder
        f = entry.formula or {}
        out["so3_pct"] += mol_per_g * f.get("S", 0.0) * _OXIDE_M["SO3"] * 100.0
        out["na2o_pct"] += (mol_per_g * f.get("Na", 0.0) / 2.0
                            * _OXIDE_M["Na2O"] * 100.0)
        out["k2o_pct"] += (mol_per_g * f.get("K", 0.0) / 2.0
                           * _OXIDE_M["K2O"] * 100.0)
    out["na2o_eq_pct"] = out["na2o_pct"] + 0.658 * out["k2o_pct"]
    return out


def binder_oxide_diagnostics(config: TinnConfig,
                             registry: Optional[Registry] = None,
                             state: Optional[SimulationState] = None
                             ) -> Dict[str, float]:
    """E3 diagnostic: SO3 and alkali content in the oxide wt% a mill
    certificate reports (per 100 g of binder as batched, inert residual
    included). Derived from the registry formulas of every phase with mass.
    Na2O-equivalent uses the standard 0.658 = M(Na2O)/M(K2O).

    With `state`, the AS-BUILT values are added under `*_as_built`: the same
    oxides recomputed from `initial_phase_mol`, i.e. what the rasterized RVE
    actually holds. These differ from the recipe because the particle sampler
    hits a small population's volume target only approximately (a 0.6 wt%
    carrier at 32^3 is ~90 voxels, where one particle is the whole
    population), and it is the AS-BUILT dose that drives the simulated
    chemistry (E3 review finding). A large gap means the carrier needs a finer
    material_psd or a larger grid — never a silent correction here."""
    reg = registry or default_registry()
    out = _oxides_from_mass(
        dict(config.binder.mass_fractions), reg, basis_g=1.0)
    if state is None:
        return out
    mass = {}
    total = 0.0
    for i, pid in enumerate(KINETIC_PHASE_IDS):
        mm = reg.get(pid).molar_mass_g_mol
        if not mm:
            continue
        grams = float(state.initial_phase_mol[i]) * mm
        mass[pid] = grams
        total += grams
    # the inert residual carries no oxides but IS part of the batched basis
    frac = config.binder.mass_fractions
    assigned = sum(frac.values())
    if assigned > 0.0:
        total += total * (config.binder.unassigned / assigned)
    for k, v in _oxides_from_mass(mass, reg, basis_g=total).items():
        out[f"{k}_as_built"] = v
    return out


def casi_channel(state: SimulationState) -> Optional[str]:
    """The ONE channel the cluster Ca/Si observable reads: the multi-endmember
    (solid solution) channel holding the largest pooled Si mass. Summing every
    Si-bearing solid solution into one ratio would blend C-S-H with siliceous
    hydrogarnet / straetlingite / M-S-H into a composition matching no actual
    phase (E2 review finding), so exactly one channel is selected."""
    pools = state.cluster_endmember_mol
    if pools.shape[0] == 0 or not len(state.endmember_ids):
        return None
    si_col = ELEMENT_IDS.index("Si")
    by_channel: Dict[str, List[int]] = {}
    for j, (h, _dc) in enumerate(state.endmember_ids):
        by_channel.setdefault(h, []).append(j)
    best, best_si = None, 0.0
    for h in state.hydrate_ids:            # deterministic order
        idxs = by_channel.get(h, [])
        if len(idxs) < 2:
            continue
        rows_si = state.endmember_elements[idxs][:, si_col]
        if not np.any(rows_si > 0.0):
            continue
        si_mass = float(np.clip(pools[:, idxs], 0.0, None).sum(axis=0) @ rows_si)
        if si_mass > best_si:
            best, best_si = h, si_mass
    return best


def cluster_ca_si(state: SimulationState) -> Dict[int, float]:
    """E2 observable (PRD 4.5 v3.0): Ca/Si of each cluster's holdings in the
    dominant silicate solid-solution channel (see casi_channel), from the
    per-cluster endmember pools — the cluster-resolution composition map
    comparable to SEM-EDS element maps. Diagnostics only; clusters whose
    pool in that channel is dust are skipped."""
    h = casi_channel(state)
    if h is None:
        return {}
    pools = state.cluster_endmember_mol
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    idxs = [j for j, (hh, _dc) in enumerate(state.endmember_ids) if hh == h]
    rows = state.endmember_elements[idxs]
    ch_pools = np.clip(pools[:, idxs], 0.0, None)
    global_scale = float(ch_pools.sum())
    out: Dict[int, float] = {}
    for c in range(pools.shape[0]):
        mols = ch_pools[c]
        if float(mols.sum()) <= 1e-9 * max(global_scale, 1e-300):
            continue
        elems = mols @ rows
        si = float(elems[el["Si"]])
        if si > 0.0:
            out[c] = float(elems[el["Ca"]]) / si
    return out


def casi_map_rgb(state: SimulationState, lo: float = 0.8, hi: float = 2.2
                 ) -> Optional[np.ndarray]:
    """Central-slice Ca/Si map (E2): cluster-resolution ratios painted over
    the cluster label field, blue(lo) -> red(hi); non-cluster voxels dark,
    ratio-less clusters gray. Returns None when no ratios exist."""
    if not hi > lo:
        raise ValueError(f"casi_map_rgb requires hi > lo, got [{lo}, {hi}]")
    ratios = cluster_ca_si(state)
    if not ratios:
        return None
    n = state.grid_size
    labels = state.cluster_id[n // 2]
    img = np.full((n, n, 3), 30, dtype=np.float64)
    gray = np.array([120.0, 120.0, 120.0])
    for c in np.unique(labels):
        if c < 0:
            continue
        m = labels == c
        r = ratios.get(int(c))
        if r is None:
            img[m] = gray
        else:
            t = min(1.0, max(0.0, (r - lo) / (hi - lo)))
            img[m] = np.array([60 + 195 * t, 60, 255 - 195 * t])
    return img.astype(np.uint8)


def phase_volume_fractions(state: SimulationState) -> Dict[str, float]:
    """Volume fraction of the RVE per solid channel (anhydrous and hydrates).

    The two channel sets live in different namespaces and E3 made them
    collide: a bundle declares equilibrium phases named `hemihydrate`,
    `arcanite`, `thenardite` — exactly the ids of the undissolved carriers.
    The re-precipitated hydrate used to overwrite the anhydrous entry (a
    ~100x under-report of the carrier still sitting in the RVE), so a
    colliding hydrate is suffixed instead (E3 review finding)."""
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
            key = f"{h} (hydrate)" if h in SOLID_PHASE_IDS else h
            out[key] = v
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
    # ABSOLUTE micrometer bin edges (rev.2): resolution-independent, so PSD
    # tables from different voxel sizes are directly comparable; bins finer
    # than the current resolution simply stay empty
    edges = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
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
    """PRD §6.3: non-blocking physical-plausibility bands. Statuses: pass /
    warn / info ("info" = the band's premise does not apply to this mix, the
    value is shown but not judged — rev.2)."""
    checks: List[dict] = []

    def add(name: str, ok: bool, value, status: Optional[str] = None) -> None:
        checks.append({"check": name,
                       "status": status or ("pass" if ok else "warn"),
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
        # the 0.03-0.08 band is an OPC anchor; pozzolanic reactions carry a
        # legitimately higher shrinkage and no citable per-SCM bounds exist,
        # so for blends the check reports "info" instead of judging (rev.2)
        last_a = rows[-1]["alpha"]
        mf = config.binder.mass_fractions
        # salt carriers are excluded here for the same reason as in the
        # shrinkage denominator: dissolving them is not binder reaction, and
        # counting them would shrink the SCM share below the blend threshold
        reacted = {p: mf.get(p, 0.0) * last_a.get(p, 0.0) for p in mf
                   if p not in SALT_PHASE_IDS}
        tot_reacted = sum(reacted.values())
        scm_share = (sum(v for p, v in reacted.items() if p in SCM_PHASE_IDS)
                     / tot_reacted if tot_reacted > 0 else 0.0)
        if scm_share > 0.10:
            add("chem_shrinkage_ml_per_g", True,
                {"value": sh, "scm_reacted_mass_share": scm_share,
                 "note": "OPC band not applicable to blends (PRD 6.3)"},
                status="info")
        else:
            add("chem_shrinkage_ml_per_g", 0.03 <= sh <= 0.08, sh)
        # main-solution pH band: clusters holding >= 1 % of the liquid are
        # judged; nearly-dry pockets are known noisy diagnostics (documented)
        # and are counted, not judged (rev.2)
        main_ph: List[float] = []
        pocket_out: List[float] = []
        for r in rows:
            lm = r.get("ledger_metrics", {})
            fr = {str(k): v for k, v in lm.get("cluster_liq_frac", {}).items()}
            for k, p in lm.get("cluster_ph", {}).items():
                if fr and fr.get(str(k), 0.0) < 0.01:
                    if not (12.4 <= p <= 13.9):
                        pocket_out.append(p)
                else:
                    main_ph.append(p)
        if main_ph or pocket_out:
            val = {"main_range": ([min(main_ph), max(main_ph)]
                                  if main_ph else None),
                   "pocket_outliers": len(pocket_out)}
            if pocket_out:
                val["pocket_outlier_range"] = [min(pocket_out), max(pocket_out)]
            add("cluster_ph_band_12.4_13.9",
                all(12.4 <= p <= 13.9 for p in main_ph), val)
    return {
        "note": ("physical plausibility bands (PRD 6.3), pass/warn/info - "
                 "this is not scientific validation"),
        "checks": checks,
    }


# --------------------------------------------------------------- PNG writing

# convex-combination colors per volume channel (fractions sum to 1 per voxel)
_COL_ANHYDROUS = np.array([90, 90, 90], dtype=np.float64)    # clinker: gray
_COL_SCM = np.array([30, 110, 130], dtype=np.float64)        # reactive SCM: teal
_COL_INERT = np.array([70, 150, 70], dtype=np.float64)       # unassigned filler: green
_COL_SALT = np.array([200, 190, 80], dtype=np.float64)       # salt carriers: ochre
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
    """Central z-slice as an RGB8 image: clinker gray, reactive SCM teal,
    undissolved salt carriers ochre, unassigned inert filler green, hydrates
    orange, liquid blue, gas near-white (convex combination). The carriers get
    their own colour because lumping them with inert filler rendered a 2.6 %
    gypsum population as unreactive filler (E3 review finding)."""
    from .registry import CLINKER_PHASE_IDS as _CLK
    from .registry import SCM_PHASE_IDS as _SCM
    from .registry import SOLID_PHASE_IDS
    z = state.grid_size // 2
    clk = [SOLID_PHASE_IDS.index(p) for p in _CLK]
    scm = [SOLID_PHASE_IDS.index(p) for p in _SCM]
    salt = [SOLID_PHASE_IDS.index(p) for p in SALT_PHASE_IDS]
    other = [i for i in range(len(SOLID_PHASE_IDS))
             if i not in clk + scm + salt]
    anh = state.anhydrous_fraction[clk, z].sum(axis=0)
    scm_f = state.anhydrous_fraction[scm, z].sum(axis=0)
    salt_f = state.anhydrous_fraction[salt, z].sum(axis=0)
    inert = state.anhydrous_fraction[other, z].sum(axis=0)
    hyd = state.hydrate_fraction[:, z].sum(axis=0)
    liq = state.capillary_liquid[z]
    gas = state.capillary_gas[z]
    img = (anh[..., None] * _COL_ANHYDROUS + scm_f[..., None] * _COL_SCM
           + salt_f[..., None] * _COL_SALT
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
    notes: List[str] = []
    summary_ph: Dict[float, dict] = {}
    summary_path = run / "summary.json"
    if summary_path.is_file():
        try:
            for row in json.loads(summary_path.read_text(encoding="utf-8"))["outputs"]:
                lm = row.get("ledger_metrics", {})
                if lm.get("cluster_ph"):
                    merged = {"cluster_ph": lm["cluster_ph"]}
                    if lm.get("cluster_liq_frac"):
                        merged["cluster_liq_frac"] = lm["cluster_liq_frac"]
                    summary_ph[float(row["time_h"])] = merged
        except Exception:
            notes.append("summary.json unreadable - per-cluster pH omitted "
                         "from the report and the pH band")
    else:
        notes.append("summary.json absent (interrupted run?) - per-cluster "
                     "pH omitted from the report and the pH band")

    rows: List[dict] = []
    config = None
    backend_id = None
    first_state = None
    for ck in ckpts:
        state = load_checkpoint(str(ck), reg)
        config = state.config
        if registry is None:
            # the run's own registry (a config may declare its own SCM glass)
            reg = registry_for(config)
        backend_id = state.backend_id
        if first_state is None:
            first_state = state   # initial_phase_mol is constant across a run
        gel_eps = gel_porosity_vector(config, state.hydrate_ids, reg)
        row = state_row(state, reg, gel_eps)
        check = ledger.check_all(state, reg)
        row["ledger_metrics"] = dict(check.metrics)
        row["ledger_violations"] = list(check.violations)
        if row["time_h"] in summary_ph:
            row["ledger_metrics"].update(summary_ph[row["time_h"]])
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
        # E2: cluster-resolution Ca/Si composition map (PRD 4.5 v3.0)
        casi = cluster_ca_si(state)
        if casi:
            row["cluster_ca_si"] = {int(k): float(v) for k, v in casi.items()}
            row["casi_channel"] = casi_channel(state)
            rgb = casi_map_rgb(state)
            if rgb is not None:
                casi_name = f"casi_{ck.name}.png"
                write_png(str(out / casi_name), rgb)
                row["casi_png"] = casi_name
        rows.append(row)
    # checkpoint NAMES sort lexicographically (ckpt_1000 < ckpt_999) — time is
    # the authority for series order
    rows.sort(key=lambda r: r["time_h"])

    result = {
        "run_dir": str(run),
        "config_hash": config.config_hash(),
        "backend": backend_id,
        # E3: what SO3/alkali the recipe carries AND what the rasterized RVE
        # actually holds (mill-certificate units; the *_as_built keys expose
        # the small-population sampling gap instead of hiding it)
        "binder_oxides": binder_oxide_diagnostics(config, reg, first_state),
        "outputs": rows,
        "sanity_band": sanity_band(rows, config),
        "notes": notes,
    }
    (out / "report.json").write_text(json.dumps(result, indent=1),
                                     encoding="utf-8")
    return result
