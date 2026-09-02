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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import ELEMENT_IDS

LIQ_EPS = 1e-9
_AXES = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))
# transport-frozen dust threshold: a domain whose liquid volume is below this
# fraction of the mean wet-domain volume gets no edges (mirrors the engine's
# trace-water chemistry freeze and bounds the BE matrix contrast)
DUST_WATER_REL = 1e-6


def label_clusters(liquid: np.ndarray,
                   periodic_axes: Tuple[bool, bool, bool] = (True, True, True)
                   ) -> Tuple[np.ndarray, int]:
    """Return (labels int64 with -1 for dry voxels, cluster count).

    v4.0/RT-W3: a non-periodic axis (a declared exposed face de-periodizes
    the WHOLE axis) suppresses the min-label flood across that seam — a
    cluster must not share one well-mixed solution across the cut. The
    all-periodic default path is byte-identical (branch only, zero extra
    array ops)."""
    mask = liquid > LIQ_EPS
    n_vox = liquid.size
    labels = np.where(mask, np.arange(n_vox, dtype=np.int64).reshape(liquid.shape),
                      np.int64(n_vox))
    seams = []          # per-(ax, shift): wrapped slice to blank, None = periodic
    for ax, shift in _AXES:
        if periodic_axes[ax]:
            seams.append(None)
        else:
            idx: list = [slice(None)] * 3
            idx[ax] = 0 if shift > 0 else -1   # np.roll(+1) wraps N-1 into 0
            seams.append(tuple(idx))
    # min-label flooding advances >= 1 voxel per sweep along the longest geodesic,
    # which is bounded by the voxel count — never bail out on a legal topology
    for _ in range(n_vox + 1):
        prev = labels
        m = labels
        for (ax, shift), cut in zip(_AXES, seams):
            r = np.roll(labels, shift, axis=ax)
            if cut is not None:
                r[cut] = n_vox         # the wrapped face is not a neighbor
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


# O and H are the solute FRAME (values relative to H2O): they may carry
# either sign in the pore-solution ledger (PRD 4.6.5 S1-OPEN-1)
_FRAME_COLS = frozenset(ELEMENT_IDS.index(el) for el in ("O", "H"))


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


def label_domains(labels: np.ndarray, n_clusters: int,
                  tiles: Tuple[int, int, int]
                  ) -> Tuple[np.ndarray, int, np.ndarray]:
    """Equilibration domains = connected cluster INTERSECTED with a static
    axis-aligned tiling (PRD 4.6.2), per-axis tile edges (z, y, x) — banded
    tilings (transverse = grid) are the planar-front cost lever of the W3
    design review. All tiles == grid returns the cluster labels themselves,
    the literal legacy identity the 1-domain bit-compare gate relies on.
    Ids are consecutive, ordered by (cluster id, tile index): deterministic,
    restart-stable, periodic-aware (each tile edge divides the grid).

    Returns (domain_id with -1 dry, n_domains, domain_to_cluster)."""
    n = labels.shape[0]
    if all(t == n for t in tiles):
        return labels, n_clusters, np.arange(n_clusters, dtype=np.int64)
    counts = []
    idx = []
    for ax, t in enumerate(tiles):
        if n % t != 0:
            raise ValueError(f"tile {t} (axis {ax}) does not divide grid {n}")
        counts.append(n // t)
        idx.append(np.arange(n, dtype=np.int64) // t)
    nz, ny, nx = counts
    tile = ((idx[0][:, None, None] * ny + idx[1][None, :, None]) * nx
            + idx[2][None, None, :])
    n_tiles = nz * ny * nx
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
    harmonic face conductances of each interface divided by the TPFA
    center distance (geometric transmissibility: T = d0 * edge_g); water
    is the liquid volume per domain (vox^3). Dust domains (water below
    DUST_WATER_REL of the mean wet-domain water) carry no edges -
    transport-frozen."""
    n_domains: int
    edge_a: np.ndarray          # (E,) int64, edge_a < edge_b
    edge_b: np.ndarray          # (E,) int64
    edge_g: np.ndarray          # (E,) float64
    water: np.ndarray           # (D,) float64
    dust: np.ndarray            # (D,) bool


def build_domain_graph(domain_id: np.ndarray, n_domains: int,
                       g: np.ndarray, liquid: np.ndarray,
                       periodic_axes: Tuple[bool, bool, bool] = (True, True,
                                                                 True),
                       pitch_zyx: Tuple[float, float, float] = (1.0, 1.0,
                                                                1.0)
                       ) -> DomainGraph:
    """Edges between face-adjacent wet voxels of DIFFERENT domains (always
    domains of the same cluster, by construction of the 6-neighbor flood).
    Face conductance is the harmonic mean 2ab/(a+b) - the same rule as the
    report's diffusivity network - summed over the interface and DIVIDED by
    that axis's TPFA center distance pitch_zyx[ax] (the tile edge along the
    interface normal — per-axis since W4's anisotropic tiles). edge_g is
    therefore geometric transmissibility: T = d0 * edge_g. Wrap faces
    included on periodic axes; a non-periodic (exposed) axis drops its
    seam pair — MANDATORY once labels are de-periodized, else the seam
    edge would join different clusters (v4.0/RT-W3). Deterministic order."""
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
        if not periodic_axes[ax]:
            cut: list = [slice(None)] * 3
            cut[ax] = -1
            m[tuple(cut)] = False      # exposed-axis wrap faces are walls
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
        conds.append(face / float(pitch_zyx[ax]))
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
class BoundaryBath:
    """v4.0/RT-W3 fixed-composition ghost reservoir (PRD 4.6.3). g_bnd is
    the per-domain half-cell coupling ALREADY divided by the exposed
    axis's tile pitch (geometric transmissibility, like edge_g): the
    caller builds boundary_coupling(...) / pitch, where boundary_coupling
    carries the network solver's Dirichlet factor Σ 2·g_a — each scaling
    lives in exactly one place. exchange_be applies d0 only. c_res is the
    bath concentration per element in mol per vox^3 of liquid (c = n/W
    units)."""
    g_bnd: np.ndarray            # (D,) float64 >= 0; 0 = uncoupled
    c_res: np.ndarray            # (E,) float64 >= 0


def boundary_coupling(domain_id: np.ndarray, n_domains: int, g: np.ndarray,
                      axis: int, low: bool, high: bool) -> np.ndarray:
    """Half-cell conductance sum to the declared exposed face(s): Σ 2g over
    WET (domain_id >= 0) voxels of the face layer (index 0 / N-1 along
    axis), grouped by domain (bincount — deterministic). Dry face voxels
    never couple (no domain; moisture flux is out of scope). The caller
    zeroes dust domains, mirroring the internal edge exclusion."""
    out = np.zeros(n_domains)
    for idx, on in ((0, low), (-1, high)):
        if not on:
            continue
        sl: list = [slice(None)] * 3
        sl[axis] = idx
        d = domain_id[tuple(sl)]
        gv = g[tuple(sl)]
        m = d >= 0
        if m.any():
            out += np.bincount(d[m], weights=2.0 * gv[m],
                               minlength=n_domains)
    return out


# ---------------------------------------------------------------- Tier 0 (NP)
# RT-P0 (PRD 4.6.4): per-element effective conductance from frozen aqueous
# speciation under the Nernst-Planck zero-current projection. Pure functions;
# the BE solver stays in element space and reuses its block CG per column.

# small-|dc_el| guard: below this relative element-concentration difference
# the F/dc quotient is ill-conditioned and the weighted-Fick fallback is used
NP_DC_REL_EPS = 1e-9

# water dynamic viscosity, mPa*s, 0..100 C in 5 C steps (IAPWS-anchored
# tabulation, e.g. CRC Handbook); linear interpolation between nodes
_WATER_VISC_T_C = np.arange(0.0, 101.0, 5.0)
_WATER_VISC_MPA_S = np.array([
    1.7914, 1.5192, 1.3077, 1.1382, 1.0016, 0.8900, 0.7972, 0.7191,
    0.6527, 0.5958, 0.5465, 0.5036, 0.4660, 0.4329, 0.4035, 0.3774,
    0.3540, 0.3330, 0.3142, 0.2971, 0.2816])


def stokes_einstein_factor(temperature_k: float) -> float:
    """D(T)/D(298.15 K) = (T/298.15) * (eta(298.15)/eta(T)) — first-order
    temperature correction of the 25 C dw table (PRD 4.6.4). Exactly 1.0 at
    298.15 K; refuses outside the 0-100 C tabulation."""
    t_c = float(temperature_k) - 273.15
    if not (0.0 <= t_c <= 100.0):
        raise ValueError(
            f"stokes_einstein_factor needs 0..100 C, got {t_c:.2f} C")
    if temperature_k == 298.15:
        return 1.0
    eta = float(np.interp(t_c, _WATER_VISC_T_C, _WATER_VISC_MPA_S))
    eta_25 = float(np.interp(25.0, _WATER_VISC_T_C, _WATER_VISC_MPA_S))
    return (float(temperature_k) / 298.15) * (eta_25 / eta)


@dataclass
class NPConductance:
    """Per-edge, per-element transmissibilities from one frozen speciation
    state. t_edge/t_bnd are what exchange_be consumes in place of
    d0 * edge_g / d0 * g_bnd; deff_edge is diagnostics (vox^2/h)."""
    t_edge: np.ndarray                    # (n_edges, n_elem) vox^3/h
    t_bnd: Optional[np.ndarray]           # (n_domains, n_elem) or None
    deff_edge: np.ndarray                 # (n_edges, n_elem) vox^2/h
    counts: Dict[str, int] = field(default_factory=dict)
    charge_flux_rel_max: float = 0.0      # zero-current witness (unclamped)


def _np_deff(dc_s, cbar_s, dc_el, cmax_el, dw, z, nu):
    """Clamp-laddered effective diffusivity (rows, n_elem) for one batch of
    faces. dc_s/cbar_s are (rows, S) species concentration difference and
    face-mean; dc_el is the BE driving force per element (the caller's
    convention: species collapse for interior edges, x - c_res for bath
    faces); cmax_el scales the small-difference guard. Returns
    (deff, counts, charge_rel_max). PRD 4.6.4 ladder: (1) |dc_el| dust ->
    weighted-Fick D-bar; (2) F_NP / dc_el; (3) negative -> drop the Phi term
    (pure Fick); (4) still negative -> D-bar; (5) cap at D_max over the
    species present on the face. 0 <= deff <= D_max keeps the M-matrix."""
    rows, n_elem = dc_el.shape
    counts = {"np_smalldc": 0, "np_phi_clamped": 0, "np_fick_clamped": 0,
              "np_cap_clamped": 0, "np_empty_el": 0}
    if rows == 0:
        return np.zeros((0, n_elem)), counts, 0.0
    zd = z * dw
    den = cbar_s @ (z * zd)                        # (rows,) sum z^2 D cbar
    num = dc_s @ zd                                # (rows,) sum z D dc
    ok_den = den > 0.0
    phi = np.where(ok_den, num / np.where(ok_den, den, 1.0), 0.0)
    s1 = (dc_s * dw[None, :]) @ nu                 # (rows, n_elem) Fick part
    s2 = (cbar_s * zd[None, :]) @ nu               # migration weight
    cbar_el = cbar_s @ nu
    dbar_num = (cbar_s * dw[None, :]) @ nu
    has_el = cbar_el > 0.0
    dbar = np.where(has_el, dbar_num / np.where(has_el, cbar_el, 1.0), 0.0)
    dmax = np.where(cbar_s > 0.0, dw[None, :], 0.0).max(axis=1)  # (rows,)

    small = np.abs(dc_el) <= NP_DC_REL_EPS * cmax_el
    safe_dc = np.where(small, 1.0, dc_el)
    full = (s1 - phi[:, None] * s2) / safe_dc      # rung 2
    fick = s1 / safe_dc                            # rung 3 fallback
    neg_full = (~small) & (full < 0.0)
    neg_fick = neg_full & (fick < 0.0)             # rung 4
    deff = np.where(small, dbar,
                    np.where(neg_full, np.where(neg_fick, dbar, fick), full))
    # rung 5 — relative tolerance so exact-equality cases (all-equal D, where
    # deff == dmax to rounding) do not count as clamps
    over = deff > dmax[:, None] * (1.0 + 1e-12)
    deff = np.where(over, dmax[:, None], deff)
    deff = np.where(has_el, np.maximum(deff, 0.0), 0.0)

    counts["np_smalldc"] = int(np.count_nonzero(small & has_el))
    counts["np_phi_clamped"] = int(np.count_nonzero(neg_full & ~neg_fick))
    counts["np_fick_clamped"] = int(np.count_nonzero(neg_fick))
    counts["np_cap_clamped"] = int(np.count_nonzero(over & has_el))
    counts["np_empty_el"] = int(np.count_nonzero(~has_el))

    # zero-current witness on the species-level projected flux, faces where
    # the projection actually applied (den > 0): sum_i z_i J_i must vanish
    f_s = dc_s * dw[None, :] - phi[:, None] * (cbar_s * zd[None, :])
    q_net = np.abs(f_s @ z)
    q_abs = np.abs(f_s * z[None, :]).sum(axis=1)
    wit = ok_den & (q_abs > 0.0)
    charge_rel_max = float((q_net[wit] / q_abs[wit]).max()) if wit.any() else 0.0
    return deff, counts, charge_rel_max


def np_effective_conductance(graph: DomainGraph, species_mol: np.ndarray,
                             water: np.ndarray, dw_vox2_h: np.ndarray,
                             z: np.ndarray, nu: np.ndarray,
                             g_bnd: Optional[np.ndarray] = None,
                             c_res: Optional[np.ndarray] = None,
                             c_res_species: Optional[np.ndarray] = None
                             ) -> NPConductance:
    """Edge- and element-resolved transmissibilities from one frozen
    speciation state (PRD 4.6.4). species_mol is (D, S) aqueous species mol
    per domain (solvent excluded), water the domain liquid volume (vox^3;
    c = n/W units), dw_vox2_h the species diffusivities ALREADY carrying the
    Stokes-Einstein factor and the geometry factor. Interior faces use the
    harmonic face mean (the existing face mixing rule). Bath faces: with no
    reservoir speciation (c_res_species None - the aerated-water bath) the
    one-sided domain state is used (a harmonic mean against zero would
    kill the migration term identically); with a speciated solute bath
    (RT-P0d) the species difference is c - s_res and the face mean is the
    harmonic mean wherever BOTH sides carry the species, one-sided
    otherwise - so an all-zero s_res reproduces the O/H path bitwise.
    Bath O/H in c_res only shifts the element driving force, matching the
    BE's (x - c_res) form exactly."""
    ea, eb = graph.edge_a, graph.edge_b
    n_elem = nu.shape[1]
    w = np.maximum(np.asarray(water, dtype=np.float64), 0.0)
    with np.errstate(invalid="ignore"):
        c = np.where(w[:, None] > 0.0,
                     np.maximum(species_mol, 0.0)
                     / np.where(w[:, None] > 0.0, w[:, None], 1.0), 0.0)
    ca, cb = c[ea], c[eb]
    dc_s = ca - cb
    ssum = ca + cb
    cbar_s = np.where(ssum > 0.0, 2.0 * ca * cb / np.where(ssum > 0.0, ssum, 1.0),
                      0.0)
    dc_el = dc_s @ nu
    cmax_el = np.maximum(ca @ nu, cb @ nu)
    deff, counts, qrel = _np_deff(dc_s, cbar_s, dc_el, cmax_el,
                                  dw_vox2_h, z, nu)
    t_edge = graph.edge_g[:, None] * deff

    t_bnd = None
    if g_bnd is not None:
        res = (np.zeros(n_elem) if c_res is None
               else np.asarray(c_res, dtype=np.float64))
        dc_el_b = c @ nu - res[None, :]
        cmax_b = np.maximum(c @ nu, np.abs(res)[None, :])
        if c_res_species is None:
            dc_b, cbar_b = c, c
        else:
            s_res = np.asarray(c_res_species, dtype=np.float64)[None, :]
            dc_b = c - s_res
            both = (c > 0.0) & (s_res > 0.0)
            cbar_b = np.where(both,
                              2.0 * c * s_res / np.where(both, c + s_res, 1.0),
                              c)
        deff_b, counts_b, qrel_b = _np_deff(dc_b, cbar_b, dc_el_b, cmax_b,
                                            dw_vox2_h, z, nu)
        t_bnd = np.asarray(g_bnd, dtype=np.float64)[:, None] * deff_b
        for k, v in counts_b.items():
            counts[k] += v
        qrel = max(qrel, qrel_b)
    return NPConductance(t_edge=t_edge, t_bnd=t_bnd, deff_edge=deff,
                         counts=counts, charge_flux_rel_max=qrel)


@dataclass
class ExchangeResult:
    delta: np.ndarray            # (D, E) antisymmetric inventory change
    status: str                  # "ok" | "not_converged"
    cg_iterations: int
    max_edge_flux_mol: float
    repair_rel: float            # largest relative negative-dust repair
    boundary_net: np.ndarray     # (E,) net element flux INTO the system


def exchange_be(graph: DomainGraph, inventory: np.ndarray, dt_h: float,
                d0_vox2_h: Optional[float] = None,
                bath: Optional[BoundaryBath] = None,
                tol: float = 1e-12, max_iter: int = 10000, *,
                np_cond: Optional[NPConductance] = None) -> ExchangeResult:
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
    # RT-P0b: exactly one conductance source — the scalar D0 (as (E,1)/(D,1)
    # broadcast views, so every column sees the SAME float sequence as the
    # historical 1D path: bit-identity by construction, not by luck) or the
    # per-edge/per-element NP transmissibilities.
    if (d0_vox2_h is None) == (np_cond is None):
        raise ValueError("exchange_be needs exactly one of d0_vox2_h/np_cond")
    ea, eb = graph.edge_a, graph.edge_b
    # RT-W3: a bath-coupled domain participates even with no internal
    # edges (an isolated surface-connected pore physically leaches)
    t_bnd2 = None
    if np_cond is None:
        t_e2 = (d0_vox2_h * graph.edge_g)[:, None]      # (E, 1) vox^3/h
        if bath is not None and np.any(bath.g_bnd > 0.0):
            t_bnd2 = (d0_vox2_h * bath.g_bnd)[:, None]  # (D, 1)
    else:
        t_e2 = np_cond.t_edge                            # (E, n_elem)
        if bath is not None:
            if np_cond.t_bnd is None:
                raise ValueError(
                    "bath given but np_cond carries no boundary "
                    "transmissibilities")
            if np.any(np_cond.t_bnd > 0.0):
                t_bnd2 = np_cond.t_bnd                   # (D, n_elem)
    if dt_h <= 0.0 or (graph.edge_a.size == 0 and t_bnd2 is None):
        return ExchangeResult(delta, "ok", 0, 0.0, 0.0, np.zeros(n_elem))
    # only edge-connected (or bath-coupled) domains participate; the
    # diagonal floors at a tiny positive value on those rows
    act = np.zeros(n_dom, dtype=bool)
    act[ea] = True
    act[eb] = True
    if t_bnd2 is not None:
        act |= (t_bnd2 > 0.0).any(axis=1)
    w_act = np.where(act, np.maximum(graph.water, 1e-300), 1.0)
    deg = np.zeros((n_dom, t_e2.shape[1]))
    np.add.at(deg, ea, t_e2)
    np.add.at(deg, eb, t_e2)
    # bath term: known c_R eliminated to the RHS — diagonal gains dt*T_bnd
    # (SPD/M-matrix preserved, diagonal dominance improves), RHS gains
    # dt*T_bnd*c_res (PRD 4.6.3). The sealed path aliases w_eff = w_act.
    w2 = w_act[:, None]
    w_eff = w2 if t_bnd2 is None else w2 + dt_h * t_bnd2
    diag = w_eff + dt_h * deg                            # (D, cols)
    b = inventory * act[:, None]
    if t_bnd2 is not None:
        b = b + (dt_h * t_bnd2) * bath.c_res[None, :]

    def matvec(x):
        y = w_eff * x
        d = x[ea] - x[eb]
        contrib = dt_h * t_e2 * d
        np.add.at(y, ea, contrib)
        np.add.at(y, eb, -contrib)
        return y

    # per-column normalization: repeated bath drains push inventories to
    # denormal scale (measured 1e-155), where rz underflows, beta blows up
    # and CG overflows to NaN. The system is linear - solve in O(1) scale
    # and rescale the solution (deterministic, columnwise).
    col_scale = np.maximum(np.abs(b).sum(axis=0), 1e-300)
    b = b / col_scale[None, :]
    x = b / diag
    r = b - matvec(x)
    z = r / diag
    p = z.copy()
    rz = (r * z).sum(axis=0)
    iters = 0
    for iters in range(1, max_iter + 1):
        if np.all(np.abs(r).sum(axis=0) <= tol):
            break
        ap = matvec(p)
        pap = (p * ap).sum(axis=0)
        alpha = np.where(pap > 0.0, rz / np.where(pap > 0.0, pap, 1.0), 0.0)
        x = x + alpha[None, :] * p
        r = r - alpha[None, :] * ap
        z = r / diag
        rz_new = (r * z).sum(axis=0)
        beta = np.where(rz > 0.0, rz_new / np.where(rz > 0.0, rz, 1.0), 0.0)
        p = z + beta[None, :] * p
        rz = rz_new
    else:
        return ExchangeResult(delta, "not_converged", iters, 0.0, 0.0,
                              np.zeros(n_elem))
    x = x * col_scale[None, :]

    flux = dt_h * t_e2 * (x[ea] - x[eb])            # (n_edges, n_elem)
    np.add.at(delta, ea, -flux)
    np.add.at(delta, eb, flux)
    max_flux = float(np.abs(flux).max()) if flux.size else 0.0
    boundary_net = np.zeros(n_elem)
    if t_bnd2 is not None:
        rows = np.flatnonzero((t_bnd2 > 0.0).any(axis=1))  # ascending, deterministic
        f_out = dt_h * t_bnd2[rows] * (x[rows] - bath.c_res[None, :])
        delta[rows] -= f_out                          # the same floats feed
        boundary_net = -f_out.sum(axis=0)             # rows AND the ledger
        if f_out.size:
            max_flux = max(max_flux, float(np.abs(f_out).max()))

    # deterministic negative-dust repair: clip, charge the largest entry.
    # O and H are FRAME elements (solute-frame values relative to H2O):
    # a negative entry there is water-frame acid (an OH- deficit), a
    # legitimate state after ligand exchange desorbs at an OH-depleted
    # face (PRD 4.6.5 S1-OPEN-1) - the BE solve is linear and signed, so
    # those columns skip the repair; solutes stay amounts (>= 0).
    new = inventory + delta
    repair_rel = 0.0
    for e in range(n_elem):
        if e in _FRAME_COLS:
            continue
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
    return ExchangeResult(delta, "ok", iters, max_flux, repair_rel,
                          boundary_net)
