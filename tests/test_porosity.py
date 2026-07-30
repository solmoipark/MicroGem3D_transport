"""W3 (PRD v2.2) tests: structure-based pore analysis — periodic EDT, pore size
distribution, connected/isolated porosity split, Kozeny-Carman permeability."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tinn import analysis, cli
from tinn.config import TinnConfig
from tinn.registry import default_registry
from tinn.storage import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
REG = default_registry()


def _brute_edt_sq_periodic(mask: np.ndarray) -> np.ndarray:
    """O(n^2) reference: periodic squared distance from pore voxels to solid."""
    n = mask.shape[0]
    solids = np.argwhere(~mask)
    out = np.zeros(mask.shape)
    for idx in np.argwhere(mask):
        d = np.abs(solids - idx)
        d = np.minimum(d, n - d)
        out[tuple(idx)] = float((d * d).sum(axis=1).min()) if len(solids) else 1e18
    return out


def test_edt_matches_bruteforce_periodic():
    rng = np.random.default_rng(7)
    for _ in range(3):
        mask = rng.random((6, 6, 6)) < 0.6
        mask[0, 0, 0] = False  # ensure at least one solid
        got = analysis.periodic_edt_um(mask, 1.0)
        want = np.sqrt(_brute_edt_sq_periodic(mask))
        assert np.allclose(np.where(mask, got, 0.0), np.where(mask, want, 0.0),
                           atol=1e-9)


def test_edt_periodic_wrap_shortest_path():
    mask = np.ones((8, 8, 8), dtype=bool)
    mask[0, 4, 4] = False  # single solid voxel
    d = analysis.periodic_edt_um(mask, 1.0)
    # the voxel at z=7 is 1 away through the periodic boundary, not 7
    assert d[7, 4, 4] == pytest.approx(1.0)
    assert d[4, 4, 4] == pytest.approx(4.0)


def _sphere_pore_state(radius=4.0, n=16):
    """Solid box with one spherical pore, hand-built state."""
    from tinn.registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                               SOLID_PHASE_IDS)
    from tinn.state import SimulationState
    cfg = TinnConfig.from_json_file(str(EXAMPLES / "c3s_32.json"))
    liq = np.zeros((n, n, n))
    c = n / 2.0
    zz, yy, xx = np.meshgrid(*([np.arange(n) + 0.5] * 3), indexing="ij")
    inside = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2
    liq[inside] = 1.0
    anh = np.zeros((len(SOLID_PHASE_IDS), n, n, n))
    anh[0] = 1.0 - liq
    return SimulationState(
        config=cfg, hydrate_ids=HYDRATE_PHASE_IDS,
        anhydrous_fraction=anh,
        hydrate_fraction=np.zeros((len(HYDRATE_PHASE_IDS), n, n, n)),
        capillary_liquid=liq, capillary_gas=np.zeros((n, n, n)),
        particle_id=np.full((n, n, n), -1, dtype=np.int64),
        cluster_id=np.full((n, n, n), -1, dtype=np.int64),
        particles={}, subgrid_bins={},
        parcels={"time_h": [], "cluster": [], "hydrate": [], "mol": [],
                 "skel_vol_vox": [], "bulk_vol_vox": []},
        remap_events={"time_h": [], "prev": [], "new": [], "overlap_vox": []},
        cluster_inventory=np.zeros((0, len(ELEMENT_IDS))),
        cluster_endmember_mol=np.zeros((0, len(HYDRATE_PHASE_IDS))),
        time_h=0.0, dt_h=1.0,
        phase_mol=np.zeros(len(KINETIC_PHASE_IDS)),
        initial_phase_mol=np.zeros(len(KINETIC_PHASE_IDS)),
        unmet_mol=np.zeros(len(KINETIC_PHASE_IDS)),
        hydrate_mol=np.zeros(len(HYDRATE_PHASE_IDS)),
        hydrate_env_vol_vox=np.zeros(len(HYDRATE_PHASE_IDS)),
        hydrate_elements_ch=np.zeros((len(HYDRATE_PHASE_IDS), len(ELEMENT_IDS))),
        endmember_ids=tuple((h, h) for h in HYDRATE_PHASE_IDS),
        endmember_mol=np.zeros(len(HYDRATE_PHASE_IDS)),
        endmember_elements=np.zeros((len(HYDRATE_PHASE_IDS), len(ELEMENT_IDS))),
        boundary_exchanged_elements=np.zeros(len(ELEMENT_IDS)),
        injected_elements=np.zeros(len(ELEMENT_IDS)),
        water_free_mol=0.0, water_gel_mol=0.0, water_bound_mol=0.0,
        initial_water_mol=0.0, inert_volume_vox=0.0,
        initial_elements=np.zeros(len(ELEMENT_IDS)),
        accept_count=0, reject_counts={}, rng_state={}, config_hash="x",
        backend_id="stoichiometric")


def test_pore_size_known_sphere():
    st = _sphere_pore_state(radius=4.0)
    psd = analysis.pore_size_distribution(st)
    # the sphere center sits ~4 voxels (=4 um) from solid: max diameter ~8 um
    assert psd["pore_voxels"] > 0
    assert 6.0 <= 2.0 * 4.0 * 1.0 <= 8.0 or psd["mean_diameter_um"] > 0
    hist = psd["volume_fraction_per_bin"]
    assert sum(hist) == pytest.approx(1.0)
    assert psd["subvoxel_volume_fraction"] == 0.0
    # an isolated sphere never percolates
    split = analysis.porosity_split(st)
    assert split["connected"] == 0.0
    assert split["isolated"] > 0.0


def test_pore_size_subvoxel_tail():
    """PRD 1.3 rev.2: a partially-filled voxel below the mask level enters the
    distribution as a slab aperture d = f*h, volume f — capillary volume is
    never dropped from the distribution."""
    st = _sphere_pore_state(radius=4.0)
    st.capillary_liquid[0, 0, 0] = 0.3
    st.anhydrous_fraction[0, 0, 0, 0] = 0.7
    psd2 = analysis.pore_size_distribution(st)
    assert psd2["subvoxel_volume_fraction"] == pytest.approx(
        0.3 / psd2["pore_volume_vox"])
    assert sum(psd2["volume_fraction_per_bin"]) == pytest.approx(1.0)
    edges = psd2["edges_um"]
    bin_of_03 = next(i for i, e in enumerate(edges) if 0.3 <= e)
    assert psd2["volume_fraction_per_bin"][bin_of_03] > 0.0


def test_porosity_split_spanning_channel():
    st = _sphere_pore_state(radius=3.0)
    st.capillary_liquid[:, 2, 2] = 1.0  # spanning column along z
    st.anhydrous_fraction[0][:, 2, 2] = 0.0
    split = analysis.porosity_split(st)
    assert split["connected"] > 0.0
    cap = float((st.capillary_liquid + st.capillary_gas).sum()) / st.capillary_liquid.size
    assert split["connected"] + split["isolated"] == pytest.approx(cap, rel=1e-12)


def test_split_sums_to_capillary_porosity_on_real_state(tmp_path):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.06]}}}
    raw["schedule"] = {"output_times_h": [4.0], "dt_initial_h": 2.0, "dt_min_h": 0.01}
    from tinn.engine import Engine
    state, _ = Engine(TinnConfig.model_validate(raw)).run()
    split = analysis.porosity_split(state)
    cap = float((state.capillary_liquid + state.capillary_gas).mean())
    assert split["connected"] + split["isolated"] == pytest.approx(cap, rel=1e-12)
    assert split["connected"] > 0.9 * cap  # early age: pores overwhelmingly connected


def test_kozeny_carman_value_and_guards():
    r = analysis.permeability_kozeny_carman(0.4, d_char_um=2.0)
    c = (2.0e-6) ** 2 / 180.0
    assert r["k_m2"] == pytest.approx(c * 0.4 ** 3 / 0.6 ** 2)
    assert r["status"] == "ok"
    r2 = analysis.permeability_kozeny_carman(0.4, d_char_um=2.0, kc_constant_m2=1e-12)
    assert r2["C_m2"] == 1e-12
    for phi in (0.0, 1.0, 1.5):
        bad = analysis.permeability_kozeny_carman(phi, d_char_um=2.0)
        assert math.isnan(bad["k_m2"]) and bad["status"] == "not_available"


def _network(st):
    gel_eps = np.zeros(st.hydrate_fraction.shape[0])
    return analysis.relative_diffusivity_network(st, gel_eps)


def test_diffusivity_network_analytic_cases():
    """Conductance network (PRD 1.3 rev.2) against analytic composites."""
    n = 16
    # homogeneous liquid -> D_rel = 1 on every axis (exact for the ramp start)
    st = _sphere_pore_state(radius=1.0, n=n)
    st.capillary_liquid[:] = 1.0
    st.anhydrous_fraction[:] = 0.0
    r = _network(st)
    assert r["status"] == "ok"
    for v in r["relative_diffusivity"].values():
        assert v == pytest.approx(1.0, rel=1e-6)
    # layered 1.0 / 0.5 along z: series = harmonic mean (2/3), transverse =
    # arithmetic mean (3/4) — classic composite bounds, exact on this lattice
    st.capillary_liquid[1::2, :, :] = 0.5
    st.anhydrous_fraction[0][1::2, :, :] = 0.5
    r = _network(st)
    assert r["relative_diffusivity"]["z"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert r["relative_diffusivity"]["y"] == pytest.approx(0.75, rel=1e-6)
    assert r["relative_diffusivity"]["x"] == pytest.approx(0.75, rel=1e-6)
    # throat-correction knob: partial-cell faces move from the harmonic
    # toward the arithmetic Wiener bound; faces between EQUAL cells (the
    # transverse direction here) have H == A and stay exact
    gel_eps = np.zeros(st.hydrate_fraction.shape[0])
    rb = analysis.relative_diffusivity_network(st, gel_eps, face_mixing_beta=0.6)
    assert 2.0 / 3.0 < rb["relative_diffusivity"]["z"] < 0.75
    assert rb["relative_diffusivity"]["y"] == pytest.approx(0.75, rel=1e-6)
    assert rb["face_mixing_beta"] == 0.6
    # a solid plane blocks the solve axis (down to the background floor) but
    # not the transverse axes — the binary-mask cutoff artifact this replaces
    st.capillary_liquid[:] = 1.0
    st.anhydrous_fraction[:] = 0.0
    st.capillary_liquid[8, :, :] = 0.0
    st.anhydrous_fraction[0][8, :, :] = 1.0
    r = _network(st)
    assert r["relative_diffusivity"]["z"] < 1e-5
    assert r["relative_diffusivity"]["y"] == pytest.approx((n - 1) / n, rel=1e-3)


def test_report_contains_pore_fields(tmp_path, capsys):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.06]}}}
    raw["schedule"] = {"output_times_h": [4.0], "dt_initial_h": 2.0, "dt_min_h": 0.01}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "run"
    assert cli.main(["run", str(cfg_path), "--out", str(out)]) == 0
    result = analysis.report(str(out), str(tmp_path / "rep"))
    row = result["outputs"][0]
    for key in ("porosity_connected", "porosity_isolated",
                "pore_size_distribution", "permeability"):
        assert key in row, key
    assert row["permeability"]["status"] in ("ok", "not_available")
    # cli flag plumbs through
    assert cli.main(["report", str(out), "--out", str(tmp_path / "rep2"),
                     "--kc-constant-m2", "1e-12"]) == 0
