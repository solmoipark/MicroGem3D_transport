"""SCM extension tests (PRD v2.1): oxide-glass registry entries, logistic
reaction schedules, blended kinetics, and the 3D GEMS blend path.

Compositions/densities and logistic parameters mirror InverseGems
(configs/materials.yaml, configs/scm_reaction.yaml)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tinn import ledger
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.kinetics import (SCM_LOGISTIC_PRESETS, ParrotKilloh,
                           scm_alpha, scm_effective_params)
from tinn.registry import (ATOMIC_MASS_G_MOL, CLINKER_PHASE_IDS, ELEMENT_IDS,
                           KINETIC_PHASE_IDS, RegistryError, SALT_PHASE_IDS,
                           SCM_PHASE_IDS, SOLID_PHASE_IDS, default_registry,
                           scm_entry)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))
needs_gems = pytest.mark.skipif(
    not (BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")

REG = default_registry()
FA30 = {"C3S": 0.46353, "C2S": 0.05817, "C3A": 0.07549, "C4AF": 0.05107,
        "fly_ash": 0.30}


# ---------------- registry: oxide glasses ----------------

def test_kinetic_phase_order():
    # positional ledger contract: clinker, then SCM, then the E3 salt carriers
    # (appended, so every pre-E3 index keeps its meaning)
    assert KINETIC_PHASE_IDS == CLINKER_PHASE_IDS + SCM_PHASE_IDS + SALT_PHASE_IDS
    assert SCM_PHASE_IDS == ("slag", "fly_ash", "metakaolin", "silica_fume")
    assert SALT_PHASE_IDS == ("gypsum", "hemihydrate", "anhydrite",
                              "arcanite", "thenardite")


def test_fly_ash_registry_entry_matches_composition():
    fa = REG.get("fly_ash")
    # Si mol per 100 g = SiO2 wt% / M(SiO2)
    m_sio2 = ATOMIC_MASS_G_MOL["Si"] + 2 * ATOMIC_MASS_G_MOL["O"]
    assert fa.formula["Si"] == pytest.approx(51.8 / m_sio2, rel=1e-12)
    assert fa.formula["Na"] == pytest.approx(2 * 1.3 / (2 * ATOMIC_MASS_G_MOL["Na"]
                                                        + ATOMIC_MASS_G_MOL["O"]), rel=1e-12)
    # the material density round-trips exactly through the molar volume
    assert fa.density_g_cm3 == pytest.approx(2.20, rel=1e-12)
    assert fa.kind == "scm" and fa.basis == "solid_skeleton"


def test_all_scm_entries_registered_with_densities():
    expected = {"slag": 2.90, "fly_ash": 2.20, "metakaolin": 2.50,
                "silica_fume": 2.20}
    for pid, rho in expected.items():
        e = REG.get(pid)
        assert e.density_g_cm3 == pytest.approx(rho, rel=1e-12)
        # every element must live in the ledger basis
        assert set(e.formula) <= set(ELEMENT_IDS)


def test_scm_entry_rejects_unknown_oxide():
    with pytest.raises(RegistryError):
        scm_entry("bogus", {"TiO2": 100.0}, 2.5)


# ---------------- logistic schedules ----------------

def test_scm_alpha_anchors_match_inversegems():
    fa = SCM_LOGISTIC_PRESETS["fly_ash"]
    # anchors computed with InverseGems scm_reaction.scm_alpha
    for days, expect in ((1.0, 0.0093), (7.0, 0.0623), (28.0, 0.1767),
                         (360.0, 0.3681)):
        assert scm_alpha(days, fa) == pytest.approx(expect, abs=5e-4)
    assert scm_alpha(0.0, fa) == 0.0
    # long-term ceiling is D
    assert scm_alpha(1e6, fa) == pytest.approx(0.40, abs=1e-3)


def test_availability_modifier_matches_inversegems_forward_run():
    # OPC70/FA30: R = (0.46353 + 0.3*0.05817)/(0.30*0.75) / ref -> D_eff caps
    # at the absolute max 0.60, and alpha(360 d) must reproduce the actual
    # InverseGems forward run value 0.5522 (run_20260715_063106_3245d9c4)
    eff = scm_effective_params(FA30)
    assert eff["fly_ash"][3] == pytest.approx(0.60, rel=1e-12)
    assert scm_alpha(360.0, eff["fly_ash"]) == pytest.approx(0.5522, abs=5e-4)
    # a clinker-poor blend gets its ultimate degree scaled DOWN
    lean = {"C3S": 0.13244, "C2S": 0.01662, "fly_ash": 0.80}
    eff_lean = scm_effective_params(lean)
    assert eff_lean["fly_ash"][3] < 0.20
    # no SCM -> reference parameters untouched
    assert scm_effective_params({"C3S": 1.0}) == SCM_LOGISTIC_PRESETS


def test_blended_kinetics_clinker_unchanged_and_scm_filled():
    opc = {"C3S": 0.60, "C2S": 0.14, "C3A": 0.07, "C4AF": 0.10}
    pk_pure = ParrotKilloh("pk_elakneswaran_2018", 0.5, 293.15, 381.8, opc)
    pk_blend = ParrotKilloh("pk_elakneswaran_2018", 0.5, 293.15, 381.8, FA30)
    a_pure = pk_pure.alpha_at(168.0)
    a_blend = pk_blend.alpha_at(168.0)
    # clinker trajectories are identical (retardation uses clinker-only alpha)
    assert a_blend[:4] == pytest.approx(a_pure[:4], rel=1e-12)
    fa_i = KINETIC_PHASE_IDS.index("fly_ash")
    eff = scm_effective_params(FA30)["fly_ash"]
    assert a_blend[fa_i] == pytest.approx(scm_alpha(7.0, eff), rel=1e-12)
    # phases with no mass have no target
    assert a_pure[fa_i] == 0.0
    assert a_blend[KINETIC_PHASE_IDS.index("slag")] == 0.0


# ---------------- config ----------------

def _blend_raw(**overrides):
    raw = json.loads((EXAMPLES / "opc_gems_32.json").read_text(encoding="utf-8"))
    raw["binder"] = {"mass_fractions": dict(FA30)}
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw.update(overrides)
    return raw


def test_config_accepts_scm_recipe():
    cfg = TinnConfig.model_validate(_blend_raw())
    assert cfg.binder.mass_fractions["fly_ash"] == 0.30
    assert cfg.binder.unassigned == pytest.approx(1.0 - sum(FA30.values()))


def test_config_rejects_scm_with_stoichiometric_backend():
    raw = _blend_raw(chemistry={"backend": "stoichiometric"})
    with pytest.raises(ValidationError, match="no reaction rule"):
        TinnConfig.model_validate(raw)


# ---------------- geometry: real SCM density ----------------

def test_geometry_blend_uses_scm_density():
    from tinn.geometry import initialize_rve
    cfg = TinnConfig.model_validate(_blend_raw())
    rve = initialize_rve(cfg, REG)
    vols = {p: float(rve.anhydrous_fraction[rve.phase_ids.index(p)].sum())
            for p in ("C3S", "fly_ash")}
    # volume ratio = (mass/density) ratio: FA is much lighter than C3S.
    # populations sample independently now, so the ratio carries each
    # material's own (reported) sampling error instead of being exact
    expect = (0.30 / 2.20) / (0.46353 / REG.get("C3S").density_g_cm3)
    assert vols["fly_ash"] / vols["C3S"] == pytest.approx(expect, rel=1e-2)


# ---------------- GEMS blend path ----------------

@pytest.fixture(scope="module")
def blend_run():
    raw = _blend_raw(schedule={"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                               "dt_min_h": 0.001})
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    state, summary = eng.run()
    return cfg, state, summary


@needs_gems
def test_blend_backend_react_with_fa_release(tmp_path):
    from tinn.gems import GemsBackend, GemsWorker
    worker = GemsWorker(str(BUNDLE), python_executable=str(GEMS_PYTHON),
                        work_root=str(tmp_path))
    b = GemsBackend(worker, 293.15)
    rel = {"C3S": 2e-12, "fly_ash": 1e-12}
    r = b.react(rel, 6e-10, np.zeros(len(ELEMENT_IDS)))
    assert r.status == "ok"
    solids = {pc.phase_id for pc in r.parcels if pc.mol > 1e-16}
    assert "CSHQ" in solids
    # FA's alkalis are in play: element closure must include Na/K rows
    total = sum((pc.elements for pc in r.parcels), np.zeros(len(ELEMENT_IDS)))
    total = total + r.residual_inventory
    na = ELEMENT_IDS.index("Na")
    assert total[na] > 0.0


@needs_gems
def test_blend_3d_run_closure_and_fa_alpha(blend_run):
    _, state, summary = blend_run
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    fa_i = KINETIC_PHASE_IDS.index("fly_ash")
    assert state.alpha()[fa_i] > 0.0            # fly ash actually reacted
    assert state.initial_phase_mol[fa_i] > 0.0
    # K/Na from FA glass are tracked through the whole ledger
    assert state.initial_elements[ELEMENT_IDS.index("K")] > 0.0
    w_err = abs(state.water_free_mol + state.water_gel_mol + state.water_bound_mol
                - state.initial_water_mol)
    assert w_err <= 1e-8 * state.initial_water_mol + 1e-24


@needs_gems
def test_blend_3d_sanity_band_uses_clinker_only(blend_run):
    cfg, _, summary = blend_run
    for c in summary["sanity_band"]["checks"]:
        # no alpha-band rows at 2/6 h, and C3S>=C2S must still be evaluated
        assert c["status"] in ("pass", "warn")
    names = {c["check"] for c in summary["sanity_band"]["checks"]}
    assert "alpha_order_C3S_ge_C2S" in names

# ---------------- per-material populations (PRD v2.2, W1) ----------------

# (dense_hash, placement_hash) per legacy example. dense_hash covers the
# per-channel anhydrous array, so it MOVES whenever the solid channel list
# grows — E3 appended five all-zero salt-carrier channels (pre-E3 values were
# 689171a7... / a9ce52e1...). placement_hash covers what must never move:
# the occupancy FIELD (channel sum), water, gas and particle ids — i.e. the
# RNG draws and rasterization. A physics regression breaks both; a channel
# list extension breaks only the first.
LEGACY_HASHES = {
    "c3s_32.json": (
        "941c09fdaa49112831e925128073edc726da5ea53c847f83bd28dc091c910570",
        "83cffe827c2e56dc1de5e5ebac12ede16ed555dcc36725e2e42d11455fb5ed01"),
    "opc_srm114q_32.json": (
        "83471efa81c1abe91cedf2e7acca9823fc3fd39f14bf59e4495104d1b819adb4",
        "478acbe54441d3499d1dd6683174e49bdc5fea0b2755c5f15e4dc369ae424311"),
}

SLAG_PSD = {"bins": [
    {"d_lo_um": 0.5, "d_hi_um": 1.0, "volume_fraction": 0.10},
    {"d_lo_um": 1.0, "d_hi_um": 2.0, "volume_fraction": 0.25},
    {"d_lo_um": 2.0, "d_hi_um": 4.0, "volume_fraction": 0.35},
    {"d_lo_um": 4.0, "d_hi_um": 8.0, "volume_fraction": 0.30}]}


def test_material_psd_schema_validation():
    raw = _blend_raw()
    raw["material_psd"] = {"bogus": SLAG_PSD}
    with pytest.raises(ValidationError, match="material_psd"):
        TinnConfig.model_validate(raw)
    raw = _blend_raw()
    raw["material_psd"] = {"slag": SLAG_PSD}  # recipe has fly_ash, not slag
    with pytest.raises(ValidationError, match="no slag mass"):
        TinnConfig.model_validate(raw)
    raw = _blend_raw()
    big = {"bins": [{"d_lo_um": 20.0, "d_hi_um": 31.0, "volume_fraction": 1.0}]}
    raw["material_psd"] = {"fly_ash": big}
    with pytest.raises(ValidationError, match="rasterizable"):
        TinnConfig.model_validate(raw)


def test_material_shape_schema_validation():
    """material_shape (PRD 1.2 rev.2): same key discipline + fit checks."""
    raw = _blend_raw()
    raw["material_shape"] = {"bogus": {"aspects": [1.5, 1.0, 0.7]}}
    with pytest.raises(ValidationError, match="material_shape"):
        TinnConfig.model_validate(raw)
    raw = _blend_raw()
    raw["material_shape"] = {"clinker": {"aspects": [1.5, 1.0, -0.7]}}
    with pytest.raises(ValidationError, match="positive"):
        TinnConfig.model_validate(raw)
    raw = _blend_raw()
    # elongation stretches the largest grain past the periodic RVE
    raw["material_shape"] = {"clinker": {"aspects": [40.0, 1.0, 1.0]}}
    with pytest.raises(ValidationError, match="major axis"):
        TinnConfig.model_validate(raw)


def test_config_hash_stable_without_material_psd():
    raw = _blend_raw()
    h0 = TinnConfig.model_validate(raw).config_hash()
    raw2 = dict(raw)
    raw2["material_psd"] = None
    assert TinnConfig.model_validate(raw2).config_hash() == h0
    raw3 = dict(raw)
    raw3["material_psd"] = {"fly_ash": SLAG_PSD}
    assert TinnConfig.model_validate(raw3).config_hash() != h0


def test_config_hash_stable_without_material_shape():
    """material_shape follows the same None-pops-from-hash contract (rev.2)."""
    raw = _blend_raw()
    h0 = TinnConfig.model_validate(raw).config_hash()
    raw4 = dict(raw)
    raw4["material_shape"] = None
    assert TinnConfig.model_validate(raw4).config_hash() == h0
    raw5 = dict(raw)
    raw5["material_shape"] = {"clinker": {"aspects": [1.5, 1.0, 0.7]}}
    assert TinnConfig.model_validate(raw5).config_hash() != h0


def test_geometry_legacy_bitwise_unchanged():
    import hashlib
    from tinn.geometry import initialize_rve
    for name, (dense_expect, place_expect) in LEGACY_HASHES.items():
        cfg = TinnConfig.from_json_file(str(EXAMPLES / name))
        rve = initialize_rve(cfg, REG)
        assert rve.dense_hash() == dense_expect, name
        h = hashlib.sha256()
        for arr in (rve.anhydrous_fraction.sum(axis=0), rve.capillary_liquid,
                    rve.capillary_gas, rve.particle_id):
            h.update(np.ascontiguousarray(arr).tobytes())
        assert h.hexdigest() == place_expect, f"{name} placement moved"
        # the E3 salt channels exist but stay empty for a salt-free recipe
        salt_idx = [SOLID_PHASE_IDS.index(p) for p in SALT_PHASE_IDS]
        assert not np.any(rve.anhydrous_fraction[salt_idx])


def _blend_rve(material_psd=None):
    from tinn.geometry import initialize_rve
    raw = _blend_raw()
    if material_psd is not None:
        raw["material_psd"] = material_psd
    return initialize_rve(TinnConfig.model_validate(raw), REG)


def test_geometry_per_material_targets_and_purity():
    rve = _blend_rve({"fly_ash": SLAG_PSD})
    mats = rve.report["materials"]
    assert set(mats) == {"clinker", "fly_ash"}
    for name, row in mats.items():
        assert row["rel_error"] <= 0.02, (name, row)
    # SCM particles project ONLY onto their own channel: fly_ash dense volume
    # equals the fly_ash material's achieved volume exactly
    fa_dense = float(rve.anhydrous_fraction[rve.phase_ids.index("fly_ash")].sum())
    assert fa_dense == pytest.approx(mats["fly_ash"]["volume_achieved_vox"], rel=1e-12)
    # material column present and both populations placed
    assert set(np.unique(rve.particles["material"])) == {0, 1}
    # finer fly-ash PSD -> smaller mean FA particle diameter than clinker
    m = rve.particles["material"]
    assert (rve.particles["diameter_um"][m == 1].mean()
            < rve.particles["diameter_um"][m == 0].mean())


def test_geometry_all_scm_binder_initializes():
    # {"slag": 1.0} has no clinker population at all — must not divide by zero
    from tinn.geometry import initialize_rve
    raw = _blend_raw()
    raw["binder"] = {"mass_fractions": {"slag": 1.0}}
    rve = initialize_rve(TinnConfig.model_validate(raw), REG)
    assert set(rve.report["materials"]) == {"slag"}
    total = rve.anhydrous_fraction.sum(axis=0) + rve.capillary_liquid + rve.capillary_gas
    assert np.max(np.abs(total - 1.0)) <= 1e-12


def test_geometry_per_material_determinism_and_conservation():
    a = _blend_rve({"fly_ash": SLAG_PSD})
    b = _blend_rve({"fly_ash": SLAG_PSD})
    assert a.dense_hash() == b.dense_hash()
    # voxel identity with two populations
    total = a.anhydrous_fraction.sum(axis=0) + a.capillary_liquid + a.capillary_gas
    assert np.max(np.abs(total - 1.0)) <= 1e-12


def test_geometry_ellipsoid_shapes():
    """Ellipsoid rasterizer (PRD 1.2 rev.2): volume-exact bookkeeping, the
    requested anisotropy (axis-aligned reference), and a shaped blend RVE that
    is deterministic, conserving, and structurally distinct from spheres."""
    a = _blend_rve({"fly_ash": SLAG_PSD})
    from tinn.geometry import _rasterize_ellipsoid, _sphere_volume
    rv = 4.0
    ratios = np.array([2.0, 1.0, 0.5])   # product 1 -> volume preserved
    widx, frac, deficit = _rasterize_ellipsoid(
        np.array([16.0, 16.0, 16.0]), rv, ratios, np.eye(3), 32)
    assert float(frac.sum()) + deficit == pytest.approx(_sphere_volume(rv), rel=1e-9)
    zz, yy, xx = np.unravel_index(widx, (32, 32, 32))
    assert (zz.max() - zz.min() + 1) >= 2 * (xx.max() - xx.min() + 1) - 2
    # shaped blend RVE: deterministic, conserving, and clinker grains are
    # actually elongated while spherical fly ash stays untouched
    raw = _blend_raw()
    raw["material_shape"] = {"clinker": {"aspects": [1.6, 1.0, 0.7]}}
    from tinn.geometry import initialize_rve
    s1 = initialize_rve(TinnConfig.model_validate(raw), REG)
    s2 = initialize_rve(TinnConfig.model_validate(raw), REG)
    assert s1.dense_hash() == s2.dense_hash()
    total = s1.anhydrous_fraction.sum(axis=0) + s1.capillary_liquid + s1.capillary_gas
    assert np.max(np.abs(total - 1.0)) <= 1e-12
    for row in s1.report["materials"].values():
        assert row["rel_error"] <= 0.02, row
    assert s1.dense_hash() != a.dense_hash()  # shape actually changes structure
