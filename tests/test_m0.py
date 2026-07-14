"""M0 tests: config schema, registry, RVE initialization (PRD §3 M0, ~15 tests)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tinn import cli
from tinn.config import PSD, TinnConfig
from tinn.geometry import initialize_rve
from tinn.registry import PhaseEntry, Registry, RegistryError, default_registry

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _base_config(**overrides) -> dict:
    cfg = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    cfg.update(overrides)
    return cfg


def _opc_config(**overrides) -> dict:
    cfg = json.loads((EXAMPLES / "opc_srm114q_32.json").read_text(encoding="utf-8"))
    cfg.update(overrides)
    return cfg


# ---------- config ----------

def test_config_valid_and_hash_stable():
    a = TinnConfig.model_validate(_base_config())
    b = TinnConfig.from_json_file(str(EXAMPLES / "c3s_32.json"))
    assert a.config_hash() == b.config_hash()
    assert len(a.config_hash()) == 64
    # OPC example with pk preset also validates
    assert TinnConfig.model_validate(_opc_config()).binder.unassigned == pytest.approx(0.09)


def test_config_rejects_bad_wc_and_temperature():
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(w_c=-0.1))
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(temperature_K=100.0))


def test_config_rejects_bad_recipe():
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(binder={"mass_fractions": {"C3S": 0.8, "C2S": 0.5}}))
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(binder={"mass_fractions": {"MgO": 1.0}}))


def test_config_rejects_bad_psd():
    bad = {"bins": [{"d_lo_um": 1.0, "d_hi_um": 2.0, "volume_fraction": 0.5}]}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(psd=bad))


def test_config_rejects_unknown_pk_preset():
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_opc_config(kinetics={"kind": "pk", "preset": "pk_bogus"}))


def test_config_rejects_bad_rve():
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(rve={"grid_size": 48, "voxel_size_um": 1.0, "seed": 1}))
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(rve={"grid_size": 32, "voxel_size_um": 0.4, "seed": 1}))


def test_config_rejects_psd_larger_than_rve():
    # 16 um particles do not fit a 32 * 0.5 um = 16 um periodic box
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(rve={"grid_size": 32, "voxel_size_um": 0.5, "seed": 1}))
    # near-extent diameters that the rasterizer halo cannot fit are rejected too
    bad_psd = {"bins": [{"d_lo_um": 29.0, "d_hi_um": 31.0, "volume_fraction": 1.0}]}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(psd=bad_psd))


def test_config_cross_validation():
    # tabulated table must cover every reacting binder phase
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(
            _base_config(binder={"mass_fractions": {"C3S": 0.5, "C2S": 0.5}}))
    # output times must not exceed the tabulated horizon (168 h in the fixture)
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(schedule={"output_times_h": [6.0, 500.0]}))
    # stoichiometric rules must be element-balanced
    bad_rule = {"backend": "stoichiometric",
                "stoichiometric_rules": {"C3S": {"water_mol": 2.0, "products": {"CH": 3.0}}}}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(chemistry=bad_rule))
    # stoichiometric backend needs a rule for every reacting phase
    partial = {"backend": "stoichiometric",
               "stoichiometric_rules": {"C2S": {"water_mol": 4.3,
                                                "products": {"CSH": 1.0, "CH": 0.3}}}}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(chemistry=partial))
    # gems3k takes no rules, and unused rules never enter its config_hash
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(_base_config(
            chemistry={"backend": "gems3k",
                       "stoichiometric_rules": {"C3S": {"water_mol": 5.3,
                                                        "products": {"CSH": 1.0, "CH": 1.3}}}}))
    g = TinnConfig.model_validate(_base_config(
        chemistry={"backend": "gems3k", "gems_bundle_lst": "gems_bundles/PC/PC-dat.lst"}))
    assert g.chemistry.stoichiometric_rules is None


# ---------- registry ----------

def test_registry_molar_data():
    reg = default_registry()
    c3s = reg.get("C3S")
    assert c3s.molar_mass_g_mol == pytest.approx(228.31, abs=0.05)
    assert c3s.density_g_cm3 == pytest.approx(3.15, abs=0.05)
    assert reg.get("CSH").gel_porosity == pytest.approx(0.28)


def test_registry_missing_basis_is_hard_error():
    with pytest.raises(RegistryError):
        Registry((PhaseEntry("X", "hydrate", {"Ca": 1, "O": 1}, 20.0, basis=None),))


def test_registry_unknown_phase():
    with pytest.raises(KeyError):
        default_registry().get("mullite")


# ---------- geometry ----------

@pytest.fixture(scope="module")
def c3s_rve():
    cfg = TinnConfig.model_validate(_base_config())
    return cfg, initialize_rve(cfg, default_registry())


def test_geometry_deterministic_same_seed(c3s_rve):
    cfg, rve = c3s_rve
    again = initialize_rve(cfg, default_registry())
    assert rve.dense_hash() == again.dense_hash()


def test_geometry_different_seed_differs(c3s_rve):
    cfg, rve = c3s_rve
    other = TinnConfig.model_validate(
        _base_config(rve={"grid_size": 32, "voxel_size_um": 1.0, "seed": 43}))
    assert initialize_rve(other, default_registry()).dense_hash() != rve.dense_hash()


def test_geometry_voxel_identity(c3s_rve):
    _, rve = c3s_rve
    total = rve.anhydrous_fraction.sum(axis=0) + rve.capillary_liquid + rve.capillary_gas
    assert np.max(np.abs(total - 1.0)) <= 1e-12
    assert rve.anhydrous_fraction.min() >= 0.0
    assert rve.capillary_liquid.min() >= 0.0
    assert rve.capillary_gas.min() >= 0.0


def test_geometry_solid_fraction_and_wc_achieved(c3s_rve):
    _, rve = c3s_rve
    r = rve.report
    assert r["solid_fraction_rel_error"] <= 0.02
    assert r["w_c_rel_error"] <= 0.02
    # errors are reported, not hidden
    for key in ("sampling_residual_volume_vox", "unplaced_volume_vox",
                "raster_deficit_volume_vox"):
        assert key in r


def test_geometry_c3s_only_has_single_phase(c3s_rve):
    _, rve = c3s_rve
    idx = {p: i for i, p in enumerate(rve.phase_ids)}
    assert rve.anhydrous_fraction[idx["C3S"]].sum() > 0.0
    for p in ("C2S", "C3A", "C4AF", "inert"):
        assert rve.anhydrous_fraction[idx[p]].sum() == 0.0


def test_geometry_opc_multiphase_matches_recipe():
    cfg = TinnConfig.model_validate(_opc_config())
    rve = initialize_rve(cfg, default_registry())
    reg = default_registry()
    vol = rve.anhydrous_fraction.sum(axis=(1, 2, 3))
    mass = vol * np.array([reg.get(p).density_g_cm3 for p in rve.phase_ids])
    frac = mass / mass.sum()
    expect = {"C3S": 0.60, "C2S": 0.14, "C3A": 0.07, "C4AF": 0.10, "inert": 0.09}
    for p, f in zip(rve.phase_ids, frac):
        assert f == pytest.approx(expect[p], abs=1e-9)
    assert rve.particle_id.max() >= 0
    assert rve.particles["placed"].any()


# ---------- cli ----------

def test_cli_validate_config(capsys):
    assert cli.main(["validate-config", str(EXAMPLES / "c3s_32.json")]) == 0
    assert "config OK" in capsys.readouterr().out
    assert cli.main(["validate-config", str(EXAMPLES / "does_not_exist.json")]) == 2
