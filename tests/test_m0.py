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

def test_absent_transport_section_keeps_legacy_hash():
    """v4.0/RT: "transport": null, {}, and all-None fields hash exactly like
    the section's absence — a config without RT modes is unchanged physics."""
    base = TinnConfig.model_validate(_base_config()).config_hash()
    for form in (None, {}, {"exchange_tau_h": None},
                 {"exchange_tau_h": None, "exchange_tau_h_per_phase": None}):
        assert TinnConfig.model_validate(
            _base_config(transport=form)).config_hash() == base
    # an ACTIVE tau must change the hash (gems3k config: tau needs it)
    g = json.loads((EXAMPLES / "opc_cnash_32.json").read_text(encoding="utf-8"))
    g_base = TinnConfig.model_validate(g).config_hash()
    g["transport"] = {"exchange_tau_h": 720.0}
    assert TinnConfig.model_validate(g).config_hash() != g_base


def test_transport_config_validation():
    """v4.0/RT: tau must be positive, per-phase values non-negative and
    non-empty, unknown keys refused, and the stoichiometric backend refused
    (rate limiting its nonexistent re-equilibration would be a silent no-op)."""
    g = json.loads((EXAMPLES / "opc_cnash_32.json").read_text(encoding="utf-8"))
    for bad in ({"exchange_tau_h": 0.0},
                {"exchange_tau_h": -5.0},
                {"exchange_tau_h": float("inf")},
                {"exchange_tau_h": True},
                {"exchange_tau_h_per_phase": {}},
                {"exchange_tau_h_per_phase": {"CSHQ": -1.0}},
                {"exchange_tau_h_per_phase": {"CSHQ": float("nan")}},
                {"exchange_tau_h_per_phase": {"CSHQ": True}},
                # only-zero overrides with no global tau declare rate
                # limiting but limit nothing - refused, not silently inert
                {"exchange_tau_h_per_phase": {"CSHQ": 0.0}},
                {"exchange_tau_h": 10.0, "unknown_knob": 1}):
        with pytest.raises(ValidationError):
            TinnConfig.model_validate({**g, "transport": bad})
    with pytest.raises(ValidationError, match="silent no-op"):
        TinnConfig.model_validate(
            _base_config(transport={"exchange_tau_h": 10.0}))


def test_domain_partition_config_validation_and_hash():
    """v4.0/RT mode C knobs: tile must divide the grid, d0 is required and
    finite; an absent/None domains key keeps the legacy hash."""
    base = TinnConfig.model_validate(_base_config()).config_hash()
    assert TinnConfig.model_validate(
        _base_config(transport={"domains": None})).config_hash() == base
    ok = TinnConfig.model_validate(_base_config(
        transport={"domains": {"tile_vox": 8, "d0_m2_s": 1e-9}}))
    assert ok.config_hash() != base
    for bad in ({"tile_vox": 7, "d0_m2_s": 1e-9},      # does not divide 32
                {"tile_vox": 8},                        # d0 required
                {"tile_vox": 8, "d0_m2_s": 0.0},
                {"tile_vox": 8, "d0_m2_s": float("inf")},
                {"tile_vox": 8, "d0_m2_s": 1e-9, "dirty_rtol": -1.0},
                {"tile_vox": 8, "d0_m2_s": 1e-9, "unknown": 1}):
        with pytest.raises(ValidationError):
            TinnConfig.model_validate(_base_config(
                transport={"domains": bad}))


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
        assert f == pytest.approx(expect.get(p, 0.0), abs=1e-9)  # SCM channels: 0
    assert rve.particle_id.max() >= 0
    assert rve.particles["placed"].any()


# ---------- cli ----------

def test_cli_validate_config(capsys):
    assert cli.main(["validate-config", str(EXAMPLES / "c3s_32.json")]) == 0
    assert "config OK" in capsys.readouterr().out
    assert cli.main(["validate-config", str(EXAMPLES / "does_not_exist.json")]) == 2


# ---------- measured PSD input (PRD 1.2 rev.2) ----------

def test_psd_cumulative_conversion_and_window():
    # full-window curve: consecutive points become bins, exact fractions
    psd = PSD.model_validate({"cumulative": [[1.0, 0.0], [2.0, 0.25], [4.0, 1.0]]})
    assert [(b.d_lo_um, b.d_hi_um) for b in psd.bins] == [(1.0, 2.0), (2.0, 4.0)]
    assert psd.bins[0].volume_fraction == pytest.approx(0.25)
    assert psd.bins[1].volume_fraction == pytest.approx(0.75)
    assert psd.measured_window_excluded() == 0.0
    # real lab curve (SRM 114q style): window renormalized, exclusion reported
    psd2 = PSD.model_validate({"cumulative": [[1.0, 0.05], [4.0, 0.55], [8.0, 0.95]]})
    assert psd2.measured_window_excluded() == pytest.approx(0.10)
    assert sum(b.volume_fraction for b in psd2.bins) == pytest.approx(1.0)
    assert psd2.bins[0].volume_fraction == pytest.approx(0.5 / 0.9)


def test_psd_cumulative_rejections():
    for bad in ([[1.0, 0.5]],                    # fewer than 2 points
                [[2.0, 0.0], [1.0, 1.0]],        # descending diameters
                [[1.0, 0.5], [2.0, 0.4]],        # decreasing fractions
                [[-1.0, 0.0], [2.0, 1.0]],       # non-positive diameter
                [[1.0, 0.3], [2.0, 0.3]]):       # zero-mass window
        with pytest.raises(ValidationError):
            PSD.model_validate({"cumulative": bad})


def test_psd_rosin_rammler_discretization():
    import math
    rr = {"d_prime_um": 15.0, "n": 1.1, "d_min_um": 0.5, "d_max_um": 40.0}
    psd = PSD.model_validate({"rosin_rammler": rr})
    assert psd.bins[0].d_lo_um == 0.5 and psd.bins[-1].d_hi_um == 40.0
    assert sum(b.volume_fraction for b in psd.bins) == pytest.approx(1.0)
    cdf = lambda d: 1.0 - math.exp(-((d / 15.0) ** 1.1))
    window = cdf(40.0) - cdf(0.5)
    assert psd.measured_window_excluded() == pytest.approx(1.0 - window)
    # each bin carries exactly its window-normalized CDF mass
    b = psd.bins[3]
    assert b.volume_fraction == pytest.approx(
        (cdf(b.d_hi_um) - cdf(b.d_lo_um)) / window, rel=1e-12)


def test_psd_input_form_rules():
    with pytest.raises(ValidationError, match="not both"):
        PSD.model_validate({"cumulative": [[1.0, 0.0], [2.0, 1.0]],
                            "rosin_rammler": {"d_prime_um": 15.0, "n": 1.1,
                                              "d_min_um": 0.5, "d_max_um": 40.0}})
    with pytest.raises(ValidationError, match="needs one of"):
        PSD.model_validate({})
    # bins are the DERIVED canonical form: with a measured input present they
    # are rebuilt from it (what makes serialized configs revalidate cleanly)
    stale = [{"d_lo_um": 1.0, "d_hi_um": 2.0, "volume_fraction": 1.0}]
    psd = PSD.model_validate({"bins": stale,
                              "cumulative": [[1.0, 0.0], [2.0, 0.5], [4.0, 1.0]]})
    assert len(psd.bins) == 2 and psd.bins[1].d_hi_um == 4.0


def test_psd_truncation_to_grid():
    coarse = {"cumulative": [[1.0, 0.0], [8.0, 0.5], [64.0, 1.0]]}
    raw = _base_config(psd=dict(coarse))
    with pytest.raises(ValidationError, match="rasterizable"):
        TinnConfig.model_validate(raw)   # 64 um cannot fit a 32^3 RVE
    raw = _base_config(psd={**coarse, "truncate_to_grid": True})
    cfg = TinnConfig.model_validate(raw)
    allowed = (32 - 2.0 - 3.0 ** 0.5) * 1.0
    assert cfg.psd.d_max_um <= allowed
    assert cfg._psd_truncation["__shared__"] == pytest.approx(0.5)
    assert sum(b.volume_fraction for b in cfg.psd.bins) == pytest.approx(1.0)
    # idempotent: revalidating the truncated dump reproduces the same hash
    again = TinnConfig.model_validate(
        json.loads(json.dumps(cfg.model_dump(mode="json"))))
    assert again.config_hash() == cfg.config_hash()


def test_psd_hash_contract_for_new_fields():
    raw = _base_config()
    h0 = TinnConfig.model_validate(raw).config_hash()
    # explicit default-valued new fields hash identically to a legacy config
    raw2 = _base_config()
    raw2["psd"] = {"bins": [b.model_dump() for b in PSD.synthetic_default().bins],
                   "truncate_to_grid": False, "cumulative": None,
                   "rosin_rammler": None}
    assert TinnConfig.model_validate(raw2).config_hash() == h0
    # a measured input is a DIFFERENT physics input -> different hash
    raw3 = _base_config()
    raw3["psd"] = {"cumulative": [[0.5, 0.0], [8.0, 0.7], [16.0, 1.0]]}
    assert TinnConfig.model_validate(raw3).config_hash() != h0


def test_geometry_measured_psd_example_initializes():
    cfg = TinnConfig.from_json_file(str(EXAMPLES / "opc_srm114q_measured_psd_64.json"))
    rve = initialize_rve(cfg, default_registry())
    row = rve.report["materials"]["clinker"]
    assert row["rel_error"] <= 0.02
    # NIST SP 260-166 Table 8 exclusions, both surfaced
    assert row["psd_window_excluded"] == pytest.approx(0.052, abs=1e-12)
    assert 0.01 < row["psd_truncated_volume_fraction"] < 0.02
    # sphere-based SSA sits in a physical range and BELOW the measured Blaine
    # (381.8 m2/kg) since the sub-um fines lie outside the measured window
    assert 150.0 < row["ssa_est_m2_kg"] < 400.0
