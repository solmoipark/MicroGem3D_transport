"""M2 tests: Parrot--Killoh kinetics (2 presets) + 4-phase 3D run (PRD §3 M2, §6.2)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tinn import ledger
from tinn.config import PK_PRESETS as PK_PRESET_NAMES
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.kinetics import (PK_PRESETS, ParrotKilloh, REFERENCE_BLAINE_M2_KG,
                           make_kinetics)
from tinn.registry import HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS, default_registry

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SRM114Q = {"C3S": 0.60, "C2S": 0.14, "C3A": 0.07, "C4AF": 0.10}


def _pk(preset="pk_elakneswaran_2018", **kw):
    args = dict(preset_name=preset, w_c=0.5, temperature_k=293.15,
                blaine_m2_kg=381.8, phase_mass_fractions=SRM114Q)
    args.update(kw)
    return ParrotKilloh(**args)


# ---------------- published parameter tables (PRD §6.2 anchors) ----------------

def test_preset_tables_match_published_values():
    e = PK_PRESETS["pk_elakneswaran_2018"]
    assert e.params[0] == (1.5, 0.70, 0.050, 1.10, 3.3, 1.80, 41570.0)   # C3S
    assert e.params[1] == (0.5, 1.00, 0.006, 0.20, 5.0, 1.35, 20785.0)   # C2S
    assert e.params[2] == (1.0, 0.85, 0.040, 1.00, 3.2, 1.60, 54040.0)   # C3A
    assert e.params[3] == (0.37, 0.70, 0.015, 0.40, 3.7, 1.45, 34087.0)  # C4AF
    assert e.reference_temperature_k == 293.15 and e.surface_scales_all_rates

    c = PK_PRESETS["pk_cemgems_2021"]
    assert c.params[0] == (1.5, 0.70, 0.050, 1.10, 3.3, 2.00, 41570.0)
    assert c.params[1] == (0.5, 1.00, 0.020, 0.70, 5.0, 1.55, 20785.0)
    assert c.params[2] == (1.0, 0.85, 0.040, 1.00, 3.2, 1.80, 54040.0)
    assert c.params[3] == (0.37, 0.70, 0.020, 0.40, 3.7, 1.65, 34087.0)
    assert c.reference_temperature_k == 298.15 and not c.surface_scales_all_rates
    # config's allowed preset names stay in sync with the kinetics registry
    assert set(PK_PRESET_NAMES) == set(PK_PRESETS)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="unknown P&K preset"):
        _pk(preset="pk_bogus")


def test_jander_diffusion_rate_anchor():
    # analytic anchor at T=T0, Blaine=385, RH=1: R_df = K2 (1-a)^(2/3) / (1-(1-a)^(1/3))
    pk = _pk(blaine_m2_kg=REFERENCE_BLAINE_M2_KG)
    for alpha, expected in ((0.1, 1.3505550989656108), (0.5, 0.15268107879394868)):
        _, r_df, _ = pk.rate_components(np.full(4, alpha))
        assert r_df[0] == pytest.approx(expected, rel=1e-12)


def test_surface_scaling_policy_differs_between_presets():
    a = np.full(4, 0.3)
    base_e = _pk(blaine_m2_kg=REFERENCE_BLAINE_M2_KG).controlling_rates(a)
    dbl_e = _pk(blaine_m2_kg=2 * REFERENCE_BLAINE_M2_KG).controlling_rates(a)
    # 2018: all rates scale with the Blaine ratio, so the controlling min doubles
    assert dbl_e == pytest.approx(2.0 * base_e, rel=1e-12)
    # 2021: only nucleation/growth scales; where diffusion/shell controls,
    # doubling Blaine must NOT double the controlling rate
    base_c = _pk("pk_cemgems_2021", blaine_m2_kg=REFERENCE_BLAINE_M2_KG).controlling_rates(a)
    dbl_c = _pk("pk_cemgems_2021", blaine_m2_kg=2 * REFERENCE_BLAINE_M2_KG).controlling_rates(a)
    assert np.any(dbl_c < 2.0 * base_c - 1e-12)


def test_arrhenius_factors():
    pk = _pk(temperature_k=303.15)
    expected = np.exp(np.array([41570.0, 20785.0, 54040.0, 34087.0]) / 8.314
                      * (1.0 / 293.15 - 1.0 / 303.15))
    assert pk.temperature_factors() == pytest.approx(expected, rel=1e-12)


def test_rh_cutoff_stops_hydration():
    assert _pk(relative_humidity=0.55).humidity_factor() == 0.0
    assert _pk(relative_humidity=0.50).humidity_factor() == 0.0
    assert _pk(relative_humidity=1.0).humidity_factor() == 1.0
    dry = _pk(relative_humidity=0.5)
    assert np.all(dry.controlling_rates(np.full(4, 0.2)) == 0.0)


def test_water_retardation_clips_and_cannot_rebound():
    pk = _pk(w_c=0.2)  # thresholds H*w/c in [0.27, 0.36]
    f_low = pk.water_retardation_factors(0.1)
    assert np.all(f_low == 1.0)
    f_hi = pk.water_retardation_factors(0.9)
    assert np.all(f_hi >= 0.0) and np.all(f_hi < 1.0)
    # far beyond the bracket root the factor is exactly 0, never positive again
    assert np.all(pk.water_retardation_factors(1.0) == 0.0)


def test_total_clinker_alpha_normalizes_to_pk_phases():
    pk = _pk()
    full = np.zeros(8); full[:4] = 1.0
    total = pk.total_clinker_alpha(full)
    assert total == pytest.approx(1.0)  # divided by 0.91, not by 1.0
    full = np.zeros(8); full[0] = 0.5
    total = pk.total_clinker_alpha(full)
    assert total == pytest.approx(0.5 * 0.60 / 0.91)


def test_trajectory_monotone_bounded_and_ordered():
    for preset in PK_PRESETS:
        pk = _pk(preset)
        times = [6.0, 24.0, 72.0, 168.0]
        prev = np.zeros(len(_pk().alpha_at(0.0)))
        for t in times:
            a = pk.alpha_at(t)
            assert np.all(a >= prev - 1e-15) and np.all(a <= 1.0)
            assert a[0] >= a[1]  # C3S alpha >= C2S alpha at all t (PRD §6.3)
            prev = a


def test_alpha_at_is_pure_function_of_time():
    pk = _pk()
    _ = pk.alpha_at(7.0)
    after = pk.alpha_at(24.0)
    fresh = _pk().alpha_at(24.0)
    assert np.array_equal(after, fresh)
    assert np.all(pk.alpha_at(0.0) == 0.0)


def test_alpha_24h_regression_snapshot():
    expect = {
        "pk_elakneswaran_2018": [0.41207163630249377, 0.13621110905159375,
                                 0.3893882681136848, 0.19067470813260054],
        "pk_cemgems_2021": [0.3594571813942425, 0.2617245655269392,
                            0.32338514256745915, 0.15004453532690035],
    }
    for preset, vals in expect.items():
        assert _pk(preset).alpha_at(24.0)[:4] == pytest.approx(vals, rel=1e-12)


def test_srm114q_1d_within_plausible_band():
    for preset in PK_PRESETS:
        pk = _pk(preset)
        total = pk.total_clinker_alpha(pk.alpha_at(24.0))
        assert 0.20 <= total <= 0.60  # PRD §6.3 band 0.25-0.55 with slack


# ---------------- config / factory ----------------

def test_config_pk_requires_blaine():
    raw = json.loads((EXAMPLES / "opc_srm114q_32.json").read_text(encoding="utf-8"))
    del raw["kinetics"]["blaine_m2_kg"]
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw)
    raw2 = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw2["kinetics"]["blaine_m2_kg"] = 381.8
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw2)


def test_make_kinetics_builds_parrot_killoh():
    cfg = TinnConfig.from_json_file(str(EXAMPLES / "opc_srm114q_32.json"))
    kin = make_kinetics(cfg)
    assert isinstance(kin, ParrotKilloh)
    assert kin.preset.name == "pk_elakneswaran_2018"
    assert kin.blaine_ratio == pytest.approx(381.8 / 385.0)


# ---------------- 4-phase 3D run (M2 DoD) ----------------

@pytest.fixture(scope="module")
def opc_run():
    raw = json.loads((EXAMPLES / "opc_srm114q_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [6.0, 24.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    state, summary = eng.run()
    return cfg, state, summary


def test_opc_run_per_phase_ledger_closure(opc_run):
    _, state, _ = opc_run
    rep = ledger.check_all(state, default_registry())
    assert rep.ok, rep.violations
    # every phase with initial mass actually dissolved (SCM channels are empty here)
    dissolved = state.initial_phase_mol - state.phase_mol
    active = state.initial_phase_mol > 0.0
    assert active.sum() == 4 and np.all(dissolved[active] > 0.0)
    assert rep.metrics["dense_ledger_max_rel_err"] <= 1e-9


def test_opc_run_produces_all_hydrates(opc_run):
    _, state, _ = opc_run
    idx = {h: i for i, h in enumerate(HYDRATE_PHASE_IDS)}
    for h in ("CSH", "CH", "C3AH6", "FH3"):
        assert state.hydrate_mol[idx[h]] > 0.0
        assert state.hydrate_fraction[idx[h]].sum() > 0.0


def test_opc_run_alpha_ordering_and_reporting(opc_run):
    _, state, summary = opc_run
    a = state.alpha()
    assert a[KINETIC_PHASE_IDS.index("C3S")] >= a[KINETIC_PHASE_IDS.index("C2S")]
    row = summary["outputs"][-1]
    assert row["alpha"]["C3S"] > 0.2
    # unmet is reported per phase, never hidden
    assert set(row["unmet_mol"]) == set(KINETIC_PHASE_IDS)
