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
                           KINETIC_PHASE_IDS, RegistryError, SCM_PHASE_IDS,
                           default_registry, scm_entry)

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
    assert KINETIC_PHASE_IDS == CLINKER_PHASE_IDS + SCM_PHASE_IDS
    assert SCM_PHASE_IDS == ("slag", "fly_ash", "metakaolin", "silica_fume")


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
    # volume ratio = (mass/density) ratio: FA is much lighter than C3S
    expect = (0.30 / 2.20) / (0.46353 / REG.get("C3S").density_g_cm3)
    assert vols["fly_ash"] / vols["C3S"] == pytest.approx(expect, rel=1e-6)


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
