"""M4 tests: GEMS->3D coupling — GemsBackend parcels through the engine
(PRD §3 M4: cluster release -> xGEMS -> parcel commit -> placement; §6.1 closure).

Worker-dependent tests skip without the xgems interpreter/bundle; config tests
run everywhere.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from tinn import ledger
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.registry import (ELEMENT_IDS, KINETIC_PHASE_IDS, default_registry,
                           element_vector)
from tinn.storage import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))

needs_gems = pytest.mark.skipif(
    not (BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")

REG = default_registry()


def _gems_cfg(**overrides) -> TinnConfig:
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw.update(overrides)
    return TinnConfig.model_validate(raw)


@pytest.fixture(scope="module")
def gems_backend(tmp_path_factory):
    from tinn.gems import GemsBackend, GemsWorker
    worker = GemsWorker(str(BUNDLE), python_executable=str(GEMS_PYTHON),
                        work_root=str(tmp_path_factory.mktemp("gems_m4")))
    return GemsBackend(worker, 293.15)


@pytest.fixture(scope="module")
def gems_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("gems_run") / "run"
    cfg = _gems_cfg(schedule={"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                              "dt_min_h": 0.001})
    eng = Engine(cfg)
    state, summary = eng.run(out_dir=str(out))
    return cfg, eng, state, summary, out


# ---------------- GemsBackend unit ----------------

@needs_gems
def test_backend_channels_exclude_clinker_and_fluids(gems_backend, tmp_path):
    ids = set(gems_backend.hydrate_ids)
    assert {"CSHQ", "Portlandite", "C3AH6", "ettringite"} <= ids
    assert not ids & {"Alite", "Belite", "Aluminate", "Ferrite", "aq_gen", "gas_gen"}


@needs_gems
def test_backend_cnash_bundle_channels_and_react(tmp_path):
    """Default CNASH bundle (hydrates-only, no clinker phases): channels
    adapt, the suppression list collapses to the bundle's actual phases
    (here: none), and a react still closes."""
    from tinn.config import DEFAULT_GEMS_BUNDLE_LST
    from tinn.gems import GemsBackend, GemsWorker
    w = GemsWorker(str(REPO / DEFAULT_GEMS_BUNDLE_LST),
                   python_executable=str(GEMS_PYTHON), work_root=str(tmp_path))
    b = GemsBackend(w, 293.15)
    ids2 = set(b.hydrate_ids)
    assert {"CNASH", "Portlandite", "C4AH19", "ettringite"} <= ids2
    assert "CSHQ" not in ids2 and not ids2 & {"aq_gen", "gas_gen"}
    assert b._suppressed == ()
    inv = np.zeros(len(ELEMENT_IDS))
    r = b.react(_release(), 6e-10, inv)
    assert r.status == "ok"
    assert any(p.phase_id == "CNASH" for p in r.parcels)
    w.close()


@needs_gems
def test_sorption_structural_alkali_endmember_refused(tmp_path):
    """RT-04A: the PC/PC-Cl CSHQ solid solution carries KSiOH/NaSiOH
    endmembers, so additional Na surface sorption double-counts alkali
    uptake. The config is valid (alkali_exchange: true) but the engine must
    refuse it on the sorbent's actual endmember rows - a CSHQ name (no CNASH
    phase in this bundle) does not prove it is alkali-free."""
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw["schedule"] = {"output_times_h": [2.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    cemdat = REPO / "gems_bundles" / "PHREEQC-cemdata18" / "cemdata18.dat"
    raw["sorption"] = {
        "operator": "phreeqc_surface", "phreeqc_dat": str(cemdat),
        "site_density_mol_per_mol": {"CSHQ-TobH": 0.02},
        "surface_species": [{"reaction": "Surf_sOH + Na+ = Surf_sONa + H+",
                             "log_k": 0.0,
                             "sorbed_elements": {"Na": 1.0, "H": -1.0}}],
        "elements": ["Na"], "alkali_exchange": True}
    cfg = TinnConfig.model_validate(raw)          # config is valid
    with pytest.raises(RuntimeError, match="structural alkali"):
        Engine(cfg)


@needs_gems
def test_sorption_stage_transient_failure_rejects_and_retries(tmp_path):
    """RT-05: a PHREEQC transient failure in the S stage is converted to a
    StepReject('sorption_failure') so the engine halves dt and retries,
    instead of terminating the run. Config/stoichiometry errors still
    propagate (not covered here). A fake operator raises
    BackendTransientError on its first call, then delegates to the real one."""
    from tinn import backend as backend_mod
    cemdat = REPO / "gems_bundles" / "PHREEQC-cemdata18" / "cemdata18.dat"
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw["schedule"] = {"output_times_h": [4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    raw["sorption"] = {"operator": "phreeqc_surface", "phreeqc_dat": str(cemdat),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.02},
                       "surface_species": [
                           {"reaction": "Surf_sOH + SO4-2 = Surf_sSO4- + OH-",
                            "log_k": 0.5,
                            "sorbed_elements": {"S": 1.0, "O": 3.0, "H": -1.0}}],
                       "elements": ["S"]}
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)

    # take one real step so the reactors are wet and the S stage actually
    # calls sorb; then inject a transient failure and prove the except path
    # converts it to a StepReject with a diagnostic detail.
    st1, rej0, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej0 is None and st1 is not None
    real_sorb = eng._sorb_op.sorb

    def always_fail(*a, **k):
        raise backend_mod.BackendTransientError("injected PHREEQC failure")

    eng._sorb_op.sorb = always_fail
    _, rej, _ = eng.try_step(st1, 2.0)
    assert rej is not None and rej.reason == "sorption_failure"
    assert "cluster" in rej.detail and "water" in rej.detail
    eng._sorb_op.sorb = real_sorb                  # restore (unused hereafter)

    # end-to-end: the run survives the injected failure by halving dt
    eng2 = Engine(cfg)
    real2 = eng2._sorb_op.sorb
    c2 = {"n": 0}

    def flaky2(*a, **k):
        c2["n"] += 1
        if c2["n"] == 1:
            raise backend_mod.BackendTransientError("injected PHREEQC failure")
        return real2(*a, **k)

    eng2._sorb_op.sorb = flaky2
    state, summary = eng2.run(out_dir=str(tmp_path / "run"))
    assert state.reject_counts.get("sorption_failure", 0) >= 1
    assert state.accept_count >= 1                  # dt halved, then advanced


def _release():
    return {"C3S": 2e-12, "C2S": 3e-13, "C3A": 2e-13, "C4AF": 1.5e-13}


@needs_gems
def test_backend_react_element_closure(gems_backend):
    inv = np.zeros(len(ELEMENT_IDS))
    water = 6e-10
    r = gems_backend.react(_release(), water, inv)
    e_in = inv.copy()
    for p, m in _release().items():
        e_in += element_vector(REG.get(p).formula, m)
    e_in += element_vector(REG.get("H2O").formula, water) + r.injected_elements
    e_out = sum((pc.elements for pc in r.parcels), np.zeros(len(ELEMENT_IDS)))
    e_out = e_out + r.residual_inventory
    e_out += element_vector(REG.get("H2O").formula, water - r.water_consumed_mol)
    assert np.abs(e_in - e_out).max() / e_in.max() <= 1e-12
    assert 0.0 < r.water_consumed_mol < water
    assert r.ph_status == "ok" and 12.4 <= r.ph <= 13.9


@needs_gems
def test_reaction_result_reports_aqueous_diagnostics(gems_backend):
    """RT-W0: read-only aqueous diagnostics ride every react() result.

    aqueous_elements is the full aq_gen element vector (solvent H/O
    included) back-scaled to cluster units; it must be finite,
    non-negative, and consistent with the reported solvent split."""
    inv = np.zeros(len(ELEMENT_IDS))
    water = 6e-10
    r = gems_backend.react(_release(), water, inv)
    assert r.ionic_strength_status == "ok"
    assert math.isfinite(r.ionic_strength) and r.ionic_strength > 0.0
    assert math.isfinite(r.aqueous_h2o_mol) and 0.0 < r.aqueous_h2o_mol < water
    aq = r.aqueous_elements
    assert aq.shape == (len(ELEMENT_IDS),)
    assert np.all(np.isfinite(aq)) and np.all(aq >= 0.0)
    h = aq[ELEMENT_IDS.index("H")]
    o = aq[ELEMENT_IDS.index("O")]
    # solvent dominates the aqueous phase: H >= 2*h2o, O >= h2o
    assert h >= 2.0 * r.aqueous_h2o_mol * (1.0 - 1e-9)
    assert o >= r.aqueous_h2o_mol * (1.0 - 1e-9)
    # solvent-subtracted aqueous solutes match the residual inventory up
    # to the gas phase (residual = aq_gen + gas_gen + suppressed traces
    # minus solvent; gas carries only O2-seed dust under sealed hydration)
    h2o_vec = element_vector(REG.get("H2O").formula, r.aqueous_h2o_mol)
    solutes = aq - h2o_vec
    assert np.abs(solutes - r.residual_inventory).max() <= 1e-6 * aq.max()


@needs_gems
def test_backend_scaling_invariance(gems_backend):
    inv = np.zeros(len(ELEMENT_IDS))
    a = gems_backend.react(_release(), 6e-10, inv)
    b = gems_backend.react({k: 10 * v for k, v in _release().items()}, 6e-9, inv)
    pa = {pc.phase_id: pc.mol for pc in a.parcels if pc.mol > 1e-16}
    pb = {pc.phase_id: pc.mol for pc in b.parcels if pc.mol > 1e-15}
    assert set(pa) == set(pb)
    for phase in pa:
        assert pb[phase] / pa[phase] == pytest.approx(10.0, rel=1e-5)


@needs_gems
def test_backend_seed_only_on_fresh_inventory(gems_backend):
    o_idx = ELEMENT_IDS.index("O")
    fresh = gems_backend.react(_release(), 6e-10, np.zeros(len(ELEMENT_IDS)))
    assert fresh.injected_elements[o_idx] > 0.0  # O2 redox seed
    again = gems_backend.react(_release(), 6e-10, fresh.residual_inventory)
    # no second seed: only solver-closure residual dust remains (D4 booking)
    assert abs(again.injected_elements[o_idx]) < 1e-3 * fresh.injected_elements[o_idx]
    assert again.status == "ok"


@needs_gems
def test_backend_parcels_have_volumes_and_composition(gems_backend):
    r = gems_backend.react(_release(), 6e-10, np.zeros(len(ELEMENT_IDS)))
    significant = [pc for pc in r.parcels if pc.mol > 1e-16]
    assert {pc.phase_id for pc in significant} >= {"CSHQ", "Portlandite"}
    for pc in significant:
        assert pc.skel_vol_cm3 > 0.0
        assert pc.elements.sum() > 0.0
    cshq = next(pc for pc in significant if pc.phase_id == "CSHQ")
    # CSHQ composition is variable (endmember sum) — it carries Ca, Si AND water
    for el in ("Ca", "Si", "H"):
        assert cshq.elements[ELEMENT_IDS.index(el)] > 0.0


# ---------------- engine 3D coupling (M4 DoD) ----------------

@needs_gems
def test_gems_run_completes_with_61_closure(gems_run):
    _, _, state, summary, _ = gems_run
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    assert rep.metrics["voxel_identity_max_err"] <= 1e-12
    w_err = abs(state.water_free_mol + state.water_gel_mol + state.water_bound_mol
                - state.initial_water_mol)
    assert w_err <= 1e-8 * state.initial_water_mol + 1e-24
    assert state.accept_count >= 2


@needs_gems
def test_gems_run_produces_gems_phases(gems_run):
    _, _, state, _, _ = gems_run
    by_phase = dict(zip(state.hydrate_ids, state.hydrate_mol))
    assert by_phase["CSHQ"] > 0.0
    assert by_phase["Portlandite"] > 0.0
    # dense placement matches the envelope volume ledger per channel
    for i, h in enumerate(state.hydrate_ids):
        dense = float(state.hydrate_fraction[i].sum())
        assert dense == pytest.approx(float(state.hydrate_env_vol_vox[i]),
                                      abs=1e-9, rel=1e-6)


@needs_gems
def test_gems_run_cluster_ph_in_band(gems_run):
    _, _, _, summary, _ = gems_run
    phs = [v for row in summary["outputs"]
           for v in row["ledger_metrics"].get("cluster_ph", {}).values()]
    assert phs, "gems runs must report per-cluster pH"
    assert all(12.4 <= p <= 13.9 for p in phs)


@needs_gems
def test_gems_run_injected_elements_tracked_and_tiny(gems_run):
    _, _, state, _, _ = gems_run
    assert np.any(state.injected_elements > 0.0)  # seed/floors tracked, not hidden
    scale = float(np.abs(state.initial_elements).max())
    assert float(state.injected_elements.max()) <= 1e-4 * scale


@needs_gems
def test_gems_run_parcel_table_immutable_rows(gems_run):
    _, _, state, _, _ = gems_run
    assert len(state.parcels["time_h"]) > 0
    assert set(state.parcels["hydrate"]) <= set(state.hydrate_ids)
    bulk = np.asarray(state.parcels["bulk_vol_vox"])
    skel = np.asarray(state.parcels["skel_vol_vox"])
    assert np.all(bulk >= skel - 1e-30)  # envelope >= skeleton


@needs_gems
def test_gems_run_no_redissolution(gems_run):
    _, _, _, summary, _ = gems_run
    rows = summary["outputs"]
    for h in rows[-1]["hydrate_mol"]:
        series = [r["hydrate_mol"].get(h, 0.0) for r in rows]
        assert all(b >= a - 1e-30 for a, b in zip(series, series[1:])), h


@needs_gems
def test_gems_run_sanity_band_report(gems_run):
    _, _, _, summary, _ = gems_run
    band = summary["sanity_band"]
    assert "not scientific validation" in band["note"]
    names = {c["check"] for c in band["checks"]}
    assert "alpha_order_C3S_ge_C2S" in names
    assert "cluster_ph_band_12.4_13.9" in names
    assert all(c["status"] in ("pass", "warn") for c in band["checks"])


@needs_gems
def test_gems_run_restart_bitwise(gems_run):
    cfg, _, straight, _, out = gems_run
    mid = load_checkpoint(str(out / "ckpt_000"), REG)   # t = 2 h
    assert mid.hydrate_ids == straight.hydrate_ids
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.dense_hash() == straight.dense_hash()
    assert restarted.full_hash() == straight.full_hash()


# ---------------- config (runs everywhere) ----------------

def test_config_gems_gel_porosity_defaults_and_validation():
    from tinn.config import DEFAULT_GEMS_GEL_POROSITY
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(encoding="utf-8"))
    cfg = TinnConfig.model_validate(raw)
    assert cfg.chemistry.gems_gel_porosity == DEFAULT_GEMS_GEL_POROSITY
    # legacy hash compatibility: the filled default hashes as the pre-CNASH
    # form, so a config that omitted the map keeps its pre-rev.2 config_hash
    # (old checkpoints stay restartable)
    raw_leg = json.loads(json.dumps(raw))
    raw_leg["chemistry"]["gems_gel_porosity"] = {"CSHQ": 0.28}
    assert TinnConfig.model_validate(raw_leg).config_hash() == cfg.config_hash()
    raw["chemistry"]["gems_gel_porosity"] = {"CSHQ": 1.5}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw)
    raw2 = json.loads((REPO / "examples" / "c3s_32.json").read_text(encoding="utf-8"))
    raw2["chemistry"]["gems_gel_porosity"] = {"CSHQ": 0.28}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw2)
