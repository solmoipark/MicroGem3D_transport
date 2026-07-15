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
def test_backend_channels_exclude_clinker_and_fluids(gems_backend):
    ids = set(gems_backend.hydrate_ids)
    assert {"CSHQ", "Portlandite", "C3AH6", "ettringite"} <= ids
    assert not ids & {"Alite", "Belite", "Aluminate", "Ferrite", "aq_gen", "gas_gen"}


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
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(encoding="utf-8"))
    cfg = TinnConfig.model_validate(raw)
    assert cfg.chemistry.gems_gel_porosity == {"CSHQ": 0.28}
    raw["chemistry"]["gems_gel_porosity"] = {"CSHQ": 1.5}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw)
    raw2 = json.loads((REPO / "examples" / "c3s_32.json").read_text(encoding="utf-8"))
    raw2["chemistry"]["gems_gel_porosity"] = {"CSHQ": 0.28}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw2)
