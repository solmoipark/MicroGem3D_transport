"""E1 tests (PRD 2.3/6.1 rev.3): solid-solution endmember preservation —
worker protocol, parcel carriage, engine booking, closure invariants,
FORMAT_VERSION 3 checkpoints, derived observables."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import analysis, ledger
from tinn.backend import Parcel, ReactionResult, STATUS_OK, StoichiometricBackend
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, default_registry,
                           element_vector)
from tinn.storage import FORMAT_VERSION, StorageError, load_checkpoint, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
CNASH_BUNDLE = REPO / "gems_bundles" / "CNASH" / "Test-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))
needs_gems = pytest.mark.skipif(
    not (BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")

REG = default_registry()


def _short_cfg(**over):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.05]}}}
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    raw.update(over)
    return TinnConfig.model_validate(raw)


# ---------------- backend metadata and parcels ----------------

def test_stoich_backend_single_endmember_metadata():
    b = StoichiometricBackend(_short_cfg().chemistry.stoichiometric_rules, REG)
    assert b.hydrate_endmembers == {h: (h,) for h in HYDRATE_PHASE_IDS}
    for h in HYDRATE_PHASE_IDS:
        assert np.array_equal(b.endmember_elements[h],
                              element_vector(REG.get(h).formula, 1.0))
    # stoich parcels leave endmember_mol=None (single-endmember convention)
    r = b.react({"C3S": 1e-12}, 1e-9, np.zeros(len(ELEMENT_IDS)))
    assert r.status == STATUS_OK
    assert all(pc.endmember_mol is None for pc in r.parcels)


def test_engine_books_single_endmember_parcels():
    """FakeSnapshot-style single-endmember booking: endmember ledger mirrors
    hydrate_mol channel-for-channel and both closure checks pass."""
    cfg = _short_cfg()
    eng = Engine(cfg)   # stoichiometric
    state, _ = eng.run()
    h_of = {h: i for i, h in enumerate(state.hydrate_ids)}
    sums = np.zeros(len(state.hydrate_ids))
    for (h, dc), m in zip(state.endmember_ids, state.endmember_mol):
        assert dc == h                      # single-endmember universe
        sums[h_of[h]] += m
    assert np.allclose(sums, state.hydrate_mol, rtol=1e-12, atol=1e-24)
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    assert rep.metrics["endmember_sum_err_mol"] <= 1e-24


def test_engine_rejects_unknown_endmember_name():
    class BadBackend:
        backend_id = "stoichiometric"
        hydrate_ids = HYDRATE_PHASE_IDS
        mode = "incremental"

        def react(self, released_mol, water_available_mol, inventory,
                  solid_elements=None):
            mol = 1e-13
            entry = REG.get("CH")
            return ReactionResult(
                status=STATUS_OK,
                parcels=[Parcel(phase_id="CH", mol=mol,
                                elements=element_vector(entry.formula, mol),
                                skel_vol_cm3=mol * entry.skeleton_molar_volume_cm3,
                                endmember_mol={"NOT_A_DC": mol})],
                water_consumed_mol=0.0,
                residual_inventory=np.asarray(inventory).copy())

    eng = Engine(_short_cfg(), reaction_backend=BadBackend())
    with pytest.raises(RuntimeError, match="unknown endmember"):
        eng.try_step(eng.initial_state(), 2.0)


# ---------------- ledger invariants ----------------

def test_ledger_detects_endmember_sum_corruption():
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    state.endmember_mol = state.endmember_mol.copy()
    j = int(np.argmax(state.endmember_mol))
    state.endmember_mol[j] *= 1.5
    rep = ledger.check_all(state, REG)
    assert any(v.startswith("balance_endmember:") for v in rep.violations)


def test_ledger_detects_endmember_element_corruption():
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    state.endmember_elements = state.endmember_elements.copy()
    j = int(np.argmax(state.endmember_mol))
    state.endmember_elements[j, 0] += 1.0   # warp one formula row
    rep = ledger.check_all(state, REG)
    assert any(v.startswith("balance_endmember_elements:") for v in rep.violations)


def test_boundary_exchange_reservation_enters_closure():
    """The RT-W2 reserved vector participates in the element-closure identity:
    zeros are a no-op; a nonzero value shifts the expected side exactly."""
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    assert ledger.check_all(state, REG).ok
    state.boundary_exchanged_elements = state.boundary_exchanged_elements.copy()
    state.boundary_exchanged_elements[0] += 1e-6
    rep = ledger.check_all(state, REG)
    assert any(v.startswith("balance_element:") for v in rep.violations)


# ---------------- storage: FORMAT_VERSION 3 ----------------

def test_checkpoint_v3_roundtrip_bitwise_with_endmembers(tmp_path):
    cfg = _short_cfg()
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    save_checkpoint(state, str(tmp_path), "em")
    loaded = load_checkpoint(str(tmp_path / "em"), REG)
    assert loaded.endmember_ids == state.endmember_ids
    assert np.array_equal(loaded.endmember_mol, state.endmember_mol)
    assert np.array_equal(loaded.endmember_elements, state.endmember_elements)
    assert loaded.full_hash() == state.full_hash()


def test_checkpoint_v2_explicitly_rejected(tmp_path):
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    save_checkpoint(state, str(tmp_path), "v2ish")
    hdr = tmp_path / "v2ish" / "header.json"
    payload = json.loads(hdr.read_text(encoding="utf-8"))
    payload["format_version"] = 2
    hdr.write_text(json.dumps(payload), encoding="utf-8")
    man = tmp_path / "v2ish" / "manifest.json"
    manifest = json.loads(man.read_text(encoding="utf-8"))
    import hashlib
    manifest["header.json"] = hashlib.sha256(hdr.read_bytes()).hexdigest()
    man.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageError, match="endmember ledger"):
        load_checkpoint(str(tmp_path / "v2ish"), REG)
    assert FORMAT_VERSION == 3


def test_run_guard_rejects_endmember_universe_mismatch(tmp_path):
    cfg = _short_cfg()
    eng = Engine(cfg)
    state, _ = eng.run()
    state.endmember_ids = tuple(list(state.endmember_ids)[::-1])  # reordered
    with pytest.raises(RuntimeError, match="endmember universe"):
        eng.run(state=state)


# ---------------- derived observables ----------------

def test_solid_solution_composition_math():
    """Hand-built two-endmember channel: ratios follow the formula rows."""
    cfg = _short_cfg()
    state = Engine(cfg).initial_state()
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    state.endmember_ids = (("CSH", "rich"), ("CSH", "lean"), ("CH", "CH"))
    state.endmember_mol = np.array([2.0, 1.0, 5.0])
    rows = np.zeros((3, len(ELEMENT_IDS)))
    rows[0, el["Ca"]], rows[0, el["Si"]], rows[0, el["H"]] = 1.8, 1.0, 4.0
    rows[1, el["Ca"]], rows[1, el["Si"]], rows[1, el["H"]] = 0.8, 1.0, 2.0
    rows[2, el["Ca"]] = 1.0
    state.endmember_elements = rows
    ss = analysis.solid_solution_composition(state)
    assert "CH" not in ss                       # single-endmember: no ratios
    got = ss["CSH"]
    assert got["ca_si"] == pytest.approx((2 * 1.8 + 1 * 0.8) / 3.0)
    assert got["h_si"] == pytest.approx((2 * 4.0 + 1 * 2.0) / 3.0)
    assert got["endmember_mol"] == {"rich": 2.0, "lean": 1.0}


# ---------------- GEMS path (real worker) ----------------

@needs_gems
def test_gems_parcels_carry_endmembers_and_scale(tmp_path):
    from tinn.gems import GemsBackend, GemsWorker
    w = GemsWorker(str(CNASH_BUNDLE), python_executable=str(GEMS_PYTHON),
                   work_root=str(tmp_path))
    b = GemsBackend(w, 293.15)
    assert set(b.hydrate_endmembers["CNASH"]) >= {"5CA", "T2C-CNASHss",
                                                  "TobH-CNASHss"}
    rel = {"C3S": 2e-12, "C2S": 3e-13, "C3A": 2e-13, "C4AF": 1.5e-13}
    r1 = b.react(rel, 6e-10, np.zeros(len(ELEMENT_IDS)))
    for pc in r1.parcels:
        assert pc.endmember_mol is not None
        assert sum(pc.endmember_mol.values()) == pytest.approx(
            pc.mol, rel=1e-9, abs=1e-30)
        # every endmember name is declared by the bundle
        assert set(pc.endmember_mol) <= set(b.hydrate_endmembers[pc.phase_id])
    # canonical-scale invariance: x10 inputs -> x10 endmembers
    r10 = b.react({k: 10 * v for k, v in rel.items()}, 6e-9,
                  np.zeros(len(ELEMENT_IDS)))
    em1 = {(pc.phase_id, dc): m for pc in r1.parcels
           for dc, m in pc.endmember_mol.items() if m > 1e-16}
    em10 = {(pc.phase_id, dc): m for pc in r10.parcels
            for dc, m in pc.endmember_mol.items() if m > 1e-15}
    for key in em1:
        assert em10[key] / em1[key] == pytest.approx(10.0, rel=1e-4)
    w.close()


@needs_gems
def test_gems_run_endmember_closure_and_observables(tmp_path):
    raw = json.loads((EXAMPLES / "opc_cnash_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    state, _ = eng.run(out_dir=str(tmp_path / "run"))
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    # a real solid solution resolved into multiple endmembers
    cnash = [m for (h, _), m in zip(state.endmember_ids, state.endmember_mol)
             if h == "CNASH" and m > 1e-16]
    assert len(cnash) >= 2
    ss = analysis.solid_solution_composition(state)
    assert 0.6 <= ss["CNASH"]["ca_si"] <= 2.5   # physical C-(A-)S-H range
    # restart equivalence extends to the endmember ledger (full_hash covers it)
    ck = sorted((tmp_path / "run").glob("ckpt_*"))[0]
    mid = load_checkpoint(str(ck), REG)
    resumed, _ = Engine(cfg).run(state=mid)
    assert resumed.full_hash() == state.full_hash()


# ---------------- review-gate regressions (E1 adversarial review) ----------------

def test_remove_guard_scales_with_member_count():
    """bincount-vs-pairwise summation legitimately diverges ~n*eps on a
    full-channel removal leg; the over-request allowance must scale with the
    member count instead of hard-crashing at 128^3 (review finding)."""
    from tinn import morphology
    n = 16
    hyd = np.zeros((4, n, n, n))
    rng = np.random.default_rng(11)
    hyd[0] = rng.random((n, n, n)) * 1e-3
    member = np.ones((n, n, n), dtype=bool)
    available = float(np.where(member, hyd[0], 0.0).sum())
    n_member = int(member.sum())
    # a request just above the OLD 1e-12 allowance but inside the scaled one
    req = available * (1.0 + 5e-13 + 1e-16 * n_member * 0.5)
    _, removed = morphology.remove(hyd, 0, req, member)
    assert removed == pytest.approx(available, rel=1e-9)   # clamped, no crash
    hyd[0][:] = 0.1
    with pytest.raises(ValueError, match="exceeds available"):
        morphology.remove(hyd, 0, float(np.sum(hyd[0])) * 1.01, member)


def test_ledger_1b_tolerates_full_vanish_dust():
    """A channel that fully redissolves keeps signed dust ~eps x turnover;
    the closure bounds reference the GLOBAL holdings scale so legitimate dust
    passes while real corruption (>> rtol x global) still trips."""
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    state.endmember_mol = state.endmember_mol.copy()
    state.hydrate_mol = state.hydrate_mol.copy()
    state.hydrate_elements_ch = state.hydrate_elements_ch.copy()
    ch = int(np.argmax(state.hydrate_mol))
    h = state.hydrate_ids[ch]
    j = next(i for i, (hh, _) in enumerate(state.endmember_ids) if hh == h)
    X = float(state.hydrate_mol[ch])
    # simulate the vanished channel: every ledger leg holds INDEPENDENT
    # signed dust (~eps x pre-vanish turnover) as the signed commit leaves it
    state.hydrate_mol[ch] = -3e-16 * X
    state.endmember_mol[j] = 1e-16 * X
    state.hydrate_elements_ch[ch] *= 0.5e-16
    rep = ledger.check_all(state, REG)
    assert not any(v.startswith("balance_endmember") for v in rep.violations), \
        rep.violations


def test_engine_rejects_none_split_for_multi_endmember_phase():
    class NoneSplitBackend:
        backend_id = "stoichiometric"
        hydrate_ids = ("CSH",)
        hydrate_endmembers = {"CSH": ("TobH", "JenD")}
        endmember_elements = {"TobH": np.zeros(len(ELEMENT_IDS)),
                              "JenD": np.zeros(len(ELEMENT_IDS))}
        mode = "incremental"

        def react(self, released_mol, water_available_mol, inventory,
                  solid_elements=None):
            entry = REG.get("CSH")
            mol = 1e-13
            return ReactionResult(
                status=STATUS_OK,
                parcels=[Parcel(phase_id="CSH", mol=mol,
                                elements=element_vector(entry.formula, mol),
                                skel_vol_cm3=mol * entry.skeleton_molar_volume_cm3,
                                endmember_mol=None)],
                water_consumed_mol=0.0,
                residual_inventory=np.asarray(inventory).copy())

    eng = Engine(_short_cfg(), reaction_backend=NoneSplitBackend())
    with pytest.raises(RuntimeError, match="omitted the endmember split"):
        eng.try_step(eng.initial_state(), 2.0)


def test_from_geometry_requires_rows_for_multi_endmember():
    from tinn.geometry import initialize_rve
    from tinn.state import SimulationState
    cfg = _short_cfg()
    rve = initialize_rve(cfg, REG)
    with pytest.raises(ValueError, match="no endmember element rows"):
        SimulationState.from_geometry(
            cfg, REG, rve, "stoichiometric", hydrate_ids=("CSH",),
            hydrate_endmembers={"CSH": ("TobH", "JenD")},
            endmember_elements=None)
