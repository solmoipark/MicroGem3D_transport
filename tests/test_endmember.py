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
    # E1 broke the format once (3); v4.0/RT broke it once more (4, PRD 2.3:
    # domain-keyed rows + boundary_water_mol) - the RT-W2 sanctioned break;
    # Tier 0 broke it once more (5, RT-P0b: frozen speciation + the
    # RESERVED Tier-1 sorbed inventory riding the same break)
    assert FORMAT_VERSION == 6


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


# ---------------- E2: per-cluster endmember pools (PRD 4.5 v3.0) ----------------

def _em_rows():
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    tob = np.zeros(len(ELEMENT_IDS))
    jen = np.zeros(len(ELEMENT_IDS))
    tob[el["Ca"]], tob[el["Si"]], tob[el["O"]], tob[el["H"]] = 0.8, 1.0, 3.0, 2.0
    jen[el["Ca"]], jen[el["Si"]], jen[el["O"]], jen[el["H"]] = 1.5, 1.0, 4.0, 3.0
    return tob, jen


class TwoEndmemberSnapshotBackend:
    """FakeSnapshotBackend variant with a two-endmember CSH channel: emits a
    fixed CSH amount at a scripted TobH/JenD split (plus optional CH), records
    the solid_elements it was fed, and closes elements/water like the W2 fake."""
    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "snapshot"

    def __init__(self, csh_mol: float, tob_frac: float, ch_mol: float = 0.0):
        self.csh_mol, self.tob_frac, self.ch_mol = csh_mol, tob_frac, ch_mol
        tob, jen = _em_rows()
        self.hydrate_endmembers = {h: (h,) for h in HYDRATE_PHASE_IDS}
        self.hydrate_endmembers["CSH"] = ("TobH", "JenD")
        self.endmember_elements = {
            h: element_vector(REG.get(h).formula, 1.0) for h in HYDRATE_PHASE_IDS
            if h != "CSH"}
        self.endmember_elements["TobH"] = tob
        self.endmember_elements["JenD"] = jen
        self.calls = []

    def react(self, released_mol, water_available_mol, inventory,
              solid_elements=None):
        self.calls.append({"released": dict(released_mol),
                           "solid_elements": None if solid_elements is None
                           else np.asarray(solid_elements).copy()})
        e_in = np.asarray(inventory, dtype=float).copy()
        for pid, mol in released_mol.items():
            e_in += element_vector(REG.get(pid).formula, mol)
        if solid_elements is not None:
            e_in += np.asarray(solid_elements)
        parcels = []
        if self.csh_mol > 0.0:
            m_t = self.csh_mol * self.tob_frac
            m_j = self.csh_mol - m_t
            tob, jen = _em_rows()
            vec = m_t * tob + m_j * jen
            entry = REG.get("CSH")
            parcels.append(Parcel(
                phase_id="CSH", mol=self.csh_mol, elements=vec,
                skel_vol_cm3=self.csh_mol * entry.skeleton_molar_volume_cm3,
                endmember_mol={"TobH": m_t, "JenD": m_j}))
            e_in = e_in - vec
        if self.ch_mol > 0.0:
            entry = REG.get("CH")
            vec = element_vector(entry.formula, self.ch_mol)
            parcels.append(Parcel(
                phase_id="CH", mol=self.ch_mol, elements=vec,
                skel_vol_cm3=self.ch_mol * entry.skeleton_molar_volume_cm3))
            e_in = e_in - vec
        h_idx = ELEMENT_IDS.index("H")
        h_new = sum(float(pc.elements[h_idx]) for pc in parcels)
        h_owned = (0.0 if solid_elements is None
                   else float(np.asarray(solid_elements)[h_idx]))
        water_used = (h_new - h_owned) / 2.0
        e_in = e_in + element_vector(REG.get("H2O").formula, water_used)
        return ReactionResult(status=STATUS_OK, parcels=parcels,
                              water_consumed_mol=water_used,
                              residual_inventory=e_in)


def _csh_slice(eng):
    return eng._em_slice["CSH"]


def test_pool_replacement_and_pool_ratio_feed():
    """Core E2 semantics: (a) a solved cluster's pool IS its own parcels after
    the step; (b) the next step's fed composition follows the cluster's pool
    ratio, NOT the global average, at the volume-share amount."""
    cfg = _short_cfg()
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b1)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    sl = _csh_slice(eng)
    # (a) absolute replacement: single wet cluster owns the whole parcel split
    pools = t1.cluster_endmember_mol
    assert pools.shape == (t1.cluster_inventory.shape[0], eng._n_em)
    total = pools[:, sl].sum(axis=0)
    assert total[0] == pytest.approx(1.5e-12, rel=1e-12)   # TobH
    assert total[1] == pytest.approx(1.5e-12, rel=1e-12)   # JenD
    # (b) bias ONE cluster's pool away from the global 50:50 (within holdings:
    # a 60:40 feed keeps the post-commit ledger non-negative) and step again —
    # the fed solid_elements must follow the pool, not the global ratio
    t1.cluster_endmember_mol = pools.copy()
    rows = np.abs(pools[:, sl]).sum(axis=1)
    c = int(np.argmax(rows))                     # the materially wet cluster
    scale = float(pools[c, sl].sum())
    t1.cluster_endmember_mol[c, sl] = [0.6 * scale, 0.4 * scale]
    b2 = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    eng2 = Engine(cfg, reaction_backend=b2)
    t2, rej2, _ = eng2.try_step(t1, 2.0)
    assert rej2 is None
    fed = [k["solid_elements"] for k in b2.calls if k["solid_elements"] is not None]
    tob, jen = _em_rows()
    amount = float(t1.hydrate_mol[HYDRATE_PHASE_IDS.index("CSH")])
    want = amount * (0.6 * tob + 0.4 * jen)
    assert any(np.allclose(f, want, rtol=1e-9, atol=1e-24) for f in fed), \
        (fed, want)
    # and the pool was replaced again by the NEW parcels (25:75), no blending
    tot2 = t2.cluster_endmember_mol[:, sl].sum(axis=0)
    assert tot2[0] / (tot2[0] + tot2[1]) == pytest.approx(0.25, rel=1e-9)


def test_overdraft_feed_rejects_at_step_level():
    """The E2 review's silent-corruption repro: a pool biased far beyond the
    global holdings would commit a negative endmember ledger with every
    closure identity green. Ledger 1b(c) must now reject the step instead."""
    cfg = _short_cfg()
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b1)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    sl = _csh_slice(eng)
    pools = t1.cluster_endmember_mol.copy()
    c = int(np.argmax(np.abs(pools[:, sl]).sum(axis=1)))
    scale = float(pools[c, sl].sum())
    pools[c, sl] = [0.9 * scale, 0.1 * scale]   # feeds 2.7e-12 TobH of 1.5e-12
    t1.cluster_endmember_mol = pools
    b2 = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    t2, rej2, _ = Engine(cfg, reaction_backend=b2).try_step(t1, 2.0)
    assert t2 is None and rej2 is not None
    assert rej2.reason.startswith("balance_endmember_negative")


def test_empty_pool_falls_back_to_global_ratio():
    """A state without pools (legacy/fresh) feeds the global channel ratio —
    E1 behavior — instead of inventing a composition."""
    cfg = _short_cfg()
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b1)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    t1.cluster_endmember_mol = np.zeros((0, eng._n_em))    # wipe pools
    b2 = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    eng2 = Engine(cfg, reaction_backend=b2)
    _, rej2, _ = eng2.try_step(t1, 2.0)
    assert rej2 is None
    fed = [k["solid_elements"] for k in b2.calls if k["solid_elements"] is not None]
    tob, jen = _em_rows()
    amount = float(t1.hydrate_mol[HYDRATE_PHASE_IDS.index("CSH")])
    want = amount * (0.5 * tob + 0.5 * jen)                # global 50:50
    assert any(np.allclose(f, want, rtol=1e-9, atol=1e-24) for f in fed)


def test_frozen_cluster_keeps_its_pool():
    """A space-filling frozen cluster is NOT re-solved: its pool must survive
    the step bit-identically (composition memory does not evaporate)."""
    cfg = _short_cfg()
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b1)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    pools_before = t1.cluster_endmember_mol.copy()
    # an assemblage larger than the whole pore space -> space-filling freeze
    b2 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5, ch_mol=6e-10)
    eng2 = Engine(cfg, reaction_backend=b2)
    t2, rej2, _ = eng2.try_step(t1, 2.0)
    assert rej2 is None
    assert float(t2.unmet_mol.sum()) > float(t1.unmet_mol.sum())  # frozen, honest
    assert np.array_equal(t2.cluster_endmember_mol, pools_before)


def test_pool_shape_guard_rejects_corruption():
    cfg = _short_cfg()
    b = TwoEndmemberSnapshotBackend(csh_mol=1e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b)
    state = eng.initial_state()
    state.cluster_endmember_mol = np.zeros((7, eng._n_em))  # wrong row count
    with pytest.raises(RuntimeError, match="endmember pool"):
        eng.try_step(state, 2.0)


def test_incremental_backend_keeps_pools_empty_zeros():
    """The stoichiometric (incremental) path never populates pools; rows track
    the labeling as zeros so a later snapshot restart starts from fallback."""
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    assert state.cluster_endmember_mol.shape[0] == state.cluster_inventory.shape[0]
    assert not np.any(state.cluster_endmember_mol)


def test_checkpoint_roundtrips_pools_and_hash_covers_them(tmp_path):
    cfg = _short_cfg()
    b = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.4)
    eng = Engine(cfg, reaction_backend=b)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None and np.any(t1.cluster_endmember_mol)
    save_checkpoint(t1, str(tmp_path), "e2")
    loaded = load_checkpoint(str(tmp_path / "e2"), REG)
    assert np.array_equal(loaded.cluster_endmember_mol, t1.cluster_endmember_mol)
    assert loaded.full_hash() == t1.full_hash()
    loaded.cluster_endmember_mol = loaded.cluster_endmember_mol.copy()
    loaded.cluster_endmember_mol.flat[0] += 1e-15
    assert loaded.full_hash() != t1.full_hash()


def test_cluster_ca_si_and_map():
    """Per-cluster Ca/Si observable: ratios follow each cluster's OWN pool;
    dust pools are skipped; the slice map pins the full rendering contract —
    colormap direction (blue=lo, red=hi), [lo,hi] clamp, gray for ratio-less
    clusters, dark background, hi>lo guard (E2 review: none were asserted)."""
    cfg = _short_cfg()
    state = Engine(cfg).initial_state()
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    state.endmember_ids = (("CSH", "rich"), ("CSH", "lean"), ("CH", "CH"))
    rows = np.zeros((3, len(ELEMENT_IDS)))
    rows[0, el["Ca"]], rows[0, el["Si"]] = 1.8, 1.0
    rows[1, el["Ca"]], rows[1, el["Si"]] = 0.8, 1.0
    rows[2, el["Ca"]] = 1.0
    state.endmember_elements = rows
    state.cluster_endmember_mol = np.array([
        [2.0, 1.0, 5.0],       # cluster 0: Ca/Si = (2*1.8+1*0.8)/3
        [0.5, 3.0, 0.0],       # cluster 1: Ca/Si = (0.5*1.8+3*0.8)/3.5
        [1e-30, 0.0, 0.0]])    # cluster 2: dust -> skipped
    got = analysis.cluster_ca_si(state)
    assert set(got) == {0, 1}
    assert got[0] == pytest.approx((2 * 1.8 + 1 * 0.8) / 3.0)
    assert got[1] == pytest.approx((0.5 * 1.8 + 3 * 0.8) / 3.5)
    n = state.grid_size
    state.cluster_id[n // 2, :4, :4] = 0
    state.cluster_id[n // 2, 8:12, 8:12] = 1
    state.cluster_id[n // 2, 16:18, 16:18] = 2     # dust cluster: no ratio
    img = analysis.casi_map_rgb(state)
    assert img is not None and img.shape == (n, n, 3)
    # direction: cluster 0 (1.467) is redder than cluster 1 (0.943)
    assert img[0, 0][0] > img[10, 10][0] and img[0, 0][2] < img[10, 10][2]
    assert img[10, 10][2] > img[10, 10][0]                 # near-lo: blue wins
    assert tuple(img[16, 16]) == (120, 120, 120)           # ratio-less: gray
    assert tuple(img[24, 24]) == (30, 30, 30)              # background stays dark
    # clamp: with hi below both ratios the pixel saturates at exact full red
    img2 = analysis.casi_map_rgb(state, lo=0.5, hi=0.6)
    assert tuple(img2[0, 0]) == (255, 60, 60)
    with pytest.raises(ValueError, match="hi > lo"):
        analysis.casi_map_rgb(state, lo=1.5, hi=1.5)


def test_cluster_ca_si_picks_dominant_silicate_channel():
    """E2 review finding: summing every Si-bearing solid solution blends
    C-S-H with hydrogarnet/M-S-H into a phase that does not exist. Exactly
    one channel — the largest pooled-Si solid solution — must be read."""
    cfg = _short_cfg()
    state = Engine(cfg).initial_state()
    el = {e: i for i, e in enumerate(ELEMENT_IDS)}
    state.hydrate_ids = ("CSH", "HG")
    state.endmember_ids = (("CSH", "rich"), ("CSH", "lean"),
                           ("HG", "a"), ("HG", "b"))
    rows = np.zeros((4, len(ELEMENT_IDS)))
    rows[0, el["Ca"]], rows[0, el["Si"]] = 1.8, 1.0
    rows[1, el["Ca"]], rows[1, el["Si"]] = 0.8, 1.0
    rows[2, el["Ca"]], rows[2, el["Si"]] = 3.0, 0.84   # siliceous hydrogarnet
    rows[3, el["Ca"]], rows[3, el["Si"]] = 3.0, 0.84
    state.endmember_elements = rows
    state.cluster_endmember_mol = np.array([[2.0, 1.0, 0.1, 0.1]])
    assert analysis.casi_channel(state) == "CSH"
    got = analysis.cluster_ca_si(state)
    # hydrogarnet does NOT pollute the C-S-H ratio
    assert got[0] == pytest.approx((2 * 1.8 + 1 * 0.8) / 3.0)
    state.cluster_endmember_mol = np.array([[0.01, 0.01, 5.0, 5.0]])
    assert analysis.casi_channel(state) == "HG"
    assert analysis.cluster_ca_si(state)[0] == pytest.approx(3.0 / 0.84)


def test_covered_portion_feed_blends_pool_and_global():
    """E2 review fix: the pool is authoritative ONLY for the mass it actually
    remembers (covered = min(psum, amount)); the remainder is fed at the
    global ratio, so a small pool can never define the composition of a much
    larger amount (the sliver/merge overdraft path)."""
    cfg = _short_cfg()
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b1)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    sl = _csh_slice(eng)
    pools = t1.cluster_endmember_mol
    c = int(np.argmax(np.abs(pools[:, sl]).sum(axis=1)))
    amount = float(t1.hydrate_mol[HYDRATE_PHASE_IDS.index("CSH")])
    # the pool remembers only HALF the amount, as pure TobH; the uncovered
    # half is fed at the global 50:50 -> TobH 3/4, JenD 1/4 of the amount
    t1.cluster_endmember_mol = pools.copy()
    t1.cluster_endmember_mol[c, sl] = [amount / 2.0, 0.0]
    b2 = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.5)
    eng2 = Engine(cfg, reaction_backend=b2)
    _, rej2, _ = eng2.try_step(t1, 2.0)
    assert rej2 is None
    fed = [k["solid_elements"] for k in b2.calls if k["solid_elements"] is not None]
    tob, jen = _em_rows()
    want = amount * (0.75 * tob + 0.25 * jen)
    assert any(np.allclose(f, want, rtol=1e-9, atol=1e-24) for f in fed), \
        (fed, want)


class PerWaterSnapshotBackend(TwoEndmemberSnapshotBackend):
    """Scripted per-cluster assemblage keyed on the cluster's water volume:
    the big pocket gets a small pure-TobH CSH (solves), the small pocket a
    space-filling CH slab (freezes)."""

    def __init__(self, split_water_mol: float):
        super().__init__(csh_mol=0.0, tob_frac=0.5)
        self.split_water_mol = split_water_mol

    def react(self, released_mol, water_available_mol, inventory,
              solid_elements=None):
        if water_available_mol > self.split_water_mol:
            self.csh_mol, self.tob_frac, self.ch_mol = 2e-13, 1.0, 0.0
        else:
            self.csh_mol, self.tob_frac, self.ch_mol = 0.0, 0.5, 2.5e-12
        r = super().react(released_mol, water_available_mol, inventory,
                          solid_elements=solid_elements)
        self.calls[-1]["water"] = water_available_mol
        return r


def test_mixed_frozen_and_solved_clusters_keep_pool_semantics():
    """The E2 mutant killer (review finding: single-cluster fixtures let
    `pools_prev = parcel_em` survive): a two-pocket step where A solves and B
    freezes must (a) feed each cluster its OWN pool composition, (b) replace
    ONLY A's pool with its parcels, (c) keep B's pool bit-identical, and
    (d) discard B's oversized assemblage entirely."""
    from tinn import transport
    from tinn.state import SimulationState
    cfg = _short_cfg(kinetics={"kind": "tabulated",
                               "table": {"times_h": [0.0, 4.0],
                                         "alpha": {"C3S": [0.0, 0.0]}}})
    backend = PerWaterSnapshotBackend(split_water_mol=8e-12)
    eng = Engine(cfg, reaction_backend=backend)
    st = eng.initial_state()   # correct universe/arrays, then rebuild geometry
    n = st.grid_size
    tob, jen = _em_rows()
    csh = HYDRATE_PHASE_IDS.index("CSH")
    sl = _csh_slice(eng)
    st.anhydrous_fraction[:] = 0.0
    st.anhydrous_fraction[0] = 1.0             # C3S everywhere...
    st.hydrate_fraction[:] = 0.0
    st.capillary_liquid[:] = 0.0
    st.capillary_gas[:] = 0.0
    # pocket A: 6^3 box, one plane partly gas (room to displace water) — far
    # from the growth seeds at z=2 so placement cannot pinch the cluster
    st.anhydrous_fraction[0][2:8, 2:8, 2:8] = 0.0
    st.capillary_liquid[2:8, 2:8, 2:8] = 1.0
    st.capillary_liquid[6, 2:8, 2:8] = 0.3
    st.capillary_gas[6, 2:8, 2:8] = 0.7
    # pocket B: 4^3 box, far from A
    st.anhydrous_fraction[0][20:24, 20:24, 20:24] = 0.0
    st.capillary_liquid[20:24, 20:24, 20:24] = 1.0
    # owned CSH: 1.0 vox dense in A (2 half-voxels), 0.5 vox in B
    for z, y, x in ((2, 2, 2), (2, 2, 3)):
        st.hydrate_fraction[csh, z, y, x] = 0.5
        st.capillary_liquid[z, y, x] = 0.5
    st.hydrate_fraction[csh, 20, 20, 20] = 0.5
    st.capillary_liquid[20, 20, 20] = 0.5
    labels, k = transport.label_clusters(st.capillary_liquid)
    assert k == 2
    a_lab = int(labels[2, 2, 2])
    b_lab = int(labels[20, 20, 20])
    # ledgers consistent with the 1.5 vox dense envelope: total CSH = 1e-14
    # mol; volume shares A = 1.0/1.5, B = 0.5/1.5 of that
    total_csh = 1e-14
    amt_a = total_csh * (1.0 / 1.5)
    amt_b = total_csh * (0.5 / 1.5)
    st.hydrate_mol[:] = 0.0
    st.hydrate_mol[csh] = total_csh
    st.hydrate_env_vol_vox[:] = 0.0
    st.hydrate_env_vol_vox[csh] = 1.5
    st.endmember_mol[:] = 0.0
    st.endmember_mol[sl] = [amt_a, amt_b]
    st.hydrate_elements_ch[:] = 0.0
    st.hydrate_elements_ch[csh] = amt_a * tob + amt_b * jen
    # pools LARGER than the owned amounts -> covered == amount, pure ratios
    pools = np.zeros((2, eng._n_em))
    pools[a_lab, sl] = [total_csh, 0.0]        # A remembers pure TobH
    pools[b_lab, sl] = [0.0, total_csh]        # B remembers pure JenD
    st.cluster_endmember_mol = pools
    st.cluster_inventory = np.zeros((2, len(ELEMENT_IDS)))
    st.cluster_id = labels
    vm_w = st.vm_vox(REG, "H2O")
    st.water_free_mol = float(st.capillary_liquid.sum()) / vm_w
    st.initial_water_mol = st.water_free_mol
    st.water_gel_mol = st.water_bound_mol = 0.0
    from tinn.registry import KINETIC_PHASE_IDS
    st.phase_mol = np.zeros(len(KINETIC_PHASE_IDS))
    st.phase_mol[KINETIC_PHASE_IDS.index("C3S")] = (
        float(st.anhydrous_fraction[0].sum()) / st.vm_vox(REG, "C3S"))
    st.initial_phase_mol = st.phase_mol.copy()
    st.unmet_mol = np.zeros(len(KINETIC_PHASE_IDS))
    st.injected_elements = np.zeros(len(ELEMENT_IDS))
    st.initial_elements = ledger.current_elements(st, REG)
    t2, rej, _ = eng.try_step(st, 2.0)
    assert rej is None
    fed = {kk["water"]: kk["solid_elements"] for kk in backend.calls}
    w_a = max(fed)
    w_b = min(fed)
    # (a) each cluster was fed its OWN pool composition at its own amount
    assert np.allclose(fed[w_a], amt_a * tob, rtol=1e-9, atol=1e-28)
    assert np.allclose(fed[w_b], amt_b * jen, rtol=1e-9, atol=1e-28)
    # (b)+(c) A's pool replaced by its parcels, B's untouched bit-for-bit
    out = t2.cluster_endmember_mol
    b_rows = [r for r in range(out.shape[0])
              if np.array_equal(out[r], pools[b_lab])]
    assert len(b_rows) == 1                    # frozen B survived unchanged
    a_rows = [r for r in range(out.shape[0])
              if np.allclose(out[r, sl], [2e-13, 0.0], rtol=1e-9, atol=1e-30)
              and r not in b_rows]
    assert len(a_rows) == 1                    # solved A = its own parcels
    # (d) B's oversized CH assemblage was discarded, not placed
    assert float(t2.hydrate_mol[HYDRATE_PHASE_IDS.index("CH")]) == 0.0


def test_pool_remap_rides_the_inventory_remap(monkeypatch):
    """E2 review mutant killer: the pool remap must see the IDENTICAL topology
    transition as the inventory remap (same label/liquid arrays, same n_new)
    and must carry pools with solved rows already replaced by parcels."""
    from tinn import transport
    cfg = _short_cfg()
    b = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b)
    seen = []
    real = transport.remap_inventories

    def spy(prev_labels, prev_liquid, new_labels, new_liquid, inv, n_new):
        seen.append((prev_labels, prev_liquid, new_labels, new_liquid,
                     np.asarray(inv).copy(), n_new))
        return real(prev_labels, prev_liquid, new_labels, new_liquid, inv, n_new)

    monkeypatch.setattr(transport, "remap_inventories", spy)
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None and len(seen) == 2
    inv_call, pool_call = seen
    for i in (0, 1, 2, 3):                     # argument identity, both calls
        assert inv_call[i] is pool_call[i]
    assert inv_call[5] == pool_call[5]
    sl = _csh_slice(eng)
    # the pool call's payload is pools_prev with the solved row = parcel split
    assert pool_call[4].shape[1] == eng._n_em
    tot = pool_call[4][:, sl].sum(axis=0)
    assert tot[0] == pytest.approx(1.5e-12, rel=1e-12)
    assert tot[1] == pytest.approx(1.5e-12, rel=1e-12)


def test_pool_remap_dryout_ignored_inventory_dryout_rejects(monkeypatch):
    """Design contract pin (E2 review): pools are compositional memory — a
    dropped pool row must NOT reject the step, while the same dryout signal
    from the INVENTORY remap must keep rejecting."""
    import dataclasses
    from tinn import transport
    cfg = _short_cfg()
    real = transport.remap_inventories

    def run_with(dryout_on_call: int):
        calls = {"n": 0}

        def fake(prev_labels, prev_liquid, new_labels, new_liquid, inv, n_new):
            res = real(prev_labels, prev_liquid, new_labels, new_liquid,
                       inv, n_new)
            calls["n"] += 1
            if calls["n"] == dryout_on_call:
                res = dataclasses.replace(res, dryout=[0])
            return res

        monkeypatch.setattr(transport, "remap_inventories", fake)
        b = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
        eng = Engine(cfg, reaction_backend=b)
        return eng.try_step(eng.initial_state(), 2.0)

    trial, rej, _ = run_with(dryout_on_call=2)      # pools: ignored
    assert rej is None and trial is not None
    trial2, rej2, _ = run_with(dryout_on_call=1)    # inventory: rejects
    assert trial2 is None and rej2 is not None
    assert rej2.reason == "cluster_dryout"


def test_ledger_flags_material_negative_endmember():
    """E2 review: an overdraft moves BOTH sides of every closure identity, so
    only the new non-negativity check (1b(c)) can witness it. Simulate the
    exact silent-corruption signature: endmember/hydrate/element ledgers move
    together, the phantom mass lands in the cluster inventory."""
    cfg = _short_cfg()
    state, _ = Engine(cfg).run()
    state.endmember_mol = state.endmember_mol.copy()
    state.hydrate_mol = state.hydrate_mol.copy()
    state.hydrate_elements_ch = state.hydrate_elements_ch.copy()
    state.cluster_inventory = state.cluster_inventory.copy()
    j = int(np.argmax(state.endmember_mol))
    h, _dc = state.endmember_ids[j]
    ch = state.hydrate_ids.index(h)
    delta = 2.0 * float(state.endmember_mol[j])
    state.endmember_mol[j] -= delta            # now materially negative
    state.hydrate_mol[ch] -= delta
    state.hydrate_elements_ch[ch] -= delta * state.endmember_elements[j]
    state.cluster_inventory[0] += delta * state.endmember_elements[j]
    rep = ledger.check_all(state, REG)
    neg = [v for v in rep.violations if v.startswith("balance_endmember_negative")]
    assert neg and h in neg[0]
    assert not any(v.startswith("balance_endmember:") for v in rep.violations)
    assert not any(v.startswith("balance_element") for v in rep.violations)


def test_report_emits_casi_only_for_pooled_runs(tmp_path):
    """E2 review: report() integration was unasserted — a pooled (snapshot)
    run must emit cluster_ca_si/casi_channel/casi_png rows and write the PNG;
    a stoichiometric run must emit none of them."""
    cfg = _short_cfg()
    b = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.4)
    out = tmp_path / "pooled"
    Engine(cfg, reaction_backend=b).run(out_dir=str(out))
    rep = analysis.report(str(out))
    rows = [r for r in rep["outputs"] if "cluster_ca_si" in r]
    assert rows
    assert rows[-1]["casi_channel"] == "CSH"
    assert (out / rows[-1]["casi_png"]).is_file()
    out2 = tmp_path / "stoich"
    Engine(cfg).run(out_dir=str(out2))
    rep2 = analysis.report(str(out2))
    assert all("cluster_ca_si" not in r and "casi_png" not in r
               for r in rep2["outputs"])


@needs_gems
def test_gems_pools_track_clusters_and_casi(tmp_path):
    raw = json.loads((EXAMPLES / "opc_cnash_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    state, _ = eng.run(out_dir=str(tmp_path / "run"))
    pools = state.cluster_endmember_mol
    assert pools.shape[0] == state.cluster_inventory.shape[0] > 0
    # pools tile the global ledger: their sum tracks endmember_mol up to the
    # remap drops (early age, fully wet: should be essentially exact)
    assert np.allclose(pools.sum(axis=0), state.endmember_mol,
                       rtol=1e-6, atol=1e-18)
    casi = analysis.cluster_ca_si(state)
    assert casi and all(0.6 <= v <= 2.5 for v in casi.values())
