"""W2 (PRD v2.2) tests: full per-cluster re-equilibration — snapshot backends,
cluster ownership (Scheme S), removal-first morphology, signed water/ledgers,
persistent worker equivalence."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import ledger, morphology
from tinn.backend import Parcel, ReactionResult, STATUS_OK
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                           default_registry, element_vector)
from tinn.storage import StorageError, load_checkpoint, save_checkpoint

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


class FakeSnapshotBackend:
    """Scripted snapshot backend: returns a fixed assemblage independent of the
    input solids — lets tests drive growth AND removal deterministically."""
    backend_id = "stoichiometric"  # reuse stoich channels
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "snapshot"

    def __init__(self, ch_mol: float, csh_mol: float = 0.0):
        self.ch_mol = ch_mol
        self.csh_mol = csh_mol
        self.calls = []

    def react(self, released_mol, water_available_mol, inventory,
              solid_elements=None):
        self.calls.append({"released": dict(released_mol),
                           "solid_elements": None if solid_elements is None
                           else np.asarray(solid_elements).copy()})
        parcels = []
        # element bookkeeping mirrors the input exactly: whatever elements came
        # in (released + solids + inventory) minus the assemblage goes back to
        # the residual, so the engine ledger closes.
        e_in = np.asarray(inventory, dtype=float).copy()
        for pid, mol in released_mol.items():
            e_in += element_vector(REG.get(pid).formula, mol)
        if solid_elements is not None:
            e_in += np.asarray(solid_elements)
        for pid, mol in (("CH", self.ch_mol), ("CSH", self.csh_mol)):
            if mol <= 0.0:
                continue
            entry = REG.get(pid)
            vec = element_vector(entry.formula, mol)
            parcels.append(Parcel(phase_id=pid, mol=mol, elements=vec,
                                  skel_vol_cm3=mol * entry.skeleton_molar_volume_cm3))
            e_in = e_in - vec
        # NET water consumption (like the real snapshot backend): water bound
        # into the NEW assemblage minus water already bound in the owned solids
        # that were fed back — negative on shrink (bound water returns free).
        h_idx = ELEMENT_IDS.index("H")
        h_new = sum(float(pc.elements[h_idx]) for pc in parcels)
        h_owned = (0.0 if solid_elements is None
                   else float(np.asarray(solid_elements)[h_idx]))
        water_used = (h_new - h_owned) / 2.0
        # the net water's H/O moves between the FREE pool (engine ledger) and
        # the solids; credit it here so the residual closes exactly
        e_in = e_in + element_vector(REG.get("H2O").formula, water_used)
        return ReactionResult(status=STATUS_OK, parcels=parcels,
                              water_consumed_mol=water_used,
                              residual_inventory=e_in)


def _short_cfg(**over):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.05]}}}
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    raw.update(over)
    return TinnConfig.model_validate(raw)


# ---------------- morphology.remove ----------------

def test_morphology_remove_exact_and_overrequest():
    hyd = np.zeros((4, 8, 8, 8))
    hyd[0, 2, 2, 2] = 0.3
    hyd[0, 2, 2, 3] = 0.1
    member = np.zeros((8, 8, 8), dtype=bool)
    member[2, 2, :] = True
    field, removed = morphology.remove(hyd, 0, 0.2, member)
    assert removed == 0.2  # exact, not approximately
    assert float(field.sum()) == 0.2
    assert hyd[0].sum() == pytest.approx(0.2)
    assert hyd[0].min() >= 0.0
    with pytest.raises(ValueError, match="exceeds available"):
        morphology.remove(hyd, 0, 5.0, member)


def test_place_overflow_beyond_shell_radius():
    """Overflow pass (PRD 2.2 rev.2): all capacity sits FAR outside the
    3-shell radius of the source — the remainder must precipitate
    cluster-wide, not reject."""
    hyd2 = np.zeros((4, 8, 8, 8))
    liq = np.zeros((8, 8, 8))
    liq[4, 4, 4] = 1.0                      # Manhattan distance 12 from source
    vac = np.zeros((8, 8, 8))
    dem = np.zeros((4, 8, 8, 8))
    dem[1, 0, 0, 0] = 0.5
    clu = np.zeros((8, 8, 8), dtype=np.int64)
    out = morphology.place(hyd2, liq, vac, dem, clu, clu)
    assert out.status == morphology.STATUS_OK
    assert out.placed_vol_vox == pytest.approx(0.5)
    assert hyd2[1, 4, 4, 4] == pytest.approx(0.5)
    assert liq[4, 4, 4] == pytest.approx(0.5)
    # cluster-wide shortfall is still an honest capacity reject
    dem[1, 0, 0, 0] = 2.0
    out2 = morphology.place(hyd2, liq, vac, dem, clu, clu)
    assert out2.status == morphology.STATUS_CAPACITY
    assert out2.unplaced_vol_vox == pytest.approx(2.0)


# ---------------- snapshot engine mechanics (no xgems needed) ----------------

def _grow_then_shrink(cfg, grow_ch=3e-12, shrink_ch=1e-12):
    """Step 1 grows CH; step 2's snapshot is SMALLER -> removal must happen."""
    eng = Engine(cfg, reaction_backend=FakeSnapshotBackend(grow_ch))
    state = eng.initial_state()
    t1, rej, _ = eng.try_step(state, 2.0)
    assert rej is None
    eng2 = Engine(cfg, reaction_backend=FakeSnapshotBackend(shrink_ch))
    t2, rej2, m2 = eng2.try_step(t1, 2.0)
    return t1, t2, rej2, m2


def test_snapshot_step_replaces_ledgers_and_closes():
    cfg = _short_cfg()
    t1, t2, rej2, m2 = _grow_then_shrink(cfg)
    assert rej2 is None
    ch_i = HYDRATE_PHASE_IDS.index("CH")
    # after step 2 the holdings equal the NEW snapshot (wet share), not the sum
    assert t2.hydrate_mol[ch_i] < t1.hydrate_mol[ch_i]
    rep = ledger.check_all(t2, REG)
    assert rep.ok, rep.violations


def test_space_filling_limit_freezes_cluster():
    """Space-filling limit (PRD 4.5): an assemblage wanting MORE envelope than
    the cluster's entire pore space is dt-independent — the step must accept
    with the cluster frozen (release back as unmet), never abort the run."""
    cfg = _short_cfg()
    state = Engine(cfg, reaction_backend=FakeSnapshotBackend(1e-30)).initial_state()
    liquid_cm3 = float(state.capillary_liquid.sum()) * state.vox_cm3
    big = 1.10 * liquid_cm3 / REG.get("CH").skeleton_molar_volume_cm3
    t3, rej3, _ = Engine(cfg, reaction_backend=FakeSnapshotBackend(big)
                         ).try_step(state, 2.0)
    assert rej3 is None
    assert float(t3.hydrate_fraction.sum()) == 0.0   # nothing was placed
    assert float(t3.unmet_mol.sum()) > 0.0           # release returned as unmet
    rep3 = ledger.check_all(t3, REG)
    assert rep3.ok, rep3.violations


def test_ch_redissolution_frees_volume():
    cfg = _short_cfg()
    t1, t2, rej2, _ = _grow_then_shrink(cfg)
    assert rej2 is None
    # dense hydrate volume shrank and the freed space is now liquid/gas
    assert t2.hydrate_fraction.sum() < t1.hydrate_fraction.sum()
    total = (t2.anhydrous_fraction.sum(axis=0) + t2.hydrate_fraction.sum(axis=0)
             + t2.capillary_liquid + t2.capillary_gas)
    assert np.max(np.abs(total - 1.0)) <= 1e-12


def test_bound_water_returns_on_shrink():
    cfg = _short_cfg()
    t1, t2, rej2, _ = _grow_then_shrink(cfg)
    assert rej2 is None
    # water partition invariant holds with signed moves
    for st in (t1, t2):
        s = st.water_free_mol + st.water_gel_mol + st.water_bound_mol
        assert s == pytest.approx(st.initial_water_mol, rel=1e-12)
    assert t2.water_bound_mol < t1.water_bound_mol  # bound water RETURNED


def test_total_dryout_degrades_to_unmet():
    """rev.2 review finding: with gel-conduit dissolution, a fully dried RVE
    (no liquid cluster anywhere) used to reject cluster_dryout forever — the
    shortfall is dt-independent, so the run aborted. Unreachable sites must
    give their target back as honest unmet and the step must ACCEPT."""
    cfg = _short_cfg()
    eng = Engine(cfg, reaction_backend=FakeSnapshotBackend(3e-12))
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    # surgical gasification: every drop of capillary liquid becomes gas, the
    # water and element ledgers are re-anchored consistently
    t1.capillary_gas += t1.capillary_liquid
    t1.capillary_liquid[:] = 0.0
    t1.initial_water_mol -= t1.water_free_mol
    t1.initial_elements = (t1.initial_elements
                           - t1.cluster_inventory.sum(axis=0)
                           - element_vector(REG.get("H2O").formula,
                                            t1.water_free_mol))
    t1.water_free_mol = 0.0
    t1.cluster_inventory = np.zeros((0, len(ELEMENT_IDS)))
    t1.cluster_endmember_mol = np.zeros((0, t1.cluster_endmember_mol.shape[1]))
    t2, rej2, _ = eng.try_step(t1, 2.0)   # kinetics still demand dn > 0
    assert rej2 is None, rej2 and rej2.reason
    assert float(t2.unmet_mol.sum()) > float(t1.unmet_mol.sum())
    # frozen everything: assemblage untouched, no phantom dissolution
    assert np.array_equal(t2.hydrate_fraction, t1.hydrate_fraction)
    rep = ledger.check_all(t2, REG)
    assert rep.ok, rep.violations


def test_saturated_gasfree_pocket_freezes_not_aborts():
    """rev.2 review finding 5, REPRODUCED then fixed: in a gas-free saturated
    cluster, snapshot growth displaces liquid that chemistry does not consume;
    the reconciliation deficit has no gas space to land in and balance_water
    rejected at EVERY dt (snapshot deltas are not dt-scaled) — an abort. The
    freeze gate now includes the water-relocation limit."""
    cfg = _short_cfg()
    eng = Engine(cfg, reaction_backend=FakeSnapshotBackend(1e-10))
    state = eng.initial_state()
    assert float(state.capillary_gas.sum()) == 0.0
    t1, rej, _ = eng.try_step(state, 2.0)
    assert rej is None, rej and rej.reason        # frozen, not rejected
    assert float(t1.hydrate_fraction.sum()) == 0.0
    assert float(t1.unmet_mol.sum()) > 0.0
    rep = ledger.check_all(t1, REG)
    assert rep.ok, rep.violations
    # control: the same growth with gas headroom proceeds and places
    state2 = eng.initial_state()
    move = 0.2 * state2.capillary_liquid
    state2.capillary_gas += move
    state2.capillary_liquid -= move
    moved_mol = float(move.sum()) / state2.vm_vox(REG, "H2O")
    state2.initial_water_mol -= moved_mol
    state2.water_free_mol -= moved_mol
    state2.initial_elements = state2.initial_elements - element_vector(
        REG.get("H2O").formula, moved_mol)
    t2, rej2, _ = eng.try_step(state2, 2.0)
    assert rej2 is None, rej2 and rej2.reason
    assert float(t2.hydrate_fraction.sum()) > 0.0
    rep2 = ledger.check_all(t2, REG)
    assert rep2.ok, rep2.violations


def test_place_overflow_dust_and_negative_liquid():
    """rev.2 review findings on the overflow pass: negative-dust liquid must
    not enter the capacity pool (no negative hydrate takes), and a near-exact
    fit within the dust tolerance places instead of rejecting."""
    n = 8
    hyd = np.zeros((4, n, n, n))
    liq = np.zeros((n, n, n))
    vac = np.zeros((n, n, n))
    clu = np.zeros((n, n, n), dtype=np.int64)
    liq[4, 4, 4] = 0.5                 # the only real capacity, far from source
    liq[0, 0, 1] = -1e-16              # placement dust from an earlier source
    dem = np.zeros((4, n, n, n))
    dem[1, 0, 0, 0] = 0.5 * (1.0 + 5e-13)   # within the 1e-12 fit tolerance
    out = morphology.place(hyd, liq, vac, dem, clu, clu)
    assert out.status == morphology.STATUS_OK
    assert hyd.min() >= 0.0                          # no negative takes
    assert hyd[1, 0, 0, 1] == 0.0                    # dust voxel excluded
    assert out.placed_vol_vox == pytest.approx(0.5, rel=1e-11)
    # beyond the tolerance: still an honest capacity reject
    dem[1, 0, 0, 0] = 0.6
    out2 = morphology.place(hyd, liq, vac, dem, clu, clu)
    assert out2.status == morphology.STATUS_CAPACITY


def test_snapshot_backend_receives_owned_solids():
    cfg = _short_cfg()
    b1 = FakeSnapshotBackend(3e-12)
    eng = Engine(cfg, reaction_backend=b1)
    state = eng.initial_state()
    t1, _, _ = eng.try_step(state, 2.0)
    b2 = FakeSnapshotBackend(3e-12)
    eng2 = Engine(cfg, reaction_backend=b2)
    eng2.try_step(t1, 2.0)
    # first step: no solids yet -> solid_elements None or zeros; second: CH owned
    assert all(c["solid_elements"] is None or not np.any(c["solid_elements"])
               for c in b1.calls)
    assert any(c["solid_elements"] is not None and np.any(c["solid_elements"])
               for c in b2.calls)


def test_ownership_partition_conserves_exactly():
    # synthetic: split a channel pool by recon shares + dry remainder == total
    rng = np.random.default_rng(3)
    n = 8
    hyd = rng.random((n, n, n))
    recon = rng.integers(-1, 3, size=(n, n, n))
    pool = rng.random(len(ELEMENT_IDS))
    tot = float(hyd.sum())
    b = np.bincount((recon + 1).ravel(), weights=hyd.ravel(), minlength=4)
    shares = b[1:] / tot
    owned = shares[:, None] * pool
    dry_share = b[0] / tot
    recon_sum = owned.sum(axis=0) + dry_share * pool
    assert recon_sum == pytest.approx(pool, rel=1e-12)


def test_snapshot_rollback_invariance():
    cfg = _short_cfg()
    eng = Engine(cfg, reaction_backend=FakeSnapshotBackend(3e-12))
    state = eng.initial_state()
    t1, _, _ = eng.try_step(state, 2.0)

    class Exploding(FakeSnapshotBackend):
        def react(self, *a, **k):
            from tinn.backend import BackendTransientError
            raise BackendTransientError("boom")

    before = t1.full_hash()
    eng2 = Engine(cfg, reaction_backend=Exploding(0.0))
    trial, rej, _ = eng2.try_step(t1, 2.0)
    assert trial is None and rej is not None
    assert t1.full_hash() == before


def test_placement_balance_gross_removal_leg():
    p = ledger.PlacementBalance(1.0, 1.0, 1.0,
                                backend_removal_vol_vox=0.5,
                                removed_vol_vox=0.2)
    st_cfg = _short_cfg()
    eng = Engine(st_cfg, reaction_backend=FakeSnapshotBackend(3e-12))
    state = eng.initial_state()
    rep = ledger.check_all(state, REG, p)
    assert "balance_placement" in rep.violations


def test_checkpoint_v2_roundtrip(tmp_path):
    cfg = _short_cfg()
    eng = Engine(cfg, reaction_backend=FakeSnapshotBackend(3e-12))
    state = eng.initial_state()
    t1, _, _ = eng.try_step(state, 2.0)
    t1.accept_count = 1
    save_checkpoint(t1, str(tmp_path), "ck")
    loaded = load_checkpoint(str(tmp_path / "ck"), REG)
    assert loaded.full_hash() == t1.full_hash()
    assert loaded.hydrate_elements_ch.shape == t1.hydrate_elements_ch.shape
    # format-1 header is rejected loudly
    hdr = tmp_path / "ck" / "header.json"
    data = json.loads(hdr.read_text(encoding="utf-8"))
    data["format_version"] = 1
    hdr.write_text(json.dumps(data), encoding="utf-8")
    man = tmp_path / "ck" / "manifest.json"
    import hashlib
    manifest = json.loads(man.read_text(encoding="utf-8"))
    manifest["header.json"] = hashlib.sha256(hdr.read_bytes()).hexdigest()
    man.write_text(json.dumps(manifest), encoding="utf-8")
    manifest["manifest.json"] = None
    with pytest.raises(StorageError, match="format 1"):
        load_checkpoint(str(tmp_path / "ck"), REG)


# ---------------- real GEMS snapshot path ----------------

def _gems_cfg(**over):
    raw = json.loads((EXAMPLES / "opc_gems_32.json").read_text(encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw.update(over)
    return TinnConfig.model_validate(raw)


@needs_gems
def test_gems_snapshot_step_closure():
    cfg = _gems_cfg(schedule={"output_times_h": [2.0], "dt_initial_h": 2.0,
                              "dt_min_h": 0.001})
    eng = Engine(cfg)
    assert eng.backend.mode == "snapshot"
    state = eng.initial_state()
    t1, rej, m = eng.try_step(state, 2.0)
    assert rej is None
    t2, rej2, m2 = eng.try_step(t1, 2.0)   # second step re-equilibrates solids
    assert rej2 is None
    assert m2["max_element_err_mol"] <= 1e-20
    assert m2["injected_max_rel"] <= 1e-6
    rep = ledger.check_all(t2, REG)
    assert rep.ok, rep.violations


@needs_gems
def test_persistent_worker_matches_spawn(tmp_path):
    from tinn.gems import GemsWorker
    n_c3s = 1.0 / 228.3145
    n_w = 0.5 / 18.015
    elements = {"Ca": 3 * n_c3s, "Si": n_c3s,
                "O": 5 * n_c3s + n_w + 1e-7, "H": 2 * n_w}
    a = GemsWorker(str(BUNDLE), python_executable=str(GEMS_PYTHON),
                   work_root=str(tmp_path / "a"), persistent=True)
    b = GemsWorker(str(BUNDLE), python_executable=str(GEMS_PYTHON),
                   work_root=str(tmp_path / "b"), persistent=False)
    ra = a.equilibrate_elements(elements, 293.15)
    ra2 = a.equilibrate_elements(elements, 293.15)  # same server, second call
    rb = b.equilibrate_elements(elements, 293.15)
    a.close()
    assert repr(ra.ph) == repr(rb.ph) == repr(ra2.ph)
    assert ra.phase_masses_kg == rb.phase_masses_kg
