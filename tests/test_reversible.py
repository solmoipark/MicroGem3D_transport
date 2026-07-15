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
    # space-filling limit: an assemblage wanting MORE envelope than the
    # cluster's entire pore space is dt-independent — the step must accept
    # with the cluster frozen (release back as unmet), never abort the run
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
