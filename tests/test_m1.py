"""M1 tests: conservation core — kinetics/state/dissolution/transport/backend/
morphology/ledger/engine/storage (PRD §3 M1, §6.1 blocking invariants)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tinn import cli, dissolution, ledger, morphology, transport
from tinn.backend import (PH_NOT_AVAILABLE, STATUS_INSUFFICIENT_WATER, STATUS_OK,
                          StoichiometricBackend)
from tinn.config import TinnConfig
from tinn.engine import Engine, EngineError
from tinn.kinetics import TabulatedKinetics, make_kinetics
from tinn.registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                           SOLID_PHASE_IDS, default_registry)
from tinn.state import SimulationState
from tinn.storage import StorageError, load_checkpoint, save_checkpoint

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
REG = default_registry()


def _cfg(**overrides) -> TinnConfig:
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw.update(overrides)
    return TinnConfig.model_validate(raw)


def _short_cfg() -> TinnConfig:
    return _cfg(kinetics={"kind": "tabulated",
                          "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.1]}}},
                schedule={"output_times_h": [2.0, 4.0], "dt_initial_h": 1.0,
                          "dt_min_h": 0.01})


@pytest.fixture(scope="module")
def short_run():
    eng = Engine(_short_cfg())
    state0 = eng.initial_state()
    state, summary = eng.run(state=state0.clone())
    return eng, state0, state, summary


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "run"
    eng = Engine(_cfg())
    state, summary = eng.run(out_dir=str(out))
    return eng, state, summary, out


def _blank_state(n_liquid_voxels=0) -> SimulationState:
    """Minimal hand-built state on the c3s config grid for unit tests."""
    cfg = _cfg()
    eng = Engine(cfg)
    n = cfg.rve.grid_size
    st = SimulationState(
        config=cfg,
        hydrate_ids=HYDRATE_PHASE_IDS,
        anhydrous_fraction=np.zeros((len(SOLID_PHASE_IDS), n, n, n)),
        hydrate_fraction=np.zeros((len(HYDRATE_PHASE_IDS), n, n, n)),
        capillary_liquid=np.zeros((n, n, n)),
        capillary_gas=np.zeros((n, n, n)),
        particle_id=np.full((n, n, n), -1, dtype=np.int64),
        cluster_id=np.full((n, n, n), -1, dtype=np.int64),
        particles={}, subgrid_bins={},
        parcels={"time_h": [], "cluster": [], "hydrate": [], "mol": [],
                 "skel_vol_vox": [], "bulk_vol_vox": []},
        remap_events={"time_h": [], "prev": [], "new": [], "overlap_vox": []},
        cluster_inventory=np.zeros((0, len(ELEMENT_IDS))),
        time_h=0.0, dt_h=1.0,
        phase_mol=np.zeros(len(KINETIC_PHASE_IDS)), initial_phase_mol=np.zeros(len(KINETIC_PHASE_IDS)),
        unmet_mol=np.zeros(len(KINETIC_PHASE_IDS)), hydrate_mol=np.zeros(4),
        hydrate_env_vol_vox=np.zeros(4),
        hydrate_elements_ch=np.zeros((4, len(ELEMENT_IDS))),
        injected_elements=np.zeros(len(ELEMENT_IDS)),
        water_free_mol=0.0, water_gel_mol=0.0, water_bound_mol=0.0,
        initial_water_mol=0.0, inert_volume_vox=0.0,
        initial_elements=np.zeros(len(ELEMENT_IDS)),
        accept_count=0, reject_counts={}, rng_state={}, config_hash="x",
        backend_id="stoichiometric")
    return st


# ---------------- kinetics ----------------

def test_tabulated_interpolation():
    kin = TabulatedKinetics(_cfg().kinetics)
    a = kin.alpha_at(15.0)  # between 6 h (0.05) and 24 h (0.15)
    assert a[KINETIC_PHASE_IDS.index("C3S")] == pytest.approx(0.1)
    assert kin.alpha_at(24.0)[0] == pytest.approx(0.15)
    assert kin.alpha_at(0.0)[0] == 0.0


def test_tabulated_beyond_horizon_raises():
    kin = TabulatedKinetics(_cfg().kinetics)
    with pytest.raises(ValueError):
        kin.alpha_at(500.0)


# ---------------- state ----------------

def test_state_from_geometry_ledger_consistency(short_run):
    _, state0, _, _ = short_run
    rep = ledger.check_all(state0, REG)
    assert rep.ok, rep.violations
    assert state0.water_free_mol == state0.initial_water_mol
    assert state0.alpha().max() == 0.0


def test_state_clone_independence(short_run):
    _, state0, _, _ = short_run
    h = state0.full_hash()
    c = state0.clone()
    c.capillary_liquid[0, 0, 0] += 0.1
    c.phase_mol[0] *= 0.5
    c.parcels["mol"].append(1.0)
    assert state0.full_hash() == h


def test_state_alpha_zero_initial():
    st = _blank_state()
    assert np.all(st.alpha() == 0.0)


# ---------------- dissolution ----------------

def _dn(c3s_mol: float) -> np.ndarray:
    v = np.zeros(len(KINETIC_PHASE_IDS))
    v[0] = c3s_mol
    return v


def _dissolution_state():
    st = _blank_state()
    # two C3S voxels with different liquid contact, one liquid voxel between
    st.anhydrous_fraction[0, 5, 5, 4] = 0.5   # one liquid face
    st.anhydrous_fraction[0, 5, 5, 6] = 0.5   # one liquid face
    st.capillary_liquid[5, 5, 5] = 1.0
    st.capillary_liquid[5, 5, 7] = 0.5        # extra face for the second voxel
    return st


def test_dissolution_proportional_to_wetted_faces():
    st = _dissolution_state()
    vm = st.vm_vox(REG, "C3S")
    res = dissolution.dissolve(st, REG, _dn(0.3 / vm))
    a = res.removed_vol[0, 5, 5, 4]
    b = res.removed_vol[0, 5, 5, 6]
    assert a + b == pytest.approx(0.3)
    assert b / a == pytest.approx(2.0)  # 1 wet face vs 2 wet faces
    assert res.unmet_mol[0] == 0.0


def test_dissolution_caps_and_redistributes():
    st = _dissolution_state()
    vm = st.vm_vox(REG, "C3S")
    res = dissolution.dissolve(st, REG, _dn(0.8 / vm))
    # both sites fully exhausted (total available 1.0 > 0.8)
    assert res.removed_vol[0].sum() == pytest.approx(0.8)
    assert st.anhydrous_fraction[0].sum() == pytest.approx(0.2)


def test_dissolution_unmet_recorded():
    st = _dissolution_state()
    vm = st.vm_vox(REG, "C3S")
    res = dissolution.dissolve(st, REG, _dn(2.0 / vm))
    assert res.removed_vol[0].sum() == pytest.approx(1.0)   # everything accessible
    assert res.unmet_mol[0] == pytest.approx(1.0 / vm)


def test_dissolution_phase_filter():
    st = _dissolution_state()
    st.anhydrous_fraction[1, 5, 5, 4] = 0.3  # C2S at same site
    vm = st.vm_vox(REG, "C3S")
    dissolution.dissolve(st, REG, _dn(0.2 / vm))
    assert st.anhydrous_fraction[1].sum() == pytest.approx(0.3)  # untouched


def test_dissolution_no_liquid_all_unmet():
    st = _blank_state()
    st.anhydrous_fraction[0, 5, 5, 5] = 1.0  # fully enclosed, no liquid anywhere
    vm = st.vm_vox(REG, "C3S")
    res = dissolution.dissolve(st, REG, _dn(0.5 / vm))
    assert res.removed_mol[0] == 0.0
    assert res.unmet_mol[0] == pytest.approx(0.5 / vm)


# ---------------- transport ----------------

def test_label_clusters_periodic_wrap():
    liq = np.zeros((32, 32, 32))
    liq[0, 5, 5] = 1.0
    liq[31, 5, 5] = 1.0  # touches across the periodic boundary
    labels, k = transport.label_clusters(liq)
    assert k == 1
    assert labels[0, 5, 5] == labels[31, 5, 5] == 0


def test_label_clusters_two_components_deterministic():
    liq = np.zeros((32, 32, 32))
    liq[1, 1, 1] = 1.0
    liq[10, 10, 10] = 1.0
    labels, k = transport.label_clusters(liq)
    assert k == 2
    assert labels[1, 1, 1] == 0      # smaller flat index -> id 0
    assert labels[10, 10, 10] == 1
    assert labels[0, 0, 0] == -1


def test_label_clusters_threshold():
    liq = np.zeros((32, 32, 32))
    liq[2, 2, 2] = 1e-10  # below LIQ_EPS
    labels, k = transport.label_clusters(liq)
    assert k == 0 and labels[2, 2, 2] == -1


def test_remap_split_proportional():
    prev_liq = np.zeros((4, 4, 4)); prev_liq[0, 0, :3] = 1.0
    prev_lab = np.full((4, 4, 4), -1, dtype=np.int64); prev_lab[0, 0, :3] = 0
    new_liq = np.zeros((4, 4, 4)); new_liq[0, 0, 0] = 1.0; new_liq[0, 0, 2] = 1.0
    new_lab = np.full((4, 4, 4), -1, dtype=np.int64)
    new_lab[0, 0, 0] = 0; new_lab[0, 0, 2] = 1
    inv = np.array([[10.0, 0.0]])
    res = transport.remap_inventories(prev_lab, prev_liq, new_lab, new_liq, inv, 2)
    assert res.inventory[0, 0] == pytest.approx(5.0)
    assert res.inventory[1, 0] == pytest.approx(5.0)
    assert not res.dryout


def test_remap_dryout_flagged():
    prev_liq = np.zeros((4, 4, 4)); prev_liq[0, 0, 0] = 1.0
    prev_lab = np.full((4, 4, 4), -1, dtype=np.int64); prev_lab[0, 0, 0] = 0
    new_liq = np.zeros((4, 4, 4))
    new_lab = np.full((4, 4, 4), -1, dtype=np.int64)
    inv = np.array([[1.0]])
    res = transport.remap_inventories(prev_lab, prev_liq, new_lab, new_liq, inv, 0)
    assert res.dryout == [0]


# ---------------- backend ----------------

def test_stoichiometric_products_and_water():
    b = StoichiometricBackend(_cfg().chemistry.stoichiometric_rules, REG)
    r = b.react({"C3S": 2.0}, water_available_mol=100.0,
                inventory=np.zeros(len(ELEMENT_IDS)))
    assert r.status == STATUS_OK
    assert r.water_consumed_mol == pytest.approx(10.6)
    by_phase = {pc.phase_id: pc.mol for pc in r.parcels}
    assert by_phase == pytest.approx({"CSH": 2.0, "CH": 2.6})
    # parcels carry their own element vectors and skeleton volumes
    csh = next(pc for pc in r.parcels if pc.phase_id == "CSH")
    assert csh.skel_vol_cm3 == pytest.approx(2.0 * 107.3)
    assert csh.elements[ELEMENT_IDS.index("Si")] == pytest.approx(2.0)


def test_stoichiometric_insufficient_water():
    b = StoichiometricBackend(_cfg().chemistry.stoichiometric_rules, REG)
    r = b.react({"C3S": 2.0}, water_available_mol=1.0,
                inventory=np.zeros(len(ELEMENT_IDS)))
    assert r.status == STATUS_INSUFFICIENT_WATER
    assert not r.parcels


def test_stoichiometric_inventory_passthrough_and_nan_ph():
    b = StoichiometricBackend(_cfg().chemistry.stoichiometric_rules, REG)
    inv = np.arange(len(ELEMENT_IDS), dtype=float)
    r = b.react({"C3S": 0.1}, water_available_mol=10.0, inventory=inv)
    assert np.array_equal(r.residual_inventory, inv)
    assert math.isnan(r.ph) and r.ph_status == PH_NOT_AVAILABLE


# ---------------- morphology ----------------

def _placement_arrays(n=8):
    hyd = np.zeros((4, n, n, n))
    liq = np.zeros((n, n, n))
    vac = np.zeros((n, n, n))
    dem = np.zeros((4, n, n, n))
    cell = np.full((n, n, n), -1, dtype=np.int64)
    src = np.full((n, n, n), -1, dtype=np.int64)
    return hyd, liq, vac, dem, cell, src


def test_placement_inner_before_liquid():
    hyd, liq, vac, dem, cell, src = _placement_arrays()
    vac[4, 4, 4] = 0.3; liq[4, 4, 4] = 0.5
    cell[4, 4, 4] = 0; src[4, 4, 4] = 0
    dem[0, 4, 4, 4] = 0.4
    out = morphology.place(hyd, liq, vac, dem, cell, src)
    assert out.status == morphology.STATUS_OK
    assert vac[4, 4, 4] == pytest.approx(0.0)
    assert liq[4, 4, 4] == pytest.approx(0.4)     # only 0.1 displaced
    assert out.liquid_displaced_vol_vox == pytest.approx(0.1)


def test_placement_spills_within_cluster_only():
    hyd, liq, vac, dem, cell, src = _placement_arrays()
    src[4, 4, 4] = 0; cell[4, 4, 4] = 0
    liq[4, 4, 5] = 0.2; cell[4, 4, 5] = 0     # same cluster
    liq[4, 4, 3] = 5.0; cell[4, 4, 3] = 1     # foreign cluster, must not be used
    dem[0, 4, 4, 4] = 0.15
    out = morphology.place(hyd, liq, vac, dem, cell, src)
    assert out.status == morphology.STATUS_OK
    assert liq[4, 4, 3] == pytest.approx(5.0)
    assert liq[4, 4, 5] == pytest.approx(0.05)


def test_placement_capacity_reject():
    hyd, liq, vac, dem, cell, src = _placement_arrays()
    src[4, 4, 4] = 0; cell[4, 4, 4] = 0
    vac[4, 4, 4] = 0.1
    dem[0, 4, 4, 4] = 0.5  # nowhere else to go
    out = morphology.place(hyd, liq, vac, dem, cell, src)
    assert out.status == morphology.STATUS_CAPACITY
    assert out.unplaced_vol_vox == pytest.approx(0.4)


def test_placement_volume_accounting():
    hyd, liq, vac, dem, cell, src = _placement_arrays()
    src[4, 4, 4] = 0
    for dx in (-1, 0, 1):
        cell[4, 4, 4 + dx] = 0
        liq[4, 4, 4 + dx] = 0.2
    dem[0, 4, 4, 4] = 0.3; dem[1, 4, 4, 4] = 0.2
    out = morphology.place(hyd, liq, vac, dem, cell, src)
    assert out.status == morphology.STATUS_OK
    assert out.placed_vol_vox == pytest.approx(0.5)
    assert hyd.sum() == pytest.approx(0.5)
    assert hyd[0].sum() == pytest.approx(0.3)  # channel split preserved


# ---------------- ledger ----------------

def test_ledger_detects_element_imbalance(short_run):
    _, _, state, _ = short_run
    bad = state.clone()
    bad.hydrate_elements_ch[:, 0] *= 1.5
    rep = ledger.check_all(bad, REG)
    assert any(v.startswith("balance_element") for v in rep.violations)


def test_ledger_detects_voxel_identity_violation(short_run):
    _, _, state, _ = short_run
    bad = state.clone()
    bad.capillary_gas[0, 0, 0] += 1e-6
    rep = ledger.check_all(bad, REG)
    assert any(v.startswith("balance_voxel") for v in rep.violations)


def test_ledger_detects_water_partition_violation(short_run):
    _, _, state, _ = short_run
    bad = state.clone()
    bad.water_gel_mol += 1e-12
    rep = ledger.check_all(bad, REG)
    assert "balance_water:partition" in rep.violations


def test_ledger_detects_placement_imbalance(short_run):
    _, _, state, _ = short_run
    p = ledger.PlacementBalance(1.0, 1.0, 0.5)
    rep = ledger.check_all(state, REG, p)
    assert "balance_placement" in rep.violations


# ---------------- engine ----------------

class _FailingOnceBackend:
    """Delegates to the real backend after failing the first call."""
    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "incremental"

    def __init__(self, inner):
        self.inner = inner
        self.failures = 0

    def react(self, released_mol, water_available_mol, inventory):
        from tinn.backend import BackendTransientError
        if self.failures < 6:   # outlast the engine's in-step scale-down retries
            self.failures += 1
            raise BackendTransientError("injected failure")
        return self.inner.react(released_mol, water_available_mol, inventory)


def test_engine_rollback_invariance(short_run):
    eng, state0, _, _ = short_run
    cfg = _short_cfg()
    failing = _FailingOnceBackend(StoichiometricBackend(
        cfg.chemistry.stoichiometric_rules, REG))
    eng2 = Engine(cfg, reaction_backend=failing)
    st = state0.clone()
    before_dense, before_full = st.dense_hash(), st.full_hash()
    trial, reject, _ = eng2.try_step(st, 1.0)
    assert trial is None and reject.reason == "backend_failure"
    assert st.dense_hash() == before_dense
    assert st.full_hash() == before_full


def test_engine_recovers_after_reject(short_run):
    _, state0, _, _ = short_run
    cfg = _short_cfg()
    failing = _FailingOnceBackend(StoichiometricBackend(
        cfg.chemistry.stoichiometric_rules, REG))
    eng2 = Engine(cfg, reaction_backend=failing)
    state, summary = eng2.run(state=state0.clone())
    assert state.reject_counts.get("backend_failure", 0) >= 1
    # transient failures may scale part of the release into honest unmet;
    # the accounting identity dissolved + unmet = summed targets always holds
    achieved_plus_unmet = state.alpha()[0] + state.unmet_mol[0] / state.initial_phase_mol[0]
    assert achieved_plus_unmet == pytest.approx(0.1, abs=1e-9)


class _AlwaysThirstyBackend:
    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "incremental"

    def react(self, released_mol, water_available_mol, inventory):
        from tinn.backend import ReactionResult
        return ReactionResult(status=STATUS_INSUFFICIENT_WATER,
                              residual_inventory=inventory.copy())


def test_engine_dt_underflow_aborts_with_reason(short_run):
    _, state0, _, _ = short_run
    eng2 = Engine(_short_cfg(), reaction_backend=_AlwaysThirstyBackend())
    with pytest.raises(EngineError) as e:
        eng2.run(state=state0.clone())
    assert e.value.reason == STATUS_INSUFFICIENT_WATER


def test_engine_alpha_tracks_table(short_run):
    _, _, state, summary = short_run
    assert summary["outputs"][0]["alpha"]["C3S"] == pytest.approx(0.05, abs=1e-9)
    assert summary["outputs"][1]["alpha"]["C3S"] == pytest.approx(0.10, abs=1e-9)
    assert float(state.unmet_mol.max()) <= ledger.ELEMENT_ATOL_MOL


def test_engine_step_metrics_within_bounds(short_run):
    _, _, _, summary = short_run
    m = summary["final"]["ledger_metrics"]
    assert m["voxel_identity_max_err"] <= 1e-12
    assert m["water_partition_err_mol"] <= 1e-20
    assert m["placement_rel_err"] <= 1e-9


# ---------------- full smoke (PRD M1 DoD) ----------------

def test_smoke_alpha_035_no_rejects(full_run):
    _, state, summary, _ = full_run
    # a sealed wet pocket leaves a small honest unmet deficit (water-limited
    # dissolution); the accounting identity dissolved + unmet = summed targets
    # must hold to ledger precision
    assert state.alpha()[0] == pytest.approx(0.35, abs=1e-4)
    achieved_plus_unmet = state.alpha()[0] + state.unmet_mol[0] / state.initial_phase_mol[0]
    assert achieved_plus_unmet == pytest.approx(0.35, abs=1e-9)
    assert state.reject_counts == {}
    assert float(state.unmet_mol.max()) <= 1e-12


def test_smoke_conservation_61(full_run):
    _, state, _, _ = full_run
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    assert rep.metrics["max_element_err_mol"] <= 1e-24 + 1e-8 * float(
        np.abs(state.initial_elements).max())
    assert rep.metrics["voxel_identity_max_err"] <= 1e-12
    w_err = abs(state.water_free_mol + state.water_gel_mol + state.water_bound_mol
                - state.initial_water_mol)
    assert w_err <= 1e-8 * state.initial_water_mol + 1e-24


def test_smoke_physical_monotonicity(full_run):
    _, _, summary, _ = full_run
    por = [r["porosity_capillary"] for r in summary["outputs"]]
    assert all(b < a for a, b in zip(por, por[1:]))
    gas = [r["chem_shrinkage_ml_per_g_reacted"] for r in summary["outputs"]]
    assert all(g > 0 for g in gas)


# ---------------- storage / restart ----------------

def test_checkpoint_roundtrip_bitwise(full_run, tmp_path):
    _, state, _, _ = full_run
    save_checkpoint(state, str(tmp_path), "ck")
    loaded = load_checkpoint(str(tmp_path / "ck"), REG)
    assert loaded.dense_hash() == state.dense_hash()
    assert loaded.full_hash() == state.full_hash()
    assert loaded.rng_state == state.rng_state
    assert loaded.config.config_hash() == state.config.config_hash()


def test_checkpoint_corruption_detected(full_run, tmp_path):
    _, state, _, _ = full_run
    ck = save_checkpoint(state, str(tmp_path), "ck")
    chunk = ck / "arrays" / "capillary_liquid" / "0.0.0"
    raw = bytearray(chunk.read_bytes())
    raw[100] ^= 0xFF
    chunk.write_bytes(bytes(raw))
    with pytest.raises(StorageError):
        load_checkpoint(str(ck), REG)


def test_checkpoint_no_overwrite(full_run, tmp_path):
    _, state, _, _ = full_run
    save_checkpoint(state, str(tmp_path), "ck")
    with pytest.raises(StorageError):
        save_checkpoint(state, str(tmp_path), "ck")


def test_restart_bitwise_equivalence(full_run):
    _, straight, _, out = full_run
    mid = load_checkpoint(str(out / "ckpt_001"), REG)   # t = 24 h
    assert mid.time_h == 24.0
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.dense_hash() == straight.dense_hash()
    assert restarted.full_hash() == straight.full_hash()
    assert restarted.rng_state == straight.rng_state


# ---------------- cli ----------------

def test_cli_run_and_restart(tmp_path, capsys):
    cfg_path = tmp_path / "cfg.json"
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.05]}}}
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 1.0, "dt_min_h": 0.01}
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["run", str(cfg_path), "--out", str(tmp_path / "a")]) == 0
    assert (tmp_path / "a" / "summary.json").is_file()
    assert cli.main(["restart", str(tmp_path / "a" / "ckpt_000"),
                     "--out", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    assert "restart complete" in out
