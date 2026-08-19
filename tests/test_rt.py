"""v4.0/RT tests (PRD 3): mode B rate-limited re-equilibration — bit-identity
at tau <= dt, offered-fraction feed and pool blend, ledger closure under
0 < f < 1, restart identity. Worker-dependent tests skip without the xgems
env; the fake-backend tests run everywhere."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import ledger
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.registry import HYDRATE_PHASE_IDS, default_registry
from tinn.storage import load_checkpoint

from test_endmember import TwoEndmemberSnapshotBackend

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


def _gems_cfg(transport=None) -> TinnConfig:
    raw = json.loads((EXAMPLES / "opc_gems_32.json").read_text(encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw["schedule"] = {"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    if transport is not None:
        raw["transport"] = transport
    return TinnConfig.model_validate(raw)


def _fake_cfg(transport=None) -> TinnConfig:
    """gems3k-flavored config (tau validates against it); the tests inject a
    fake snapshot backend directly, so no worker is ever constructed."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0],
                                 "alpha": {"C3S": [0.0, 0.05]}}}
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    raw["chemistry"] = {"backend": "gems3k"}
    if transport is not None:
        raw["transport"] = transport
    return TinnConfig.model_validate(raw)


@pytest.fixture(scope="module")
def rt_full(tmp_path_factory):
    out = tmp_path_factory.mktemp("rt_full") / "run"
    state, summary = Engine(_gems_cfg()).run(out_dir=str(out))
    return state, out


@pytest.fixture(scope="module")
def rt_tau(tmp_path_factory):
    # tau = 100 h against a 6 h horizon: f = 0.02-0.06 per accepted step
    out = tmp_path_factory.mktemp("rt_tau") / "run"
    cfg = _gems_cfg(transport={"exchange_tau_h": 100.0})
    state, summary = Engine(cfg).run(out_dir=str(out))
    return state, out


# ---------------- mode B: fake-backend mechanics (no GEMS needed) ----------

def test_offered_fraction_equals_dt_over_tau():
    """The tau run feeds exactly f = dt/tau of the owned solid-solution
    elements the full run feeds, and the pool becomes withheld archive +
    parcels instead of an absolute replacement (PRD 4.6.1)."""
    # step 1 (no tau): builds a CSH pool at a 50:50 TobH/JenD split
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng1 = Engine(_fake_cfg(), reaction_backend=b1)
    t1, rej, _ = eng1.try_step(eng1.initial_state(), 2.0)
    assert rej is None
    sl = eng1._em_slice["CSH"]
    hi = HYDRATE_PHASE_IDS.index("CSH")
    owned_before = float(t1.hydrate_mol[hi])

    # step 2, full mode: capture the fed solid elements
    b_full = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    _, rej_f, _ = Engine(_fake_cfg(), reaction_backend=b_full).try_step(t1, 2.0)
    assert rej_f is None
    fed_full = [c["solid_elements"] for c in b_full.calls
                if c["solid_elements"] is not None]
    assert len(fed_full) == 1

    # step 2, tau = 4 h at dt = 2 h => f = 0.5 on the multi-DC channel
    b_tau = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    eng_tau = Engine(_fake_cfg(transport={"exchange_tau_h": 4.0}),
                     reaction_backend=b_tau)
    t2, rej_t, _ = eng_tau.try_step(t1, 2.0)
    assert rej_t is None
    fed_tau = [c["solid_elements"] for c in b_tau.calls
               if c["solid_elements"] is not None]
    assert len(fed_tau) == 1
    assert np.allclose(fed_tau[0], 0.5 * fed_full[0], rtol=1e-12, atol=1e-30)

    # pool blend: withheld archive (0.5 * owned, cap 1) + this step's parcels
    pool_csh = t2.cluster_endmember_mol[:, sl].sum(axis=0)
    want_total = 0.5 * owned_before + 2e-12
    assert float(pool_csh.sum()) == pytest.approx(want_total, rel=1e-9)
    # ledger closure survives the partial feed
    rep = ledger.check_all(t2, REG)
    assert not [v for v in rep.violations if v.startswith("balance_endmember")]


def test_tau_run_ledger_update_is_offered_signed():
    """hydrate_mol moves by (parcels - offered), not (parcels - owned): with
    f = 0.5 the withheld half stays on the books."""
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng1 = Engine(_fake_cfg(), reaction_backend=b1)
    t1, _, _ = eng1.try_step(eng1.initial_state(), 2.0)
    hi = HYDRATE_PHASE_IDS.index("CSH")
    owned = float(t1.hydrate_mol[hi])
    b_tau = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    eng_tau = Engine(_fake_cfg(transport={"exchange_tau_h": 4.0}),
                     reaction_backend=b_tau)
    t2, rej, _ = eng_tau.try_step(t1, 2.0)
    assert rej is None
    # new = owned - offered(0.5*owned) + parcels(2e-12)
    assert float(t2.hydrate_mol[hi]) == pytest.approx(
        0.5 * owned + 2e-12, rel=1e-12)


def test_unknown_per_phase_tau_is_refused():
    with pytest.raises(ValueError, match="unknown hydrate channel"):
        Engine(_fake_cfg(transport={
            "exchange_tau_h_per_phase": {"NotAPhase": 10.0}}),
            reaction_backend=TwoEndmemberSnapshotBackend(1e-12, 0.5))


# ---------------- mode B: coupled GEMS gates ------------------------------

@needs_gems
def test_tau_below_dt_is_bitwise_full_mode(rt_full, tmp_path):
    """tau <= dt_min => f == 1.0 exactly at every attempted dt => the engine
    takes the aliased legacy arrays — the trajectory is bit-identical."""
    full_state, _ = rt_full
    cfg = _gems_cfg(transport={"exchange_tau_h": 1e-4})
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "tau_tiny"))
    assert state.full_hash() == full_state.full_hash()


@needs_gems
def test_rate_limited_run_closes_all_ledgers(rt_full, rt_tau):
    """0 < f < 1 over a real coupled run: every blocking gate held at every
    accepted step (the run completed), the final state still closes, and the
    mode measurably changed the trajectory."""
    full_state, _ = rt_full
    tau_state, _ = rt_tau
    rep = ledger.check_all(tau_state, REG)
    assert rep.ok, rep.violations
    assert tau_state.full_hash() != full_state.full_hash()
    assert tau_state.time_h == full_state.time_h


@needs_gems
def test_restart_bit_identity_rate_limited(rt_tau):
    tau_state, out = rt_tau
    mid = load_checkpoint(str(out / "ckpt_000"), REG)   # t = 2 h
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.dense_hash() == tau_state.dense_hash()
    assert restarted.full_hash() == tau_state.full_hash()
