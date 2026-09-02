"""v4.0/RT tests (PRD 3): mode B rate-limited re-equilibration — bit-identity
at tau <= dt, offered-fraction feed and pool blend, ledger closure under
0 < f < 1, restart identity. Worker-dependent tests skip without the xgems
env; the fake-backend tests run everywhere."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import ledger, transport
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.registry import ELEMENT_IDS, HYDRATE_PHASE_IDS, default_registry
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


def test_tau_on_bundle_without_solid_solutions_is_refused():
    """A global tau against a bundle whose hydrates are all single-endmember
    would rate-limit nothing: exact legacy physics under a mode-B config
    hash. Refused loudly, never silently degraded (review finding)."""
    b = TwoEndmemberSnapshotBackend(1e-12, 0.5)
    b.hydrate_endmembers = {h: (h,) for h in HYDRATE_PHASE_IDS}
    with pytest.raises(ValueError, match="no hydrate channel"):
        Engine(_fake_cfg(transport={"exchange_tau_h": 100.0}),
               reaction_backend=b)


def test_undercovered_pool_archives_at_one_minus_f():
    """A pool that remembers less than the holdings must still be drawn at
    the f rate — min(psum, offered) would consume it at up to 2x (review
    finding). With the pool halved, the archive keeps (1-f) of the memory."""
    b1 = TwoEndmemberSnapshotBackend(csh_mol=3e-12, tob_frac=0.5)
    eng1 = Engine(_fake_cfg(), reaction_backend=b1)
    t1, _, _ = eng1.try_step(eng1.initial_state(), 2.0)
    sl = eng1._em_slice["CSH"]
    # halve the pool memory: psum = 1.5e-12 < owned = 3e-12
    t1.cluster_endmember_mol = t1.cluster_endmember_mol.copy()
    t1.cluster_endmember_mol[:, sl] *= 0.5
    psum_before = float(t1.cluster_endmember_mol[:, sl].sum())
    b_tau = TwoEndmemberSnapshotBackend(csh_mol=2e-12, tob_frac=0.25)
    eng_tau = Engine(_fake_cfg(transport={"exchange_tau_h": 4.0}),
                     reaction_backend=b_tau)
    t2, rej, _ = eng_tau.try_step(t1, 2.0)
    assert rej is None
    # fed pool part = f * psum -> archive = (1 - f) * psum survives, plus
    # this step's 2e-12 of parcels
    pool_after = float(t2.cluster_endmember_mol[:, sl].sum())
    assert pool_after == pytest.approx(0.5 * psum_before + 2e-12, rel=1e-9)


# ---------------- mode C: domain partition and exchange operator ----------

def test_label_domains_tiling_and_degenerate_limit():
    """tile == grid returns the literal cluster labels (legacy identity);
    a finer tile refines every cluster, ids ordered by (cluster, tile),
    every domain inside exactly one cluster."""
    rng = np.random.default_rng(7)
    liquid = (rng.random((8, 8, 8)) > 0.4) * 0.5
    labels, n_cl = transport.label_clusters(liquid)
    d_id, n_dom, d2c = transport.label_domains(labels, n_cl, (8, 8, 8))
    assert d_id is labels and n_dom == n_cl
    assert np.array_equal(d2c, np.arange(n_cl))
    d_id, n_dom, d2c = transport.label_domains(labels, n_cl, (4, 4, 4))
    assert n_dom >= n_cl
    wet = labels >= 0
    assert np.all(d_id[wet] >= 0) and np.all(d_id[~wet] == -1)
    # refinement: every domain's voxels lie in exactly one cluster
    for d in range(n_dom):
        cl = np.unique(labels[d_id == d])
        assert cl.size == 1 and cl[0] == d2c[d]
    # determinism across recomputation
    d_id2, n2, d2c2 = transport.label_domains(labels, n_cl, (4, 4, 4))
    assert n2 == n_dom and np.array_equal(d_id, d_id2)
    assert np.array_equal(d2c, d2c2)


def _two_domain_graph(w1, w2, g_edge, pitch):
    # edge_g carries geometric transmissibility G/p since the W4 per-axis
    # pitch generalization
    return transport.DomainGraph(
        n_domains=2,
        edge_a=np.array([0], dtype=np.int64),
        edge_b=np.array([1], dtype=np.int64),
        edge_g=np.array([g_edge / pitch]),
        water=np.array([w1, w2]),
        dust=np.zeros(2, dtype=bool))


def test_two_domain_exchange_matches_analytic_decay():
    """RT analytic anchor 1 (PRD 6.2): a single BE step obeys
    dc+ = dc / (1 + dt*lambda), lambda = (D0 G / p)(1/W1 + 1/W2), exactly;
    substepping converges first-order to the exponential."""
    w1, w2, g_edge, d0, p = 2.0, 1.0, 0.3, 1.5, 4.0
    lam = (d0 * g_edge / p) * (1.0 / w1 + 1.0 / w2)
    inv = np.array([[6.0], [0.0]])
    dt = 0.8
    res = transport.exchange_be(_two_domain_graph(w1, w2, g_edge, p),
                                inv, dt, d0)
    assert res.status == "ok"
    new = inv + res.delta
    dc0 = inv[0, 0] / w1 - inv[1, 0] / w2
    dc1 = new[0, 0] / w1 - new[1, 0] / w2
    assert dc1 == pytest.approx(dc0 / (1.0 + dt * lam), rel=1e-12)
    # conservation is exact by flux form
    assert new.sum() == pytest.approx(6.0, abs=0.0)
    # n substeps -> exponential, first order in dt
    for n_sub, tol in ((8, 0.05), (64, 0.007)):
        cur = inv.copy()
        for _ in range(n_sub):
            r = transport.exchange_be(_two_domain_graph(w1, w2, g_edge, p),
                                      cur, dt / n_sub, d0)
            cur = cur + r.delta
        dcn = cur[0, 0] / w1 - cur[1, 0] / w2
        assert dcn == pytest.approx(dc0 * np.exp(-lam * dt), rel=tol)


def test_ring_exchange_invariants():
    """RT ring invariants: exact per-element totals, positivity after
    repair, uniform concentration is a stationary point, and bitwise
    determinism across repeated calls."""
    rng = np.random.default_rng(3)
    k = 12
    ea = np.arange(k, dtype=np.int64)
    eb = np.roll(ea, -1)
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    graph = transport.DomainGraph(
        n_domains=k, edge_a=lo, edge_b=hi,
        edge_g=(0.1 + rng.random(k)) / 2.0,
        water=0.5 + rng.random(k), dust=np.zeros(k, dtype=bool))
    inv = rng.random((k, len(ELEMENT_IDS))) * 1e-9
    res = transport.exchange_be(graph, inv, 5.0, 2.0)
    assert res.status == "ok"
    new = inv + res.delta
    assert np.all(new >= 0.0)
    for e in range(len(ELEMENT_IDS)):
        assert float(new[:, e].sum()) == pytest.approx(
            float(inv[:, e].sum()), rel=1e-13)
    # uniform c stationary: n = W * const per element
    uni = np.outer(graph.water, np.linspace(0.5, 1.5, len(ELEMENT_IDS))) * 1e-8
    res_u = transport.exchange_be(graph, uni, 5.0, 2.0)
    assert float(np.abs(res_u.delta).max()) <= 1e-22
    # determinism
    res2 = transport.exchange_be(graph, inv, 5.0, 2.0)
    assert np.array_equal(res.delta, res2.delta)


def test_conductance_field_matches_report_network_rule():
    """RT analytic anchor 2 precursor: the shared helper reproduces the
    report solver's cell rule (liquid + 0.0025 * gel hydrates), floorless."""
    liquid = np.zeros((4, 4, 4))
    liquid[0] = 0.7
    hyd = np.zeros((2, 4, 4, 4))
    hyd[0, 1] = 0.5      # gel-bearing
    hyd[1, 2] = 0.5      # crystalline
    g = transport.conductance_field(liquid, hyd, np.array([0.28, 0.0]),
                                    0.0025)
    assert g[0, 0, 0] == pytest.approx(0.7)
    assert g[1, 0, 0] == pytest.approx(0.0025 * 0.5)
    assert g[2, 0, 0] == 0.0 and g[3, 0, 0] == 0.0


# ---------------- mode C: engine integration ------------------------------

def test_one_domain_limit_is_bitwise_full_mode_stoich(tmp_path):
    """tile_vox == grid_size makes label_domains return the cluster labels
    themselves - the whole trajectory must be bit-identical to mode full
    (the key cheap gate: runs without xGEMS)."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [24.0, 72.0], "dt_initial_h": 6.0,
                       "dt_min_h": 0.01}
    full_cfg = TinnConfig.model_validate(raw)
    dom_cfg = TinnConfig.model_validate(
        {**raw, "transport": {"domains": {"tile_vox": 32,
                                          "d0_m2_s": 1.0e-9}}})
    s_full, _ = Engine(full_cfg).run(out_dir=str(tmp_path / "full"))
    s_dom, _ = Engine(dom_cfg).run(out_dir=str(tmp_path / "dom"))
    assert s_dom.full_hash() == s_full.full_hash()


def test_subdomain_stoich_run_closes_and_is_deterministic(tmp_path):
    """A genuinely partitioned run (tile 8 on 32^3): completes, keeps every
    blocking gate green, keys the inventory rows on domains, and reproduces
    itself bitwise on a second run."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [24.0, 72.0], "dt_initial_h": 6.0,
                       "dt_min_h": 0.01}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    cfg = TinnConfig.model_validate(raw)
    s1, _ = Engine(cfg).run(out_dir=str(tmp_path / "a"))
    rep = ledger.check_all(s1, REG)
    assert rep.ok, rep.violations
    labels, n_cl = transport.label_clusters(s1.capillary_liquid)
    _, n_dom, _ = transport.label_domains(labels, n_cl, (8, 8, 8))
    assert s1.cluster_inventory.shape[0] == n_dom
    assert n_dom > n_cl
    s2, _ = Engine(cfg).run(out_dir=str(tmp_path / "b"))
    assert s2.full_hash() == s1.full_hash()


def test_subdomain_restart_bit_identity(tmp_path):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["schedule"] = {"output_times_h": [24.0, 72.0], "dt_initial_h": 6.0,
                       "dt_min_h": 0.01}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    cfg = TinnConfig.model_validate(raw)
    straight, _ = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    mid = load_checkpoint(str(tmp_path / "run" / "ckpt_000"), REG)
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.full_hash() == straight.full_hash()


# ---------------- RT-W2b: GEM-call economy --------------------------------

def _econ_cfg(**dom):
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 24.0],
                                 "alpha": {"C3S": [0.0, 0.12]}}}
    raw["schedule"] = {"output_times_h": [6.0, 12.0, 18.0, 24.0],
                       "dt_initial_h": 6.0, "dt_min_h": 0.01}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1e-9, **dom}}
    return TinnConfig.model_validate(raw)


def test_economy_defers_element_exactly_and_age_forces():
    """With an unreachable dirty threshold, every domain equilibrates on
    first touch, then defers (releases accumulate in the inventory,
    element-exactly) until the age bound forces a re-equilibration."""
    cfg = _econ_cfg(dirty_rtol=1e9, eq_max_age_steps=2)
    eng = Engine(cfg)
    s0 = eng.initial_state()
    t1, rej, m1 = eng.try_step(s0, 6.0)
    assert rej is None
    assert m1["n_deferred"] == 0.0          # first touch: all forced
    assert np.all(t1.domain_eq_age >= 0)
    t2, rej, m2 = eng.try_step(t1, 6.0)
    assert rej is None
    assert m2["n_deferred"] > 0.0           # age 1 < 2, drift below 1e9
    # deferred releases sit in the inventory, element-exactly
    assert float(np.abs(t2.cluster_inventory).sum()) > \
        float(np.abs(t1.cluster_inventory).sum())
    assert ledger.check_all(t2, REG).ok
    t3, rej, m3 = eng.try_step(t2, 6.0)
    assert rej is None
    assert m3["n_deferred"] > 0.0           # ages at 1, bound is 2
    t4, rej, m4 = eng.try_step(t3, 6.0)
    assert rej is None
    assert m4["max_eq_age"] >= 2.0          # the age bound fires here:
    assert m4["n_gem_selected"] > 0.0       # aged domains re-equilibrate
    assert np.all(t4.domain_eq_age <= 2)
    assert ledger.check_all(t4, REG).ok


def test_economy_budget_caps_calls_deterministically():
    cfg = _econ_cfg(dirty_rtol=0.0, max_gem_calls_per_step=1,
                    eq_max_age_steps=1000)
    eng = Engine(cfg)
    s0 = eng.initial_state()
    t1, rej, _ = eng.try_step(s0, 6.0)      # first touch: forced, uncapped
    assert rej is None
    t2, rej, m2 = eng.try_step(t1, 6.0)
    assert rej is None
    # dirty candidates capped at 1; no forced categories fire here (no
    # salts, ages far below the bound, labels stable step-to-step)
    assert m2["n_gem_selected"] <= 1.0
    assert m2["n_deferred"] > 0.0
    t2b, _, m2b = eng.try_step(t1, 6.0)
    assert m2b["n_gem_selected"] == m2["n_gem_selected"]
    assert t2b.full_hash() == t2.full_hash()


def test_economy_restart_bit_identity(tmp_path):
    cfg = _econ_cfg(dirty_rtol=1e9, eq_max_age_steps=3)
    straight, _ = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    mid = load_checkpoint(str(tmp_path / "run" / "ckpt_001"), REG)
    assert mid.domain_eq_age.shape[0] > 0   # snapshots rode the checkpoint
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.full_hash() == straight.full_hash()


def test_economy_off_leaves_snapshots_empty(tmp_path):
    cfg = _econ_cfg()                        # dirty_rtol 0.0, no budget
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "pure"))
    assert state.domain_eq_inventory.shape[0] == 0
    assert state.domain_eq_age.shape[0] == 0


# ---------------- RT-W3: boundary reservoir -------------------------------

def test_label_clusters_nonperiodic_axis_splits_wrap():
    """A strip crossing the z seam is one periodic cluster but two under a
    non-periodic z; transverse wraps stay; recomputation identical."""
    liquid = np.zeros((6, 6, 6))
    liquid[0, 2, 2] = 0.5
    liquid[5, 2, 2] = 0.5              # touching only through the z seam
    lp, np_cl = transport.label_clusters(liquid)
    assert np_cl == 1
    ln, nn = transport.label_clusters(liquid, (False, True, True))
    assert nn == 2
    assert ln[0, 2, 2] != ln[5, 2, 2]
    # transverse (y) wrap still connects under non-periodic z
    liquid2 = np.zeros((6, 6, 6))
    liquid2[2, 0, 2] = 0.5
    liquid2[2, 5, 2] = 0.5
    _, n2 = transport.label_clusters(liquid2, (False, True, True))
    assert n2 == 1
    ln2, nn2 = transport.label_clusters(liquid, (False, True, True))
    assert nn2 == nn and np.array_equal(ln, ln2)


def test_label_clusters_all_periodic_default_identical():
    rng = np.random.default_rng(11)
    liquid = (rng.random((8, 8, 8)) > 0.5) * 0.4
    a, na = transport.label_clusters(liquid)
    b, nb = transport.label_clusters(liquid, (True, True, True))
    assert na == nb and np.array_equal(a, b)
    # a field with no seam-crossing pair labels identically either way
    interior = np.zeros((8, 8, 8))
    interior[2:6, 2:6, 2:6] = 0.5
    c, ncc = transport.label_clusters(interior)
    d, ndd = transport.label_clusters(interior, (False, True, True))
    assert ncc == ndd and np.array_equal(c, d)


def test_domain_graph_exposed_wrap_edge_excluded():
    """Wet slabs at z=0 and z=N-1 in one cluster via an interior bridge:
    periodic graph carries the seam edge, the exposed graph does not; a
    zero-coupling bath degrades exchange_be to the bath-None result."""
    liquid = np.zeros((6, 6, 6))
    liquid[:, 2, 2] = 0.5              # a full z-column: bridge + both faces
    labels, n_cl = transport.label_clusters(liquid, (False, True, True))
    assert n_cl == 1
    d_id, n_dom, _ = transport.label_domains(labels, n_cl, (2, 2, 2))
    g = np.where(liquid > 0, 0.5, 0.0)
    g_per = transport.build_domain_graph(d_id, n_dom, g, liquid)
    g_exp = transport.build_domain_graph(d_id, n_dom, g, liquid,
                                         (False, True, True))
    assert g_per.edge_a.size == g_exp.edge_a.size + 1   # exactly the seam
    inv = np.linspace(1.0, 2.0, n_dom)[:, None] * np.ones((1, len(ELEMENT_IDS))) * 1e-10
    r0 = transport.exchange_be(g_exp, inv, 1.0, 2.0)
    rz = transport.exchange_be(g_exp, inv, 1.0, 2.0,
                               bath=transport.BoundaryBath(
                                   g_bnd=np.zeros(n_dom),
                                   c_res=np.zeros(len(ELEMENT_IDS))))
    assert np.array_equal(r0.delta, rz.delta)
    assert np.array_equal(rz.boundary_net, np.zeros(len(ELEMENT_IDS)))


def test_boundary_coupling_halfcell_sum():
    liquid = np.zeros((4, 4, 4))
    liquid[0] = 0.5                     # wet low-z face
    liquid[-1, :2] = 0.5                # partially wet high-z face
    labels, n_cl = transport.label_clusters(liquid, (False, True, True))
    d_id, n_dom, _ = transport.label_domains(labels, n_cl, (4, 4, 4))
    g = np.where(liquid > 0, 0.3, 0.7)  # dry voxels conduct via gel: ignored
    low = transport.boundary_coupling(d_id, n_dom, g, 0, True, False)
    high = transport.boundary_coupling(d_id, n_dom, g, 0, False, True)
    both = transport.boundary_coupling(d_id, n_dom, g, 0, True, True)
    assert low.sum() == pytest.approx(2.0 * 0.3 * 16)   # 16 wet face voxels
    assert high.sum() == pytest.approx(2.0 * 0.3 * 8)   # 8 wet face voxels
    assert np.allclose(both, low + high)
    again = transport.boundary_coupling(d_id, n_dom, g, 0, True, True)
    assert np.array_equal(both, again)


def test_single_domain_bath_analytic_decay_and_ingress():
    """RT analytic anchor 4 (PRD 6.2): (c+ - c_R) = (c - c_R)/(1 + dt*lam),
    lam = D0*G_AR/(p*W), exactly; substeps -> exponential; ingress sign."""
    w, g_ar, d0, p = 2.0, 0.6, 1.5, 4.0
    lam = d0 * g_ar / (p * w)
    graph = transport.DomainGraph(
        n_domains=1, edge_a=np.empty(0, dtype=np.int64),
        edge_b=np.empty(0, dtype=np.int64), edge_g=np.empty(0),
        water=np.array([w]), dust=np.zeros(1, dtype=bool))
    c_res = np.full(len(ELEMENT_IDS), 0.25)
    inv = np.full((1, len(ELEMENT_IDS)), 3.0)         # c = 1.5 > c_res: leaches out
    dt = 0.7
    res = transport.exchange_be(graph, inv, dt, d0,
                                bath=transport.BoundaryBath(
                                    g_bnd=np.array([g_ar / p]),
                                    c_res=c_res))
    assert res.status == "ok"
    c0 = inv[0, 0] / w
    c1 = (inv[0, 0] + res.delta[0, 0]) / w
    assert c1 - 0.25 == pytest.approx((c0 - 0.25) / (1.0 + dt * lam),
                                      rel=1e-12)
    assert np.all(res.boundary_net < 0.0)          # out of the system
    # substepping converges first-order to the exponential
    cur = inv.copy()
    for _ in range(64):
        r = transport.exchange_be(graph, cur, dt / 64, d0,
                                  bath=transport.BoundaryBath(
                                      g_bnd=np.array([g_ar / p]),
                                      c_res=c_res))
        cur = cur + r.delta
    assert cur[0, 0] / w - 0.25 == pytest.approx(
        (c0 - 0.25) * np.exp(-lam * dt), rel=0.01)
    # ingress: bath above the domain concentration
    rich = np.full(len(ELEMENT_IDS), 5.0)
    r_in = transport.exchange_be(graph, inv, dt, d0,
                                 bath=transport.BoundaryBath(
                                     g_bnd=np.array([g_ar / p]),
                                     c_res=rich))
    assert np.all(r_in.boundary_net > 0.0)
    # equilibrium bath: flux bounded by CG dust, not bitwise zero
    eq = transport.exchange_be(graph, inv, dt, d0,
                               bath=transport.BoundaryBath(
                                   g_bnd=np.array([g_ar / p]),
                                   c_res=np.full(len(ELEMENT_IDS), c0)))
    assert float(np.abs(eq.boundary_net).max()) <= 1e-24 + 1e-12 * 3.0


def test_bath_flux_form_conservation_and_signs():
    rng = np.random.default_rng(5)
    k = 8
    ea = np.arange(k - 1, dtype=np.int64)
    eb = ea + 1
    graph = transport.DomainGraph(
        n_domains=k, edge_a=ea, edge_b=eb,
        edge_g=(0.2 + rng.random(k - 1)) / 2.0,
        water=0.5 + rng.random(k), dust=np.zeros(k, dtype=bool))
    g_bnd = np.zeros(k)
    g_bnd[0] = 0.8                       # bath on one end
    inv = rng.random((k, len(ELEMENT_IDS))) * 1e-9
    res = transport.exchange_be(graph, inv, 2.0, 1.0,
                                bath=transport.BoundaryBath(
                                    g_bnd=g_bnd / 2.0, c_res=np.zeros(len(ELEMENT_IDS))))
    assert res.status == "ok"
    new = inv + res.delta
    assert np.all(new >= 0.0)
    # domain loss equals the boundary accumulator (per element, to the
    # exchange-gate tolerance - summation order dust only)
    for e in range(len(ELEMENT_IDS)):
        loss = float(res.delta[:, e].sum())
        assert loss == pytest.approx(float(res.boundary_net[e]),
                                     abs=1e-24 + 1e-12 * float(
                                         np.abs(res.delta[:, e]).sum()))
    assert np.all(res.boundary_net <= 0.0)   # pure-water bath only leaches
    res2 = transport.exchange_be(graph, inv, 2.0, 1.0,
                                 bath=transport.BoundaryBath(
                                     g_bnd=g_bnd / 2.0, c_res=np.zeros(len(ELEMENT_IDS))))
    assert np.array_equal(res.delta, res2.delta)


def test_boundary_config_refusals():
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    good_dom = {"tile_vox": 8, "d0_m2_s": 1e-9}
    bath = {"axis": "z", "side": "low", "composition_mol_per_m3": {}}
    for transport_cfg, err in (
            ({"domains": good_dom,
              "boundary": {**bath, "composition_mol_per_m3": {"Xx": 1.0}}},
             "unknown bath elements"),
            ({"domains": good_dom,
              "boundary": {**bath,
                           "composition_mol_per_m3": {"Na": -1.0}}},
             "finite number"),
            ({"boundary": bath}, "needs transport.domains"),
            ({"domains": good_dom, "boundary": bath}, "gems3k")):
        with pytest.raises(Exception, match=err):
            TinnConfig.model_validate({**raw, "transport": transport_cfg})


def test_boundary_engine_run_closes_forced_and_restarts(tmp_path):
    """Fake snapshot backend, pure-water bath on z-low: elements leave with
    the closure identity intact (zero ledger edits), bath-coupled domains
    stay forced under a huge dirty threshold, and an interrupted run
    restarts bit-identically (cumulative boundary ledger included)."""
    # tile 16 (8 domains) and a small fixed parcel: the fake emits a
    # constant assemblage PER CALL, so per-domain water must cover it
    tr = {"domains": {"tile_vox": 16, "d0_m2_s": 1e-9, "dirty_rtol": 1e9,
                      "eq_max_age_steps": 1000},
          "boundary": {"axis": "z", "side": "low",
                       "composition_mol_per_m3": {}}}
    cfg = _fake_cfg(transport=tr)
    b = TwoEndmemberSnapshotBackend(csh_mol=2e-13, tob_frac=0.5)
    eng = Engine(cfg, reaction_backend=b)
    s0 = eng.initial_state()
    t1, rej, m1 = eng.try_step(s0, 2.0)
    assert rej is None
    t2, rej, m2 = eng.try_step(t1, 2.0)
    assert rej is None
    assert m2["n_boundary_coupled"] > 0.0
    assert m2["n_gem_selected"] >= m2["n_boundary_coupled"]  # forced
    assert 0.0 <= m2["boundary_supply_ratio"]
    leached = t2.boundary_exchanged_elements
    assert float(leached.sum()) < 0.0            # net out into pure water
    rep = ledger.check_all(t2, REG)
    assert rep.ok, rep.violations
    # restart bit identity through run(): straight vs resumed
    cfg2 = _fake_cfg(transport=tr)
    straight, _ = Engine(cfg2, reaction_backend=TwoEndmemberSnapshotBackend(
        2e-13, 0.5)).run(out_dir=str(tmp_path / "s"))
    mid = load_checkpoint(str(tmp_path / "s" / "ckpt_000"), REG)
    restarted, _ = Engine(mid.config,
                          reaction_backend=TwoEndmemberSnapshotBackend(
                              2e-13, 0.5)).run(state=mid)
    assert restarted.full_hash() == straight.full_hash()


def test_bath_anchor_seed_books_to_boundary():
    """W4.1 measured: supply-limited drained front domains re-earn the
    worker's redox seed on every call; at fine dt the per-call dust grows
    with step count and trips the injected cap (abort at t=168.99 h). A
    bath-coupled domain's seed is physically bath re-supply - it books to
    the boundary ledger; sealed runs keep the strict injected accounting."""
    from tinn.registry import ELEMENT_IDS
    o = ELEMENT_IDS.index("O")

    class _Seeding(TwoEndmemberSnapshotBackend):
        def react(self, *a, **k):
            r = super().react(*a, **k)
            r.injected_elements = r.injected_elements.copy()
            r.injected_elements[o] += 1e-20
            r.residual_inventory = r.residual_inventory.copy()
            r.residual_inventory[o] += 1e-20
            return r

    dom = {"tile_vox": 16, "d0_m2_s": 1e-9, "dirty_rtol": 1e9,
           "eq_max_age_steps": 1000}
    tr = {"domains": dict(dom),
          "boundary": {"axis": "z", "side": "low",
                       "composition_mol_per_m3": {}}}
    eng = Engine(_fake_cfg(transport=tr), reaction_backend=_Seeding(2e-13, 0.5))
    t1, rej, _ = eng.try_step(eng.initial_state(), 2.0)
    assert rej is None
    t2, rej, m2 = eng.try_step(t1, 2.0)
    assert rej is None
    assert float(np.abs(t2.injected_elements).sum()) == 0.0
    assert m2.get("bath_anchor_mol", 0.0) > 0.0
    rep = ledger.check_all(t2, REG)
    assert rep.ok, rep.violations
    # sealed control: the same seed stays booked as injected
    eng_s = Engine(_fake_cfg(transport={"domains": dict(dom)}),
                   reaction_backend=_Seeding(2e-13, 0.5))
    s1, rej, _ = eng_s.try_step(eng_s.initial_state(), 2.0)
    assert rej is None
    assert s1.injected_elements[o] > 0.0


def test_boundary_mirror_symmetry():
    """DoD 4: a z-mirror-symmetric field with the same bath on BOTH faces
    exchanges mirror-symmetrically, layer by layer (rel 1e-12 - summation
    order differs, so not bitwise). Witnesses face-declaration sign and
    orientation bugs."""
    n = 8
    liquid = np.zeros((n, n, n))
    liquid[:, 3:5, 3:5] = 0.4           # a z-column, mirror-symmetric in z
    labels, n_cl = transport.label_clusters(liquid, (False, True, True))
    d_id, n_dom, _ = transport.label_domains(labels, n_cl, (2, 2, 2))
    g = np.where(liquid > 0, 0.5, 0.0)
    graph = transport.build_domain_graph(d_id, n_dom, g, liquid,
                                         (False, True, True))
    g_ar = transport.boundary_coupling(d_id, n_dom, g, 0, True, True)
    # z-symmetric inventory profile: n proportional to water, uniform c
    inv = np.outer(graph.water, np.ones(len(ELEMENT_IDS))) * 2.5e-10
    res = transport.exchange_be(graph, inv, 1.0, 2.0,
                                bath=transport.BoundaryBath(
                                    g_bnd=g_ar / 2.0, c_res=np.zeros(len(ELEMENT_IDS))))
    assert res.status == "ok"
    # per-z-layer delta via domain -> layer map (tile 2 => 4 z-bands)
    layer_delta = np.zeros((n // 2, len(ELEMENT_IDS)))
    for d in range(n_dom):
        zs = np.flatnonzero((d_id == d).any(axis=(1, 2)))
        band = int(zs[0]) // 2
        layer_delta[band] += res.delta[d]
    for b in range(n // 4):
        assert np.allclose(layer_delta[b], layer_delta[-1 - b],
                           rtol=1e-12, atol=1e-30)


# ---------------- review-gate regressions (2026-08-20 round 2) ------------

def test_switch_permutation_is_remapped_exactly():
    """A seam-only cluster split can PRESERVE the domain count while
    permuting ids (review repro) - the switch therefore remaps
    unconditionally. Pin the mechanism: identical liquid, permuted labels
    => remap_inventories moves each row to its voxel set bit-exactly."""
    liquid = np.zeros((8, 8, 8))
    liquid[[0, 1, 2, 3, 6, 7], 0, 0] = 0.5   # chain closed through z seam
    liquid[4, 4, 4] = 0.5                    # independent pocket
    lp, ncp = transport.label_clusters(liquid)
    le, nce = transport.label_clusters(liquid, (False, True, True))
    dp, ndp, _ = transport.label_domains(lp, ncp, (2, 8, 8))
    de, nde, _ = transport.label_domains(le, nce, (2, 8, 8))
    assert ndp == nde                        # the count-proxy trap
    assert not np.array_equal(dp, de)        # ...while ids permute
    rows = (np.arange(ndp, dtype=float)[:, None]
            * np.ones((1, len(ELEMENT_IDS))) + 1.0)
    res = transport.remap_inventories(dp, liquid, de, liquid, rows, nde)
    assert not res.dryout
    # every voxel's row content followed its voxel set exactly
    wet = liquid > transport.LIQ_EPS
    for old_id, new_id in zip(dp[wet], de[wet]):
        assert np.array_equal(res.inventory[new_id], rows[old_id])


def test_neighbor_attribution_respects_exposed_axis():
    """Review repro: a dry face site must not be attributed to the cluster
    on the OPPOSITE face through the wrap when the axis is exposed."""
    from tinn.engine import _neighbor_best_label
    n = 6
    liquid = np.zeros((n, n, n))
    liquid[n - 1, 2, 2] = 0.9                # far face, wet
    liquid[0, 3, 2] = 0.1                    # near face neighbor, wet
    labels, n_cl = transport.label_clusters(liquid, (False, True, True))
    assert n_cl == 2
    per = _neighbor_best_label(labels, liquid)
    exp = _neighbor_best_label(labels, liquid, (False, True, True))
    site = (0, 2, 2)                         # dry site on the exposed face
    assert per[site] == labels[n - 1, 2, 2]  # periodic: far face wins
    assert exp[site] == labels[0, 3, 2]      # exposed: wall blocks the wrap


def test_start_h_snaps_to_matching_output_time():
    """A start_h within the 1e-9 validator tolerance must resolve to the
    EXACT output float, or activation/switch epsilons diverge (review
    finding: 1e-10 offset de-periodized mid-window with no remap)."""
    tr = {"domains": {"tile_vox": 16, "d0_m2_s": 1e-9},
          "boundary": {"axis": "z", "side": "low",
                       "start_h": 2.0 + 1e-10,
                       "composition_mol_per_m3": {}}}
    eng = Engine(_fake_cfg(transport=tr),
                 reaction_backend=TwoEndmemberSnapshotBackend(2e-13, 0.5))
    assert eng._b_start == 2.0


def test_dryout_surrender_witnesses():
    """W4.1 measured: the sealed->exposed switch orphans wrap-connected
    pocket clusters, which self-desiccate carrying ~1e-20 mol float dust;
    under an ACTIVE bath the dead cluster's row surrenders to the boundary
    ledger (exact floats). MATERIAL solutes or water keep the v2 reject."""
    from tinn.engine import _dryout_surrender
    cl_labels = np.array([[[0, 0, 1]]])
    prev_liquid = np.array([[[3.0, 4.0, 1e-4]]])
    dom_to_cl = np.array([0, 0, 1])
    rows = np.zeros((3, len(ELEMENT_IDS)))
    rows[0, 0] = 1.0
    rows[1, 1] = 0.5
    rows[2, :3] = 1e-9                       # dust row, trace water
    assert _dryout_surrender(rows, [2], dom_to_cl, cl_labels,
                             prev_liquid) == [2]
    rows[2, :3] = 1e-3                       # material solutes -> reject
    assert _dryout_surrender(rows, [2], dom_to_cl, cl_labels,
                             prev_liquid) is None
    rows[2, :3] = 1e-9
    wet = prev_liquid.copy()
    wet[0, 0, 2] = 5.0                       # material water -> reject
    assert _dryout_surrender(rows, [2], dom_to_cl, cl_labels, wet) is None


# ---------------- RT-W4 enablers ------------------------------------------

def test_boundary_start_h_pre_window_is_bitwise_sealed(tmp_path):
    """start_h delays the bath: up to that output boundary the run is fully
    periodic and bit-identical to the same config WITHOUT the boundary -
    the honest structural gate for the mature-paste protocol."""
    dom = {"tile_vox": 16, "d0_m2_s": 1e-9}
    sealed_cfg = _fake_cfg(transport={"domains": dom})
    exposed_cfg = _fake_cfg(transport={
        "domains": dom,
        "boundary": {"axis": "z", "side": "low", "start_h": 2.0,
                     "composition_mol_per_m3": {}}})
    b = lambda: TwoEndmemberSnapshotBackend(csh_mol=2e-13, tob_frac=0.5)
    s_sealed, _ = Engine(sealed_cfg, reaction_backend=b()).run(
        out_dir=str(tmp_path / "sealed"))
    s_exp, _ = Engine(exposed_cfg, reaction_backend=b()).run(
        out_dir=str(tmp_path / "exposed"))
    mid_sealed = load_checkpoint(str(tmp_path / "sealed" / "ckpt_000"), REG)
    mid_exp = load_checkpoint(str(tmp_path / "exposed" / "ckpt_000"), REG)
    assert mid_exp.dense_hash() == mid_sealed.dense_hash()   # t = 2 h
    # after start_h the bath engages and the trajectories diverge
    assert s_exp.full_hash() != s_sealed.full_hash()
    assert float(s_exp.boundary_exchanged_elements.sum()) < 0.0
    # start_h off an output boundary is refused
    with pytest.raises(Exception, match="coincide with an output"):
        _fake_cfg(transport={
            "domains": dom,
            "boundary": {"axis": "z", "side": "low", "start_h": 1.7,
                         "composition_mol_per_m3": {}}})


def test_banded_tiles_partition_and_hash():
    """tile_zyx banded tiling (transverse = grid): bands along z only, the
    W3-review cost lever; absent tile_zyx keeps the cubic hash."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    base = TinnConfig.model_validate({**raw, "transport": {
        "domains": {"tile_vox": 8, "d0_m2_s": 1e-9}}}).config_hash()
    with_none = TinnConfig.model_validate({**raw, "transport": {
        "domains": {"tile_vox": 8, "d0_m2_s": 1e-9,
                    "tile_zyx": None}}}).config_hash()
    assert with_none == base
    banded = TinnConfig.model_validate({**raw, "transport": {
        "domains": {"tile_vox": 8, "d0_m2_s": 1e-9,
                    "tile_zyx": (4, 32, 32)}}})
    assert banded.config_hash() != base
    with pytest.raises(Exception, match="does not divide"):
        TinnConfig.model_validate({**raw, "transport": {
            "domains": {"tile_vox": 8, "d0_m2_s": 1e-9,
                        "tile_zyx": (5, 32, 32)}}})
    # banded labeling: domains = clusters x z-bands
    rng = np.random.default_rng(9)
    liquid = (rng.random((8, 8, 8)) > 0.35) * 0.5
    labels, n_cl = transport.label_clusters(liquid)
    d_id, n_dom, d2c = transport.label_domains(labels, n_cl, (2, 8, 8))
    assert n_dom >= n_cl
    for d in range(n_dom):
        zs = np.flatnonzero((d_id == d).any(axis=(1, 2)))
        assert zs.max() - zs.min() <= 1          # confined to one 2-band
        cl = np.unique(labels[d_id == d])
        assert cl.size == 1 and cl[0] == d2c[d]


def test_dt_windows_schedule_and_hash():
    """Ported dt_windows (verbatim from platform WIP): validation matrix,
    absent windows keep the legacy hash, and the engine honors the caps."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    base = TinnConfig.model_validate(raw).config_hash()
    raw2 = json.loads(json.dumps(raw))
    raw2["schedule"]["dt_windows"] = None
    assert TinnConfig.model_validate(raw2).config_hash() == base
    good = json.loads(json.dumps(raw))
    good["schedule"] = {"output_times_h": [24.0, 168.0], "dt_initial_h": 6.0,
                        "dt_min_h": 0.01,
                        "dt_windows": [{"until_h": 24.0, "dt_h": 6.0},
                                       {"until_h": 168.0, "dt_h": 12.0}]}
    cfg = TinnConfig.model_validate(good)
    assert cfg.schedule.dt_cap_at(1.0) == 6.0
    assert cfg.schedule.dt_cap_at(30.0) == 12.0
    assert cfg.schedule.next_window_end_after(1.0) == 24.0
    for bad in ([{"until_h": 24.0, "dt_h": 5.0}],                # != initial
                [{"until_h": 24.0, "dt_h": 6.0}],                # short cover
                [{"until_h": 24.0, "dt_h": 6.0},
                 {"until_h": 20.0, "dt_h": 1.0}],                # not increasing
                [{"until_h": 168.0, "dt_h": 0.001}]):            # < dt_min
        b2 = json.loads(json.dumps(good))
        b2["schedule"]["dt_windows"] = bad
        with pytest.raises(Exception):
            TinnConfig.model_validate(b2)
    # engine honors the caps: accepted dts never exceed the window cap
    run_raw = json.loads(json.dumps(raw))
    run_raw["schedule"] = {"output_times_h": [12.0, 24.0], "dt_initial_h": 3.0,
                           "dt_min_h": 0.01,
                           "dt_windows": [{"until_h": 12.0, "dt_h": 3.0},
                                          {"until_h": 24.0, "dt_h": 6.0}]}
    caps = []

    def hook(ev):
        if ev.get("event") == "step_accepted":
            caps.append((ev["time_end_h"], ev["dt_accepted_h"],
                         ev["scheduled_dt_cap_h"]))
    Engine(TinnConfig.model_validate(run_raw)).run(audit_hook=hook)
    assert all(dt <= cap + 1e-12 for _, dt, cap in caps)
    assert any(cap == 6.0 for _, _, cap in caps)
    assert any(cap == 3.0 for _, _, cap in caps)


@needs_gems
def test_boundary_gems_leach_closes(rt_full, tmp_path):
    """PC bundle, pure-water bath, short horizon: completes, closes, Ca
    accumulator negative, trajectory differs from the sealed twin."""
    full_state, _ = rt_full
    cfg = _gems_cfg(transport={
        "domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9},
        "boundary": {"axis": "z", "side": "low",
                     "composition_mol_per_m3": {}}})
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "leach"))
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    from tinn.registry import ELEMENT_IDS
    assert state.boundary_exchanged_elements[
        ELEMENT_IDS.index("Ca")] < 0.0
    assert state.full_hash() != full_state.full_hash()


# ---------------- mode B: coupled GEMS gates ------------------------------

@needs_gems
def test_tau_below_dt_is_bitwise_full_mode(rt_full, tmp_path):
    """tau <= every ATTEMPTED dt => f == 1.0 exactly => the engine takes the
    aliased legacy arrays — the trajectory is bit-identical. tau = 1.0 h
    against the 2.0 h cruise dt (this run accepts every step; even one
    halving to 1.0 h still gives f = min(1, 1.0/1.0) = 1.0 exactly).
    tau <= dt_min itself is refused at config time (review rec. 1)."""
    full_state, _ = rt_full
    cfg = _gems_cfg(transport={"exchange_tau_h": 1.0})
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "tau_small"))
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


@needs_gems
def test_one_domain_limit_is_bitwise_full_mode_gems(rt_full, tmp_path):
    full_state, _ = rt_full
    cfg = _gems_cfg(transport={"domains": {"tile_vox": 32,
                                           "d0_m2_s": 1.0e-9}})
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "dom1"))
    assert state.full_hash() == full_state.full_hash()


@needs_gems
def test_combined_modes_run_and_close(rt_full, tmp_path):
    """B + C composed on a real coupled run: rate-limited feed inside
    diffusively coupled sub-domains - completes, closes, and diverges from
    the full-mode trajectory."""
    full_state, _ = rt_full
    cfg = _gems_cfg(transport={
        "exchange_tau_h": 100.0,
        "domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}})
    state, _ = Engine(cfg).run(out_dir=str(tmp_path / "bc"))
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    assert state.time_h == full_state.time_h
    assert state.full_hash() != full_state.full_hash()
