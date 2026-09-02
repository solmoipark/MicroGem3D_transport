"""Tier 0 (RT-P0) tests — species NP zero-current projection (PRD 4.6.4).

P0a: the vendored dw table, the Stokes-Einstein correction, worker protocol
v2 speciation, the equilibrate_elements suppression filter, and the pure NP
kernel anchors (equal-D identity, Nernst-Hartley, clamp ladder). Kernel
tests run everywhere; worker tests skip without the xgems env."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import transport
from tinn.gems import GemsWorker, load_species_dw

REPO = Path(__file__).resolve().parents[1]
DW_JSON = REPO / "gems_bundles" / "species_dw" / "species_dw.json"
PC_BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
CNASH_BUNDLE = REPO / "gems_bundles" / "CNASH" / "Test-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))
needs_gems = pytest.mark.skipif(
    not (PC_BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")


# ------------------------------------------------------------- P0a: dw data

def test_dw_table_vendored_valid():
    table = load_species_dw(str(DW_JSON))
    assert len(table.dw) >= 50                    # phreeqc.dat carries 56
    # the PRD-cited single-D0 bracket is literally these two ions
    assert table.dw["Ca+2"] == pytest.approx(0.793e-9)
    assert table.dw["OH-"] == pytest.approx(5.27e-9)
    assert table.z["Ca+2"] == 2 and table.z["OH-"] == -1
    # aliases resolve and map identities to consistent charges
    for gems_dc, phq in table.aliases.items():
        assert phq in table.dw
    assert table.lookup("SiO2@") == (table.dw["H4SiO4"], 0.0)
    assert table.lookup("no_such_species") is None
    raw = json.loads(DW_JSON.read_text(encoding="utf-8"))
    assert raw["source_sha256"] and raw["temperature_C"] == 25.0


def test_stokes_einstein_factor():
    assert transport.stokes_einstein_factor(298.15) == 1.0
    f20 = transport.stokes_einstein_factor(293.15)
    f60 = transport.stokes_einstein_factor(333.15)
    assert f20 < 1.0 < f60                        # monotone with T
    # 60 C: (333.15/298.15) * (0.8900/0.4660) within the table's resolution
    assert f60 == pytest.approx((333.15 / 298.15) * (0.8900 / 0.4660),
                                rel=0.02)
    with pytest.raises(ValueError):
        transport.stokes_einstein_factor(272.0)


# --------------------------------------------------- P0a: pure NP kernel

def _two_domain_graph(g=2.0, w=(4.0, 4.0)):
    return transport.DomainGraph(
        n_domains=2, edge_a=np.array([0]), edge_b=np.array([1]),
        edge_g=np.array([float(g)]), water=np.asarray(w, dtype=float),
        dust=np.zeros(2, dtype=bool))


def test_np_equal_d_identity():
    """Anchor 1 (PRD 4.6.4): all species at the same D collapse to exactly
    that scalar on every element — Phi vanishes by electroneutrality."""
    rng = np.random.default_rng(7)
    d0 = 1.7e-3                                    # vox^2/h, arbitrary
    z = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    nu = rng.integers(0, 3, size=(5, 4)).astype(float)
    graph = _two_domain_graph()
    # electroneutral domain states (charge balances within each domain)
    a = np.array([3.0, 1.0, 0.5, 1.5, 0.7])       # z.a = 3-1+1-3 = 0
    b = np.array([1.0, 3.0, 1.5, 0.5, 0.2])       # z.b = 1-3+3-1 = 0
    species = np.stack([a, b])
    npc = transport.np_effective_conductance(
        graph, species, graph.water, np.full(5, d0), z, nu)
    el_present = (species @ nu).max(axis=0) > 0.0
    assert np.allclose(npc.deff_edge[0][el_present], d0, rtol=1e-12)
    assert npc.t_edge[0][el_present] == pytest.approx(
        graph.edge_g[0] * d0, rel=1e-12)
    assert sum(npc.counts.values()) - npc.counts["np_smalldc"] == 0
    assert npc.charge_flux_rel_max <= 1e-12


def test_np_nernst_hartley():
    """Anchor 2: a binary 1:1 electrolyte reproduces the Nernst-Hartley salt
    diffusivity D = (z+ - z-) D+ D- / (z+ D+ - z- D-) exactly."""
    table = load_species_dw(str(DW_JSON))
    for cat, an in (("Na+", "Cl-"), ("K+", "Cl-")):
        dp, dm = table.dw[cat], table.dw[an]
        expect = 2.0 * dp * dm / (dp + dm)
        z = np.array([1.0, -1.0])
        nu = np.array([[1.0, 0.0], [0.0, 1.0]])   # element per ion
        graph = _two_domain_graph()
        species = np.array([[2.0, 2.0], [0.5, 0.5]])   # electroneutral rows
        npc = transport.np_effective_conductance(
            graph, species, graph.water, np.array([dp, dm]), z, nu)
        assert np.allclose(npc.deff_edge[0], expect, rtol=1e-10)
    # literature cross-check (25 C): NaCl ~1.61e-9, KCl ~1.99e-9 m2/s
    dna, dk, dcl = (table.dw["Na+"], table.dw["K+"], table.dw["Cl-"])
    assert 2 * dna * dcl / (dna + dcl) == pytest.approx(1.61e-9, rel=0.02)
    assert 2 * dk * dcl / (dk + dcl) == pytest.approx(1.99e-9, rel=0.02)


def test_np_clamp_ladder():
    """Every rung fires where designed, is counted, and the result stays in
    [0, D_max] (the M-matrix guarantee)."""
    graph = _two_domain_graph()
    # H+ / Cl- / K+ : a strong H-driven field drags the trace K+ AGAINST its
    # own (small, positive) gradient -> raw quotient negative -> rung 3
    # (drop Phi, pure Fick) restores D_K exactly
    z = np.array([1.0, -1.0, 1.0])
    dw = np.array([9.31e-9, 2.03e-9, 1.96e-9])
    nu = np.eye(3)
    species = np.array([[10.0, 10.5, 0.5], [1.0, 1.45, 0.45]])
    npc = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu)
    assert npc.counts["np_phi_clamped"] >= 1
    assert npc.deff_edge[0, 2] == pytest.approx(1.96e-9, rel=1e-12)
    assert np.all(npc.deff_edge >= 0.0)
    assert np.all(npc.deff_edge <= dw.max() * (1 + 1e-12))
    assert npc.charge_flux_rel_max <= 1e-12
    # small-dc (rung 1): near-identical element state -> weighted-Fick D-bar
    sp2 = np.array([[2.0, 2.0, 1.0], [2.0 + 1e-15, 2.0 + 1e-15, 1.0]])
    npc2 = transport.np_effective_conductance(
        graph, sp2, graph.water, dw, z, nu)
    assert npc2.counts["np_smalldc"] >= 2
    # empty element (no carrier species anywhere) -> conductance 0, counted
    nu3 = np.hstack([nu, np.zeros((3, 1))])
    npc3 = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu3)
    assert npc3.counts["np_empty_el"] >= 1
    assert np.all(npc3.t_edge[:, 3] == 0.0)
    # rung 4 (Fick still negative) and rung 5 (cap): driven directly through
    # the kernel primitive with an element driving force DECOUPLED from the
    # species collapse — exactly the bath call's degree of freedom
    dc_s = np.array([[1.0, 0.0, 0.0]])
    cbar = np.array([[0.5, 0.5, 0.0]])
    d4, c4, _ = transport._np_deff(
        dc_s, cbar, np.array([[-1.0, 1.0, 1.0]]), np.ones((1, 3)),
        dw, z, np.eye(3))
    assert c4["np_fick_clamped"] >= 1          # s1/dc_el < 0 -> D-bar
    d5, c5, _ = transport._np_deff(
        dc_s, cbar, np.array([[1e-3, 1.0, 1.0]]), np.ones((1, 3)),
        dw, z, np.eye(3))
    assert c5["np_cap_clamped"] >= 1           # quotient blows past D_max
    assert d5[0, 0] <= dw.max() * (1 + 1e-12)


# ------------------------------------------------- P0a: worker protocol v2

@needs_gems
def test_worker_speciation_closure():
    w = GemsWorker(str(PC_BUNDLE), python_executable=str(GEMS_PYTHON))
    try:
        info = w.info()
        assert info["protocol_version"] >= 2
        assert info["species_charge"]["Ca+2"] == 2.0
        assert info["species_charge"]["OH-"] == -1.0
        n_c3s = 1.0 / 228.3145
        n_w = 0.5 / 18.015
        r = w.equilibrate_elements(
            {"Ca": 3 * n_c3s, "Si": n_c3s, "O": 5 * n_c3s + n_w + 1e-7,
             "H": 2 * n_w}, 293.15)
        assert r.aqueous_species_mol, "protocol v2 must carry speciation"
        # element closure: sum species x DCH rows == aqueous phase elements
        se = info["species_elements"]
        aq_el = r.phase_elements_mol["aq_gen"]
        for el in ("Ca", "Si"):
            total = sum(m * se[dc].get(el, 0.0)
                        for dc, m in r.aqueous_species_mol.items())
            assert total == pytest.approx(aq_el[el], rel=1e-9)
        # electroneutrality of the reported speciation
        zq = sum(m * info["species_charge"][dc]
                 for dc, m in r.aqueous_species_mol.items())
        scale = sum(abs(m * info["species_charge"][dc])
                    for dc, m in r.aqueous_species_mol.items())
        assert abs(zq) <= 1e-9 * max(scale, 1e-30)
    finally:
        w.close()


@needs_gems
@pytest.mark.skipif(not CNASH_BUNDLE.is_file(), reason="CNASH bundle absent")
def test_equilibrate_elements_filters_suppression():
    """Regression (measured 2026-09-01): the DEFAULT clinker suppression list
    must be filtered to declared phases — a hydrates-only bundle used to
    hard-error on every 0D call. An explicit unknown name still errors."""
    from tinn.gems import GemsError, O2_SEED_MOL_O
    w = GemsWorker(str(CNASH_BUNDLE), python_executable=str(GEMS_PYTHON))
    try:
        n_c3s = 1.0 / 228.3145
        n_w = 0.5 / 18.015
        el = {"Ca": 3 * n_c3s, "Si": n_c3s,
              "O": 5 * n_c3s + n_w + 1e4 * O2_SEED_MOL_O, "H": 2 * n_w}
        r = w.equilibrate_elements(el, 293.15)      # default list: filtered
        assert r.status
        with pytest.raises(GemsError):              # explicit list: typo guard
            w.equilibrate_elements(el, 293.15,
                                   suppressed_phases=("Alite",))
    finally:
        w.close()


# ------------------------------------------------- P0b: solver integration

def test_np_charge_zero_and_element_closure():
    """Anchors 3+4 through the promoted solver: the per-element BE with NP
    conductances keeps exact flux antisymmetry (delta sums to zero per
    element from the same floats) and the kernel's zero-current witness
    stays at rounding level."""
    table = load_species_dw(str(DW_JSON))
    graph = _two_domain_graph(g=1.5, w=(3.0, 5.0))
    z = np.array([1.0, -1.0, 2.0, -2.0])
    dw = np.array([table.dw["Na+"], table.dw["Cl-"],
                   table.dw["Ca+2"], table.dw["SO4-2"]]) * 3.6e15  # vox^2/h-ish
    nu = np.eye(4)
    species = np.array([[2.0, 1.6, 0.3, 0.5], [0.5, 0.9, 0.5, 0.3]])
    npc = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu)
    assert npc.charge_flux_rel_max <= 1e-12
    inv = np.array([[4.0, 3.0, 1.0, 1.5], [1.0, 2.0, 2.0, 0.5]])
    ex = transport.exchange_be(graph, inv, 0.25, None, np_cond=npc)
    assert ex.status == "ok"
    total = ex.delta.sum(axis=0)
    assert np.all(total == 0.0)                      # same-float antisymmetry
    assert np.array_equal(ex.delta[0], -ex.delta[1])
    assert np.all(inv + ex.delta >= 0.0)


def test_np_scalar_path_bitwise_and_hash():
    """The promoted solver fed a broadcast-scalar conductance is BITWISE
    equal to the scalar path (each column sees the same float sequence);
    an active species block changes the config hash."""
    d0 = 2.4e-3
    graph = transport.DomainGraph(
        n_domains=4, edge_a=np.array([0, 1, 2]), edge_b=np.array([1, 2, 3]),
        edge_g=np.array([1.0, 2.0, 0.5]),
        water=np.array([4.0, 2.0, 3.0, 5.0]), dust=np.zeros(4, dtype=bool))
    rng = np.random.default_rng(11)
    inv = rng.uniform(0.0, 2.0, size=(4, 6))
    a = transport.exchange_be(graph, inv.copy(), 0.5, d0)
    n_elem = inv.shape[1]
    npc = transport.NPConductance(
        t_edge=np.tile((d0 * graph.edge_g)[:, None], (1, n_elem)),
        t_bnd=None, deff_edge=np.full((3, n_elem), d0))
    b = transport.exchange_be(graph, inv.copy(), 0.5, None, np_cond=npc)
    assert np.array_equal(a.delta, b.delta)
    assert a.cg_iterations == b.cg_iterations
    # exactly one conductance source
    with pytest.raises(ValueError):
        transport.exchange_be(graph, inv.copy(), 0.5, d0, np_cond=npc)
    with pytest.raises(ValueError):
        transport.exchange_be(graph, inv.copy(), 0.5, None)
    # config hash: active species differs from the scalar config
    from tinn.config import TinnConfig
    raw = json.loads((REPO / "examples" / "c3s_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"] = {"backend": "gems3k"}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    h_scalar = TinnConfig.model_validate(raw).config_hash()
    raw["transport"]["domains"] = {
        "tile_vox": 8,
        "species": {"dw_table": str(DW_JSON), "default_dw_m2_s": 1.0e-9,
                    "geometry_factor": 1.0}}
    h_np = TinnConfig.model_validate(raw).config_hash()
    assert h_np != h_scalar


@needs_gems
def test_np_engine_restart_bit_identity(tmp_path):
    """Species-active mode-C run: restart equals straight-through bit for
    bit (frozen speciation included), the reserved sorbed array round-trips
    as (0, E), and a v4-stamped checkpoint is refused."""
    from tinn.config import TinnConfig
    from tinn.engine import Engine
    from tinn.registry import ELEMENT_IDS, default_registry
    from tinn.storage import (FORMAT_VERSION, StorageError, load_checkpoint,
                              save_checkpoint)
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(PC_BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {
        "tile_vox": 8,
        "species": {"dw_table": str(DW_JSON), "default_dw_m2_s": 1.0e-9,
                    "geometry_factor": 1.0}}}
    cfg = TinnConfig.model_validate(raw)
    straight, _ = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    assert straight.aq_species_ids            # stamped, header-owned
    assert straight.domain_species_mol.shape[0] > 0
    assert straight.domain_sorbed_mol.shape == (0, len(ELEMENT_IDS))
    reg = default_registry()
    mid = load_checkpoint(str(tmp_path / "run" / "ckpt_000"), reg)
    assert tuple(mid.aq_species_ids) == tuple(straight.aq_species_ids)
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.full_hash() == straight.full_hash()
    # v4-stamped checkpoint refused (extended message)
    save_checkpoint(straight, str(tmp_path), "v4ish")
    hdr = tmp_path / "v4ish" / "header.json"
    payload = json.loads(hdr.read_text(encoding="utf-8"))
    payload["format_version"] = 4
    hdr.write_text(json.dumps(payload), encoding="utf-8")
    man = tmp_path / "v4ish" / "manifest.json"
    import hashlib as _hl
    manifest = json.loads(man.read_text(encoding="utf-8"))
    manifest["header.json"] = _hl.sha256(hdr.read_bytes()).hexdigest()
    man.write_text(json.dumps(manifest), encoding="utf-8")
    assert FORMAT_VERSION == 5
    with pytest.raises(StorageError, match="species transport"):
        load_checkpoint(str(tmp_path / "v4ish"), reg)


# ------------------------------------------------ P0d: speciated solute bath

def test_np_bath_species_identity_and_harmonic():
    """RT-P0d kernel contract: an all-zero bath species vector reproduces
    the aerated-water path bitwise; a bath carrying the SAME species state
    as the domain drives zero element difference (the small-dc rung, D-bar);
    a solute bath keeps 0 <= t_bnd <= g * D_max (M-matrix)."""
    rng = np.random.default_rng(11)
    z = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    nu = rng.integers(0, 3, size=(5, 4)).astype(float)
    dw = np.array([1.3e-3, 2.0e-3, 0.8e-3, 1.1e-3, 1.7e-3])
    graph = _two_domain_graph()
    a = np.array([3.0, 1.0, 0.5, 1.5, 0.7])
    b = np.array([1.0, 3.0, 1.5, 0.5, 0.2])
    species = np.stack([a, b])
    g_bnd = np.array([1.5, 0.0])
    c_res = np.zeros(4)
    ref = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu, g_bnd=g_bnd, c_res=c_res)
    zero = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu, g_bnd=g_bnd, c_res=c_res,
        c_res_species=np.zeros(5))
    assert np.array_equal(ref.t_bnd, zero.t_bnd)            # bitwise
    assert ref.counts == zero.counts
    # bath == domain 0 state (per vox^3 of liquid): no driving force
    s_res = a / graph.water[0]
    same = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu, g_bnd=g_bnd,
        c_res=s_res @ nu, c_res_species=s_res)
    assert same.counts["np_smalldc"] >= ref.counts["np_smalldc"]
    assert np.all(np.isfinite(same.t_bnd))
    # a distinct electroneutral solute bath: bounded transmissibility
    s_res = np.array([2.0, 0.0, 0.0, 1.0, 0.3]) / graph.water[0]   # z.s = 0
    sol = transport.np_effective_conductance(
        graph, species, graph.water, dw, z, nu, g_bnd=g_bnd,
        c_res=s_res @ nu, c_res_species=s_res)
    assert np.all(sol.t_bnd >= 0.0)
    assert np.all(sol.t_bnd[0] <= g_bnd[0] * dw.max() * (1 + 1e-12))
    assert np.all(sol.t_bnd[1] == 0.0)                     # uncoupled row


@needs_gems
def test_np_engine_speciates_solute_bath():
    """RT-P0d engine contract: a Na2SO4 reservoir with species transport
    is speciated once at init (frozen, electroneutral, provenance-
    recorded) in the run's aqueous species order; the summed species
    reproduce the declared element composition per vox^3."""
    from tinn.config import TinnConfig
    from tinn.engine import Engine
    from tinn.registry import ELEMENT_IDS
    raw = json.loads((REPO / "examples" / "qualification"
                      / "leach_w4_opc32.json").read_text(encoding="utf-8"))
    raw["transport"]["domains"].pop("d0_m2_s", None)
    raw["transport"]["domains"]["species"] = {
        "dw_table": "gems_bundles/species_dw/species_dw.json",
        "default_dw_m2_s": 1.0e-9, "geometry_factor": 1.0}
    raw["transport"]["boundary"]["composition_mol_per_m3"] = {
        "Na": 704.0, "S": 352.0, "O": 1408.5}
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    vec = eng._np_bath_species_vox
    assert vec is not None and vec.shape == (len(eng.backend.aq_species_ids),)
    vox_m3 = (cfg.rve.voxel_size_um * 1e-6) ** 3
    el = vec @ np.asarray(eng.backend.aq_species_elements, dtype=float)
    for name, target in (("Na", 704.0), ("S", 352.0)):
        assert el[ELEMENT_IDS.index(name)] == pytest.approx(target * vox_m3,
                                                            rel=1e-6)
    assert abs(float(np.dot(eng._np_z, vec))) <= 1e-8 * np.abs(vec).sum()
    prov = eng._np_provenance
    assert prov["bath_species_mol_per_m3"] and 6.0 < prov["bath_ph"] < 9.0
