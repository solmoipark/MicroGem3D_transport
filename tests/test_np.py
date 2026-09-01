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
