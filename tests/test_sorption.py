"""Tier 1 (RT-S1a) tests — PHREEQC SURFACE sorption operator (spec 3).

The operator is engine-detached at S1a: gates cover the null-operator
identity, the isotherm anchor (repeat calls pure-function + agreement with
a standalone PHREEQC batch), determinism/audit, and the config/ledger
plumbing. Tests needing IPhreeqc skip when phreeqpython is absent."""

import json
from pathlib import Path

import numpy as np
import pytest

from tinn import ledger
from tinn.config import SorptionConfig, TinnConfig
from tinn.registry import ELEMENT_IDS

REPO = Path(__file__).resolve().parents[1]
CEMDAT = REPO / "gems_bundles" / "PHREEQC-cemdata18" / "cemdata18.dat"
try:
    import phreeqpython  # noqa: F401
    _HAVE_IP = True
except ImportError:
    _HAVE_IP = False
needs_iphreeqc = pytest.mark.skipif(
    not (_HAVE_IP and CEMDAT.is_file()),
    reason="phreeqpython or the vendored cemdata18.dat not available")

# SO4 ligand exchange on a silanol-type site: solution loses SO4 (S 1,
# O 4) and gains one OH (O -1... net O 3, H -1). Demonstration constant.
SO4_RX = {"reaction": "Surf_sOH + SO4-2 = Surf_sSO4- + OH-",
          "log_k": 1.2,
          "sorbed_elements": {"S": 1.0, "O": 3.0, "H": -1.0}}


def _sorption_cfg(**over):
    raw = {"operator": "phreeqc_surface",
           "phreeqc_dat": str(CEMDAT),
           "site_density_mol_per_mol": {"CSHQ-TobH": 0.05,
                                        "CSHQ-JenH": 0.05},
           "surface_species": [dict(SO4_RX)],
           "elements": ["S"]}
    raw.update(over)
    return SorptionConfig.model_validate(raw)


def _solution():
    """A CH-buffered sulfate-bearing pore solution, element mols."""
    e = np.zeros(len(ELEMENT_IDS))

    def add(el, v):
        e[ELEMENT_IDS.index(el)] += v
    add("Ca", 2.0e-3)
    add("S", 5.0e-4)
    add("O", 2.0e-3 + 3 * 5.0e-4 + 1.0e-4)   # CaO + SO3 + O2 headroom
    add("H", 0.0)
    return e


@needs_iphreeqc
def test_sorption_null_operator_and_purity():
    """Zero sites / zero water / empty solution -> exact zeros (the null
    gate); repeated identical calls are bitwise identical (memo + pure
    function) and a fresh operator reproduces them (no hidden state)."""
    from tinn.backend import SorptionOperator
    op = SorptionOperator(_sorption_cfg(), 298.15)
    e = _solution()
    for kwargs in ({"sorbent_sites_mol": 0.0},
                   {"sorbent_sites_mol": 1e-4, "water_mol": 0.0}):
        r = op.sorb(e, kwargs.pop("water_mol", 2.0), **kwargs)
        assert r.status == "ok"
        assert np.all(r.sorbed_mol == 0.0) and r.site_occupancy == {}
    r0 = op.sorb(np.zeros(len(ELEMENT_IDS)), 2.0, 1e-4)
    assert np.all(r0.sorbed_mol == 0.0)

    a = op.sorb(e, 2.0, 2e-4)
    b = op.sorb(e, 2.0, 2e-4)                       # memo path
    assert np.array_equal(a.sorbed_mol, b.sorbed_mol)
    assert a.site_occupancy == b.site_occupancy
    op2 = SorptionOperator(_sorption_cfg(), 298.15)  # fresh instance
    c = op2.sorb(e, 2.0, 2e-4)
    assert np.array_equal(a.sorbed_mol, c.sorbed_mol)
    # something actually sorbed, only whitelisted solutes + O/H moved
    s_idx = ELEMENT_IDS.index("S")
    assert a.sorbed_mol[s_idx] > 0.0
    moved = {ELEMENT_IDS[i] for i in np.flatnonzero(a.sorbed_mol != 0.0)}
    assert moved <= {"S", "O", "H"}


@needs_iphreeqc
def test_sorption_isotherm_anchor():
    """The operator's partition equals a standalone PHREEQC batch of the
    same system (mol-level agreement), and more sites bind at least as
    much sulfate (monotone isotherm)."""
    from phreeqpython import PhreeqPython
    from tinn.backend import SorptionOperator, _decompose_to_reactants
    op = SorptionOperator(_sorption_cfg(), 298.15)
    e = _solution()
    water_mol = 2.0
    sites = 3e-4
    r = op.sorb(e, water_mol, sites)
    s_idx = ELEMENT_IDS.index("S")

    # standalone batch, built independently of the operator's plumbing
    pp = PhreeqPython(database=CEMDAT.name, database_directory=CEMDAT.parent)
    reactants = _decompose_to_reactants(
        {el: float(e[i]) for i, el in enumerate(ELEMENT_IDS) if e[i] > 0.0})
    water_kg = water_mol * 18.015 / 1000.0
    lines = ["SURFACE_MASTER_SPECIES", "    Surf_s Surf_sOH",
             "SURFACE_SPECIES", "    Surf_sOH = Surf_sOH", "        log_k 0",
             f"    {SO4_RX['reaction']}", f"        log_k {SO4_RX['log_k']}",
             "SELECTED_OUTPUT 1", "    -reset false",
             "    -high_precision true", "    -water true",
             "    -molalities Surf_sSO4-", "END",
             "SOLUTION 1", "    temp 25.0", f"    water {water_kg!r} kg",
             "REACTION 1"]
    lines += [f"    {f} {mol!r}" for f, mol in sorted(reactants.items())]
    lines += ["    1.0 moles",
              "SURFACE 1", f"    Surf_sOH {sites!r} 600.0 1.0",
              "    -no_edl", "END"]
    pp.ip.run_string("\n".join(lines))
    rows = pp.ip.get_selected_output_array()
    col = {str(h).strip(): i for i, h in enumerate(rows[0])}
    kgw = float(rows[-1][col["mass_H2O"]])
    bound = float(rows[-1][col["m_Surf_sSO4-(mol/kgw)"]]) * kgw
    assert r.site_occupancy["Surf_sSO4-"] == pytest.approx(bound, rel=1e-9)
    assert r.sorbed_mol[s_idx] == pytest.approx(bound, rel=1e-9)

    # monotone in sites, bounded by the offer and the site total
    lo = op.sorb(e, water_mol, sites / 4.0)
    assert lo.sorbed_mol[s_idx] <= r.sorbed_mol[s_idx] + 1e-18
    assert r.sorbed_mol[s_idx] <= min(e[s_idx], sites) * (1 + 1e-9)


def test_sorption_config_and_ledger():
    """Config-hash invisibility of the absent/null section, the validator
    refusal matrix, and the sorbed store entering element closure."""
    raw = json.loads((REPO / "examples" / "c3s_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"] = {"backend": "gems3k"}
    h0 = TinnConfig.model_validate(raw).config_hash()
    raw_null = dict(raw)
    raw_null["sorption"] = None
    assert TinnConfig.model_validate(raw_null).config_hash() == h0
    live = dict(raw)
    live["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.05},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"]}
    assert TinnConfig.model_validate(live).config_hash() != h0
    # refusal matrix
    from pydantic import ValidationError
    for patch in (
            {"elements": ["O"]},                       # O/H not whitelisted
            {"elements": ["S", "S"]},                  # duplicates
            {"elements": ["Na"]},                      # alkali w/o opt-in
            {"elements": ["Xx"]},                      # unknown element
            {"site_density_mol_per_mol": {}},          # empty density
            {"site_density_mol_per_mol": {"CSHQ-TobH": -1.0}},
            {"surface_species": []},
            {"surface_species": [{**SO4_RX,
                                  "sorbed_elements": {"Cl": 1.0}}]},
            {"surface_species": [{**SO4_RX, "reaction": "A = B"}]}):
        bad = dict(live)
        bad["sorption"] = {**live["sorption"], **patch}
        with pytest.raises(ValidationError):
            TinnConfig.model_validate(bad)
    stoich = dict(live)
    stoich["chemistry"] = {"backend": "stoichiometric"}
    with pytest.raises(ValidationError, match="gems3k"):
        TinnConfig.model_validate(stoich)
    # ledger: the sorbed store is part of current_elements, and the
    # same-float S balance check trips on a mismatched pair
    from tinn.engine import Engine
    from tinn.registry import default_registry
    cfg = TinnConfig.model_validate(
        json.loads((REPO / "examples" / "c3s_32.json").read_text(
            encoding="utf-8")))
    st = Engine(cfg).initial_state()
    reg = default_registry()
    base = ledger.current_elements(st, reg)
    st.domain_sorbed_mol = np.zeros((2, len(ELEMENT_IDS)))
    st.domain_sorbed_mol[0, ELEMENT_IDS.index("S")] = 1.5e-6
    cur = ledger.current_elements(st, reg)
    assert cur[ELEMENT_IDS.index("S")] - base[ELEMENT_IDS.index("S")] \
        == 1.5e-6
    good = ledger.SorptionBalance(
        applied_inventory_delta=np.full(len(ELEMENT_IDS), -1e-6),
        applied_sorbed_delta=np.full(len(ELEMENT_IDS), 1e-6),
        abs_scale=2e-6 * len(ELEMENT_IDS))
    st2 = Engine(cfg).initial_state()
    rep = ledger.check_all(st2, reg, sorption=good)
    assert "balance_sorption" not in rep.violations
    bad_bal = ledger.SorptionBalance(
        applied_inventory_delta=np.full(len(ELEMENT_IDS), -1e-6),
        applied_sorbed_delta=np.full(len(ELEMENT_IDS), 2e-6),
        abs_scale=3e-6 * len(ELEMENT_IDS))
    rep2 = ledger.check_all(st2, reg, sorption=bad_bal)
    assert "balance_sorption" in rep2.violations


# ------------------------------------------------- S1b: engine integration

GEMS_PYTHON = Path(r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe")
PC_BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
needs_gems = pytest.mark.skipif(
    not (PC_BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")


@needs_gems
@needs_iphreeqc
def test_sorption_engine_run_closes_and_restarts(tmp_path):
    """T->S->R spliced run: element closure holds WITH the sorbed store in
    current_elements (every accepted step passes check_all), sulfate
    actually binds, and a checkpoint restart is bit-identical."""
    import os
    from tinn.engine import Engine
    from tinn.registry import default_registry
    from tinn.storage import load_checkpoint
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(PC_BUNDLE)
    raw["chemistry"]["gems_worker_python"] = os.environ.get(
        "TINN_GEMS_PYTHON", str(GEMS_PYTHON))
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    raw["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {
                           "CSHQ-TobH": 0.02, "CSHQ-TobD": 0.02,
                           "CSHQ-JenH": 0.02, "CSHQ-JenD": 0.02},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"],
                       # RT-S1d: the CH buffer rides the same run so the
                       # closure gate covers its solution<->solids booking
                       "buffer_phase": "Portlandite"}
    cfg = TinnConfig.model_validate(raw)
    straight, summary = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    s_idx = ELEMENT_IDS.index("S")
    assert straight.domain_sorbed_mol.shape[1] == len(ELEMENT_IDS)
    assert straight.domain_sorbed_mol[:, s_idx].sum() > 0.0
    m = summary["outputs"][-1]["ledger_metrics"]
    assert m["sorbed_total_mol"] > 0.0
    assert m["sorption_sites_mol"] > 0.0
    assert m["sorption_balance_max_mol"] == 0.0     # same-float transfer
    # RT-D2: the fallback-magnitude reference scale rides every step
    assert len(m["aqueous_inventory_abs_mol_by_element"]) == len(ELEMENT_IDS)
    reg = default_registry()
    mid = load_checkpoint(str(tmp_path / "run" / "ckpt_000"), reg)
    assert mid.domain_sorbed_mol.shape[0] > 0        # store round-trips
    restarted, _ = Engine(mid.config).run(state=mid)
    assert restarted.full_hash() == straight.full_hash()


def test_sorption_ddl_config_gates():
    """RT-S2a config contract: ddl requires the physical area, no_edl
    refuses ddl-only fields (no silent ignore), the bound-species parser
    keeps charged names intact, and the new optional fields are hash-
    invisible when unset (pre-S2a configs keep their hash)."""
    from tinn.config import SurfaceReaction

    with pytest.raises(Exception, match="specific_area"):
        _sorption_cfg(surface_model="ddl")
    with pytest.raises(Exception, match="meaningless"):
        _sorption_cfg(specific_area_m2_per_mol_site=1.25e5)
    with pytest.raises(Exception, match="charging_reactions"):
        _sorption_cfg(charging_reactions=[
            {"reaction": "Surf_sOH = Surf_sO- + H+", "log_k": -9.8,
             "sorbed_elements": {"H": -1.0}}])
    cfg = _sorption_cfg(
        surface_model="ddl", specific_area_m2_per_mol_site=1.2544e5,
        charging_reactions=[
            {"reaction": "Surf_sOH = Surf_sO- + H+", "log_k": -9.8,
             "sorbed_elements": {"H": -1.0}},
            {"reaction": "Surf_sOH + Ca+2 = Surf_sOCa+ + H+",
             "log_k": -7.0,
             "sorbed_elements": {"Ca": 1.0, "H": -1.0}}],
        elements=["S", "Ca"])
    assert cfg.charging_reactions[1].bound_species == "Surf_sOCa+"
    rx = SurfaceReaction(reaction="Surf_sOH + SO4-2 = Surf_sSO4- + OH-",
                         log_k=0.5,
                         sorbed_elements={"S": 1.0, "O": 3.0, "H": -1.0})
    assert rx.bound_species == "Surf_sSO4-"

    base = json.loads((REPO / "examples" / "qualification"
                       / "deschner_opc_q32_dt06_28d.json"
                       ).read_text(encoding="utf-8"))
    base["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.05},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"]}
    h_bare = TinnConfig.model_validate(base).config_hash()
    base["sorption"]["charging_reactions"] = []
    base["sorption"]["specific_area_m2_per_mol_site"] = None
    h_explicit = TinnConfig.model_validate(base).config_hash()
    assert h_bare == h_explicit


@needs_iphreeqc
def test_sorption_ddl_operator_matches_standalone():
    """RT-S2a operator: under ddl the charging reactions ride the same
    bound-species machinery, the closure witness holds, and the result
    matches an independently assembled PHREEQC batch (rel 1e-9). The
    Gouy-Chapman surface suppresses anion binding vs no_edl (measured
    sign of the electrostatic correction)."""
    from phreeqpython import PhreeqPython
    from tinn.backend import SorptionOperator, _decompose_to_reactants

    area_per_mol = 1.2544e5
    charging = [{"reaction": "Surf_sOH = Surf_sO- + H+", "log_k": -9.8,
                 "sorbed_elements": {"H": -1.0}},
                {"reaction": "Surf_sOH + Ca+2 = Surf_sOCa+ + H+",
                 "log_k": -7.0,
                 "sorbed_elements": {"Ca": 1.0, "H": -1.0}}]
    cfg = _sorption_cfg(surface_model="ddl",
                        specific_area_m2_per_mol_site=area_per_mol,
                        charging_reactions=charging,
                        elements=["S", "Ca"])
    op = SorptionOperator(cfg, 298.15)
    e = _solution()
    water_mol, sites = 2.0, 3e-4
    r = op.sorb(e, water_mol, sites)
    s_idx = ELEMENT_IDS.index("S")
    assert r.sorbed_mol[s_idx] > 0.0
    assert "Surf_sO-" in r.site_occupancy          # charging punched too

    r0 = SorptionOperator(_sorption_cfg(), 298.15).sorb(e, water_mol,
                                                        sites)
    assert r.sorbed_mol[s_idx] < r0.sorbed_mol[s_idx]  # GC suppression

    pp = PhreeqPython(database=CEMDAT.name, database_directory=CEMDAT.parent)
    reactants = _decompose_to_reactants(
        {el: float(e[i]) for i, el in enumerate(ELEMENT_IDS) if e[i] > 0.0})
    lines = ["SURFACE_MASTER_SPECIES", "    Surf_s Surf_sOH",
             "SURFACE_SPECIES", "    Surf_sOH = Surf_sOH",
             "        log_k 0",
             f"    {SO4_RX['reaction']}",
             f"        log_k {SO4_RX['log_k']!r}"]
    for c in charging:
        lines += [f"    {c['reaction']}", f"        log_k {c['log_k']!r}"]
    lines += ["SELECTED_OUTPUT 1", "    -reset false",
              "    -high_precision true", "    -water true",
              "    -molalities Surf_sSO4-", "END",
              "SOLUTION 1", "    temp 25.0",
              f"    water {water_mol * 18.015 / 1000.0!r} kg",
              "REACTION 1"]
    lines += [f"    {f} {mol!r}" for f, mol in sorted(reactants.items())]
    lines += ["    1.0 moles", "SURFACE 1",
              f"    Surf_sOH {sites!r} {sites * area_per_mol!r} 1.0",
              "END"]
    pp.ip.run_string("\n".join(lines))
    rows = pp.ip.get_selected_output_array()
    col = {str(h).strip(): i for i, h in enumerate(rows[0])}
    kgw = float(rows[-1][col["mass_H2O"]])
    bound = float(rows[-1][col["m_Surf_sSO4-(mol/kgw)"]]) * kgw
    assert r.site_occupancy["Surf_sSO4-"] == pytest.approx(bound, rel=1e-9)


def test_sorption_wetness_dust_contract():
    """RT-S1c S-stage contract, pinned on the two MEASURED failure
    inputs: the femto-water dust cluster (water 5.6e-17 mol vs S
    7.3e-17 mol -> scaled to a ~7e4 mol/kg pseudo-solution, A(H2O)
    diverged) and the water-rich solute-noise pocket (broke the exact
    oxide decomposition at noise scale). Healthy pore solutions stay
    wet; water<=0 stays dry."""
    from tinn.engine import sorption_reactor_dry

    def offer(**mol):
        e = np.zeros(len(ELEMENT_IDS))
        for el, v in mol.items():
            e[ELEMENT_IDS.index(el)] = v
        return e

    floor = 1e-24 + 1e-12 * 1e-4          # ledger scale ~1e-4 mol
    # measured dust cluster (2026-09-02 run log): ratio ~0.76 -> dry
    dust = offer(S=7.336702161685963e-17, Ca=5.8087856419267095e-21,
                 K=1.7081002873427281e-19, O=2.2028302791834196e-16)
    assert sorption_reactor_dry(5.557966695806393e-17, dust, floor)
    # water-rich, solute at noise scale -> dry via the dust floor
    noise = offer(S=1e-19, Ca=1e-20)
    assert sorption_reactor_dry(3e-10, noise, floor)
    # healthy CH-buffered pore solution (ratio ~55) -> wet
    pore = offer(Ca=2e-6, S=5e-7, K=1e-6, O=1e-5)
    assert not sorption_reactor_dry(2e-4, pore, floor)
    assert not sorption_reactor_dry(2e-4, pore, floor, sites_mol=2e-3)
    # measured sulfate-exposure pocket: ~100 water molecules under 5e-14
    # mol of sites (ratio 2.7e5) passes the solute tests but is no
    # aqueous phase -> dry
    pocket = offer(S=1.0731403065660104e-23, Na=2.5764464160582435e-23,
                   Ca=3.977023222846777e-25, O=1e-22)
    assert sorption_reactor_dry(1.870032995508385e-19, pocket, 1e-24,
                                sites_mol=5.0473800668586434e-14)
    # water<=0 always dry; negative element dust never flips the sign
    assert sorption_reactor_dry(0.0, pore, floor)
    assert sorption_reactor_dry(-1.0, pore, floor)
    neg = offer(S=-1e-6, Ca=1e-9)
    assert sorption_reactor_dry(5e-9, neg, floor) in (True, False)


def test_sorption_nacl_carrier_decomposition():
    """RT-S2a/RT-Cl-2: chloride rides the NaCl carrier in the reactant
    decomposition (E3 precursor), then KCl, and any remainder goes as
    HCl whose proton the signed frame absorbs - exact closure round-trips
    in every branch (the carriers are neutral, so PHREEQC's totals are
    unchanged by the split)."""
    from tinn.backend import _decompose_to_reactants

    r = _decompose_to_reactants(
        {"Na": 0.55, "Cl": 0.5, "S": 0.015, "O": 0.11, "H": 0.05})
    assert r["NaCl"] == pytest.approx(0.5)
    assert r["Na2O"] == pytest.approx((0.55 - 0.5) / 2.0)
    assert r["SO3"] == pytest.approx(0.015)
    # NaOH-only solution with NaCl: exact O/H accounting, no O2 residue
    r2 = _decompose_to_reactants(
        {"Na": 0.075, "Cl": 0.05, "O": 0.025, "H": 0.025})
    assert r2["NaCl"] == pytest.approx(0.05)
    assert r2["H2O"] == pytest.approx(0.0125)
    assert "O2" not in r2
    r3 = _decompose_to_reactants(
        {"Cl": 0.5, "Na": 0.1, "K": 0.15, "O": 0.1, "H": 0.1})
    assert r3["NaCl"] == pytest.approx(0.1)
    assert r3["KCl"] == pytest.approx(0.15)
    assert r3["HCl"] == pytest.approx(0.25)
    assert r3["H2O"] == pytest.approx((0.1 - 0.25) / 2.0)   # signed frame
    assert r3["O2"] == pytest.approx((0.1 + 0.075) / 2.0)
    assert "Na2O" not in r3 and "K2O" not in r3


@needs_iphreeqc
def test_sorption_signed_oh_frame():
    """A re-offered ligand-exchange store can leave the offer's solute-
    frame H negative (the released OH- precipitated in the R stage) or O
    short of the oxide frame. Both encode EXACTLY as signed H2O/O2 terms
    (closure round-trips), the operator accepts such an offer, reports
    the frame terms, and its S witness still holds."""
    from tinn.backend import SorptionOperator, _decompose_to_reactants

    r = _decompose_to_reactants({"K": 1e-3, "S": 2e-3, "O": 6.5e-3,
                                 "H": -1.5e-3})
    assert r["H2O"] == pytest.approx(-7.5e-4)
    # O: 6.5e-3 - (K2O 0.5e-3 + SO3 6e-3) - (-7.5e-4) = 7.5e-4 -> O2 +3.75e-4
    assert r["O2"] == pytest.approx(3.75e-4)
    r2 = _decompose_to_reactants({"K": 1e-3, "O": 3e-4, "H": 2e-4})
    assert r2["O2"] == pytest.approx((3e-4 - 5e-4 - 1e-4) / 2.0)   # negative

    op = SorptionOperator(_sorption_cfg(), 298.15)
    e = _solution()
    e[ELEMENT_IDS.index("H")] = -2.0e-4          # OH- owed to the frame
    res = op.sorb(e, 2.0, 3e-4)
    assert res.status == "ok"
    assert res.site_occupancy["_frame_H2O_mol"] == pytest.approx(-1.0e-4)
    assert res.sorbed_mol[ELEMENT_IDS.index("S")] > 0.0


@needs_iphreeqc
def test_sorption_buffer_phase_desorbs_acid_reoffer():
    """RT-S1d: a re-offered store whose released base has gone to solids
    solves as an acid pseudo-solution and stays bound (S1-OPEN-2); with
    the reactor's portlandite as a buffer equilibrium phase the same offer
    desorbs, the buffer dissolves (delta booked, Ca released), and the
    S witness still closes. buffer_phase null keeps the config hash."""
    from tinn.backend import SorptionOperator

    # a base-FREE re-offer (the released OH- went to solids): K2O + SO3 in
    # excess of the base -> the closed pseudo-solution is acid and the
    # OH-releasing exchange is driven to saturation
    e = np.zeros(len(ELEMENT_IDS))
    h = ELEMENT_IDS.index("H")
    s_i, ca = ELEMENT_IDS.index("S"), ELEMENT_IDS.index("Ca")
    e[ELEMENT_IDS.index("K")] = 1.0e-4
    e[s_i] = 5.0e-4
    e[ELEMENT_IDS.index("O")] = 0.5e-4 + 3.0 * 5.0e-4
    e[h] = -2.0e-4
    sites = 3.0e-4
    plain = SorptionOperator(_sorption_cfg(), 298.15).sorb(e, 2.0, sites)
    buf_cfg = _sorption_cfg(buffer_phase="Portlandite")
    buffered = SorptionOperator(buf_cfg, 298.15).sorb(
        e, 2.0, sites, buffer_mol=5.0e-3)
    assert plain.sorbed_mol[s_i] > buffered.sorbed_mol[s_i]   # desorbs
    bd = buffered.buffer_delta_mol
    assert bd is not None and bd[ca] > 0.0                     # CH dissolved
    assert bd[h] == pytest.approx(2.0 * bd[ca]) and bd[ELEMENT_IDS.index("O")] == pytest.approx(2.0 * bd[ca])
    assert buffered.site_occupancy["_buffer_dissolved_mol"] == pytest.approx(bd[ca])
    with pytest.raises(ValueError, match="buffer_phase"):
        SorptionOperator(_sorption_cfg(), 298.15).sorb(e, 2.0, sites,
                                                       buffer_mol=1e-3)

    base = json.loads((REPO / "examples" / "qualification"
                       / "deschner_opc_q32_dt06_28d.json"
                       ).read_text(encoding="utf-8"))
    base["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.05},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"]}
    h0 = TinnConfig.model_validate(base).config_hash()
    base["sorption"]["buffer_phase"] = None
    assert TinnConfig.model_validate(base).config_hash() == h0
    base["sorption"]["buffer_phase"] = "Portlandite"
    assert TinnConfig.model_validate(base).config_hash() != h0


# ------------------------------------------------- RT-Cl-2: second site pool
CL_RX = {"reaction": "Surf_cOH + Cl- = Surf_cOHCl-", "log_k": 0.5,
         "sorbed_elements": {"Cl": 1.0}}


def test_sorption_second_pool_config_gates():
    """RT-Cl-2: the Surf_c pool is declared or it is not - a Surf_c
    reaction without its density map, a map without a Surf_c reaction, a
    reaction spanning both pools, and Surf_c under 'ddl' are refused; an
    unset pool keeps every earlier sorption config hash."""
    both = {"surface_species": [dict(SO4_RX), dict(CL_RX)],
            "elements": ["S", "Cl"]}
    with pytest.raises(ValueError, match="site_density_c_mol_per_mol"):
        _sorption_cfg(**both)
    with pytest.raises(ValueError, match="no reaction uses the Surf_c"):
        _sorption_cfg(site_density_c_mol_per_mol={"CSHQ-JenH": 0.1})
    with pytest.raises(ValueError, match="spans both"):
        _sorption_cfg(surface_species=[dict(SO4_RX), {
            "reaction": "Surf_cOH + Surf_sOH + Cl- = Surf_cOHCl- + Surf_sOH",
            "log_k": 0.0, "sorbed_elements": {"Cl": 1.0}}],
            elements=["S", "Cl"],
            site_density_c_mol_per_mol={"CSHQ-JenH": 0.1})
    with pytest.raises(ValueError, match="no_edl only"):
        _sorption_cfg(**both, site_density_c_mol_per_mol={"CSHQ-JenH": 0.1},
                      surface_model="ddl",
                      specific_area_m2_per_mol_site=1.254e5)
    cfg = _sorption_cfg(**both, site_density_c_mol_per_mol={"CSHQ-JenH": 0.1})
    assert [rx.site_pool for rx in cfg.surface_species] == ["s", "c"]
    assert cfg.surface_species[1].bound_species == "Surf_cOHCl-"
    base = json.loads((REPO / "examples" / "qualification"
                       / "deschner_opc_q32_dt06_28d.json"
                       ).read_text(encoding="utf-8"))
    base["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.05},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"]}
    h0 = TinnConfig.model_validate(base).config_hash()
    base["sorption"]["site_density_c_mol_per_mol"] = None
    assert TinnConfig.model_validate(base).config_hash() == h0


@needs_iphreeqc
def test_sorption_second_pool_operator_independent():
    """RT-Cl-2: Cl- binds on the Surf_c pool only - zero Surf_c sites sorb
    no Cl and reproduce the single-pool SO4 result on Surf_s; with sites
    the Cl uptake is bounded by the pool, the occupancy row books it, and
    the closure witness holds; sites for an undeclared pool are refused."""
    from tinn.backend import SorptionOperator
    cl, s_i = ELEMENT_IDS.index("Cl"), ELEMENT_IDS.index("S")
    e = _solution()
    e[ELEMENT_IDS.index("Na")] += 1.0e-3
    e[cl] += 1.0e-3
    two = SorptionOperator(_sorption_cfg(
        surface_species=[dict(SO4_RX), dict(CL_RX)], elements=["S", "Cl"],
        site_density_c_mol_per_mol={"CSHQ-JenH": 0.1}), 298.15)
    one = SorptionOperator(_sorption_cfg(), 298.15)
    r0 = two.sorb(e, 2.0, 3.0e-4, sites_c_mol=0.0)
    assert r0.sorbed_mol[cl] == 0.0
    assert r0.sorbed_mol[s_i] == pytest.approx(
        one.sorb(e, 2.0, 3.0e-4).sorbed_mol[s_i], rel=1e-6)
    r1 = two.sorb(e, 2.0, 3.0e-4, sites_c_mol=1.0e-4)
    assert 0.0 < r1.sorbed_mol[cl] <= 1.0e-4
    assert r1.site_occupancy["Surf_cOHCl-"] == pytest.approx(r1.sorbed_mol[cl])
    assert r1.sorbed_mol[s_i] == pytest.approx(r0.sorbed_mol[s_i], rel=5e-2)
    with pytest.raises(ValueError, match="Surf_c pool"):
        one.sorb(e, 2.0, 3.0e-4, sites_c_mol=1.0e-4)


def test_sorbed_store_folds_onto_wet_neighbours_on_cluster_death():
    """RT-Cl-3: a dead cluster's sorbed row folds onto the wet neighbours
    of its voxels (largest-liquid-contact label, voxel-count weighted,
    exact floats); with no wet neighbour anywhere the fold reports None
    (the caller keeps the hard reject)."""
    from tinn.engine import _fold_dead_sorbed_rows
    n = 4
    cl_labels = np.full((n, n, n), -1, dtype=np.int64)
    cl_labels[0] = 1                      # the dying cluster: whole z=0 plane
    cl_labels[2:] = 2
    new_labels = np.full((n, n, n), -1, dtype=np.int64)
    new_labels[1] = 0                     # wet neighbour plane (new domain 0)
    new_labels[2:] = 1                    # another wet domain, not adjacent
    liquid = np.zeros((n, n, n))
    liquid[1] = 0.5
    liquid[2:] = 1.0
    rows = np.zeros((1, len(ELEMENT_IDS)))
    rows[0, ELEMENT_IDS.index("Cl")] = 3.0e-3
    result = np.zeros((2, len(ELEMENT_IDS)))
    events = _fold_dead_sorbed_rows(rows, result, [0], np.array([1]),
                                    cl_labels, new_labels, liquid,
                                    (False, True, True))
    assert events == [(0, 0, float(n * n))]
    assert result[0, ELEMENT_IDS.index("Cl")] == 3.0e-3      # exact float
    assert result[1].sum() == 0.0
    dry = np.full((n, n, n), -1, dtype=np.int64)
    assert _fold_dead_sorbed_rows(rows, np.zeros_like(result), [0],
                                  np.array([1]), cl_labels, dry,
                                  np.zeros_like(liquid),
                                  (False, True, True)) is None


def test_surface_reaction_row_derived_from_equation():
    """RT-03 (review 2026-09-07): the sorbed_elements row is derived from
    the PHREEQC equation and must match O/H included; unbalanced or
    unparsable equations and rows outside the ledger are refused."""
    from tinn.config import SurfaceReaction, surface_reaction_removal
    assert surface_reaction_removal("Surf_sOH + SO4-2 = Surf_sSO4- + OH-") == {
        "S": 1.0, "O": 3.0, "H": -1.0}
    assert surface_reaction_removal("Surf_cOH + Cl- = Surf_cOHCl-") == {"Cl": 1.0}
    assert surface_reaction_removal("Surf_sOH = Surf_sO- + H+") == {"H": -1.0}
    assert surface_reaction_removal("Surf_sOH + Ca+2 = Surf_sOCa+ + H+") == {
        "Ca": 1.0, "H": -1.0}
    assert surface_reaction_removal("Surf_sOH + Al(OH)4- = Surf_sOAl(OH)3- + H2O") == {
        "Al": 1.0, "O": 3.0, "H": 2.0}
    with pytest.raises(ValueError, match="disagree with the equation"):
        SurfaceReaction(reaction="Surf_sOH + SO4-2 = Surf_sSO4- + OH-",
                        log_k=1.0, sorbed_elements={"S": 1.0})
    with pytest.raises(ValueError, match="not element-balanced"):
        surface_reaction_removal("Surf_sOH + SO4-2 = Surf_sSO4-")
    with pytest.raises(ValueError, match="not charge-balanced"):
        surface_reaction_removal("Surf_sOH + SO4-2 = Surf_sSO4- + OH")
    with pytest.raises(ValueError, match="outside the ledger"):
        surface_reaction_removal("Surf_sOH + Sr+2 = Surf_sOSr+ + H+")
    with pytest.raises(ValueError, match="cannot parse"):
        surface_reaction_removal("Surf_sOH + so4-2 = Surf_sSO4- + OH-")


def test_buffer_phase_refuses_per_phase_rate_limit():
    """RT-04B (review 2026-09-07): a per-phase exchange_tau on the buffer
    phase is refused at config level; a global tau stays engine-refused."""
    base = json.loads((REPO / "examples" / "qualification"
                       / "deschner_opc_q32_dt06_28d.json"
                       ).read_text(encoding="utf-8"))
    base["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.05},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"], "buffer_phase": "Portlandite"}
    base["transport"] = {"exchange_tau_h_per_phase": {"Portlandite": 100.0}}
    with pytest.raises(ValueError, match="exchange_tau_h_per_phase"):
        TinnConfig.model_validate(base)
    base["transport"] = {"exchange_tau_h_per_phase": {"ettringite": 100.0}}
    TinnConfig.model_validate(base)                     # other phases: fine


@needs_gems
@needs_iphreeqc
def test_structural_alkali_endmembers_refuse_alkali_sorption():
    """RT-04A (review 2026-09-07): the PC bundle's CSHQ carries KSiOH /
    NaSiOH; Na surface sorption on top is refused by the endmember
    element rows, not by a phase-name check."""
    import os
    from tinn.engine import Engine
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(PC_BUNDLE)
    raw["chemistry"]["gems_worker_python"] = os.environ.get(
        "TINN_GEMS_PYTHON", str(GEMS_PYTHON))
    raw["schedule"] = {"output_times_h": [2.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    raw["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {"CSHQ-TobH": 0.02},
                       "surface_species": [
                           {"reaction": "Surf_sOH + Na+ = Surf_sONa + H+",
                            "log_k": -10.0,
                            "sorbed_elements": {"Na": 1.0, "H": -1.0}}],
                       "elements": ["Na"], "alkali_exchange": True}
    with pytest.raises(RuntimeError, match="structurally"):
        Engine(TinnConfig.model_validate(raw))


@needs_gems
@needs_iphreeqc
def test_sorption_transient_failure_is_a_retried_reject(tmp_path):
    """RT-05 (review 2026-09-07): a PHREEQC transient failure in the S
    stage becomes a 'sorption_failure' trial reject (committed state kept,
    dt halved) instead of killing the run; the run then completes."""
    import os
    from tinn.backend import BackendTransientError
    from tinn.engine import Engine
    raw = json.loads((REPO / "examples" / "opc_gems_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(PC_BUNDLE)
    raw["chemistry"]["gems_worker_python"] = os.environ.get(
        "TINN_GEMS_PYTHON", str(GEMS_PYTHON))
    raw["schedule"] = {"output_times_h": [2.0, 4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    raw["transport"] = {"domains": {"tile_vox": 8, "d0_m2_s": 1.0e-9}}
    raw["sorption"] = {"operator": "phreeqc_surface",
                       "phreeqc_dat": str(CEMDAT),
                       "site_density_mol_per_mol": {
                           "CSHQ-TobH": 0.02, "CSHQ-TobD": 0.02,
                           "CSHQ-JenH": 0.02, "CSHQ-JenD": 0.02},
                       "surface_species": [dict(SO4_RX)],
                       "elements": ["S"]}
    eng = Engine(TinnConfig.model_validate(raw))
    real = eng._sorb_op.sorb
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BackendTransientError("injected PHREEQC failure")
        return real(*a, **k)

    eng._sorb_op.sorb = flaky
    state, _ = eng.run(out_dir=str(tmp_path / "run"))
    assert state.reject_counts.get("sorption_failure") == 1
    assert state.time_h == pytest.approx(4.0)
    assert calls["n"] > 1


def test_buffer_demand_from_store_delta():
    """RT-S1f: the CH transfer is sized by the store change's oxidation-
    state charge, not by the subsystem's own portlandite churn - ligand
    exchange precipitates half a CH per bound sulfate, proton-releasing
    Ca complexation dissolves half, chloride on a neutral site precipitates
    half, and a zero change books nothing."""
    from tinn.engine import buffer_demand_mol
    E = len(ELEMENT_IDS)
    idx = {el: ELEMENT_IDS.index(el) for el in ELEMENT_IDS}
    so4 = np.zeros(E); so4[idx["S"]] = 1.0; so4[idx["O"]] = 3.0; so4[idx["H"]] = -1.0
    assert buffer_demand_mol(2.0 * so4) == pytest.approx(-1.0)     # precipitates
    ca = np.zeros(E); ca[idx["Ca"]] = 1.0; ca[idx["H"]] = -1.0
    assert buffer_demand_mol(2.0 * ca) == pytest.approx(1.0)        # dissolves
    cl = np.zeros(E); cl[idx["Cl"]] = 1.0
    assert buffer_demand_mol(cl) == pytest.approx(-0.5)
    assert buffer_demand_mol(np.zeros(E)) == 0.0
    assert buffer_demand_mol(-2.0 * so4) == pytest.approx(1.0)      # desorption
