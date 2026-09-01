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
                       "elements": ["S"]}
    cfg = TinnConfig.model_validate(raw)
    straight, summary = Engine(cfg).run(out_dir=str(tmp_path / "run"))
    s_idx = ELEMENT_IDS.index("S")
    assert straight.domain_sorbed_mol.shape[1] == len(ELEMENT_IDS)
    assert straight.domain_sorbed_mol[:, s_idx].sum() > 0.0
    m = summary["outputs"][-1]["ledger_metrics"]
    assert m["sorbed_total_mol"] > 0.0
    assert m["sorption_sites_mol"] > 0.0
    assert m["sorption_balance_max_mol"] == 0.0     # same-float transfer
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
