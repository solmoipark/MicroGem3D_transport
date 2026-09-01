"""0D cross-check spike: xGEMS (CNASH/PC bundle) vs PHREEQC (Cemdata18 export).

Purpose (spike, NOT a module — PRD module cap untouched): decide whether the
official Cemdata18 PHREEQC export reproduces the GEMS bundles' equilibrium
chemistry inside THIS project's composition window. Three cases span the
window: CH-buffered OPC, post-CH (silica fume blend), and the E3
sulfate/alkali channel; a fourth high-alkali stress case probes where the
aqueous activity models are expected to diverge.

Both engines receive the SAME authoritative element-mol vector (the ledger
currency). The PHREEQC side decomposes it into neutral oxide/H2O/O2 reactants
(exact closure asserted), equilibrates against a curated candidate phase list
plus an ideal C-S-H solid solution (CSHQ or CNASH endmembers, both straight
from the .dat), and audits element closure by re-summing phase formulas parsed
from the .dat itself — nothing hand-copied.

Engine provenance:
  - GEMS side: tinn.gems.GemsWorker.equilibrate_elements (needs the xgems
    interpreter, i.e. the user's machine; TINN_GEMS_PYTHON or --gems-python).
  - PHREEQC side: phreeqpython (pip install phreeqpython), database
    gems_bundles/PHREEQC-cemdata18/cemdata18.dat — the official Empa/PSI
    ThermoMatch export of CEMDATA18 (see PROVENANCE.md next to the .dat).

Usage (user machine, both engines):
    py -3 scripts/spike_phreeqc_crosscheck.py --engines phreeqc,gems --out runs/spike_crosscheck
Cloud / no xgems (PHREEQC only):
    python3 scripts/spike_phreeqc_crosscheck.py --engines phreeqc --out runs/spike_crosscheck
Then compare (merges whatever result files exist in --out):
    py -3 scripts/spike_phreeqc_crosscheck.py --compare runs/spike_crosscheck
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.registry import ELEMENT_IDS, default_registry, element_vector  # noqa: E402

TEMPERATURE_K = 298.15
# relative O2 seed for the GEMS cold start (see run_gems_case)
O2_SEED_REL = 1e-8
H2O_G_MOL = 18.01528
DEFAULT_DAT = REPO / "gems_bundles" / "PHREEQC-cemdata18" / "cemdata18.dat"
DEFAULT_BUNDLE = "gems_bundles/CNASH/Test-dat.lst"

# ---------------------------------------------------------------------------
# Cases: released phase mols per 100 g binder basis + free water.
# Amounts are built from the registry's own formulas via element_vector, so
# the element accounting is identical to the pipeline's ledger arithmetic.
# ---------------------------------------------------------------------------

_REG = default_registry()


def _phase_mol(phase_id: str, grams: float) -> float:
    return grams / _REG.get(phase_id).molar_mass_g_mol


def _build_cases() -> Dict[str, Dict]:
    # nominal OPC mineralogy (wt% of 100 g binder), gypsum interground
    opc = {"C3S": 65.0, "C2S": 15.0, "C3A": 8.0, "C4AF": 8.0, "gypsum": 4.0}
    cases: Dict[str, Dict] = {}

    # A — CH-buffered: OPC at uniform alpha 0.5, gypsum fully offered,
    # w/b 0.5. CH present and buffering; the everyday regime.
    a_released = {p: _phase_mol(p, g) * 0.5 for p, g in opc.items() if p != "gypsum"}
    a_released["gypsum"] = _phase_mol("gypsum", opc["gypsum"])
    cases["ch_buffered_opc"] = {
        "description": "OPC alpha=0.5, gypsum fully offered, w/b 0.5 — CH buffered",
        "released_mol": a_released, "water_g": 50.0}

    # B — post-CH: 70 g OPC (alpha 0.7) + 30 g silica fume fully released,
    # w/b 0.5 on total binder. Pozzolanic Si excess exhausts CH; C-S-H
    # drops to low Ca/Si. Probes the C-S-H model far from the CH buffer.
    b_released = {p: _phase_mol(p, g * 0.70) * 0.7
                  for p, g in opc.items() if p != "gypsum"}
    b_released["gypsum"] = _phase_mol("gypsum", opc["gypsum"] * 0.70)
    b_released["silica_fume"] = _phase_mol("silica_fume", 30.0)
    cases["post_ch_sf_blend"] = {
        "description": "70 g OPC (alpha=0.7) + 30 g silica fume, w/b 0.5 — CH exhausted",
        "released_mol": b_released, "water_g": 50.0}

    # C — E3 sulfate/alkali channel: case A + arcanite 0.6 g + thenardite
    # 0.4 g fully dissolved (mirrors examples/opc_gypsum_cnash_32.json).
    c_released = dict(a_released)
    c_released["arcanite"] = _phase_mol("arcanite", 0.6)
    c_released["thenardite"] = _phase_mol("thenardite", 0.4)
    cases["e3_alkali_sulfate"] = {
        "description": "case A + arcanite 0.6 g + thenardite 0.4 g — E3 channel",
        "released_mol": c_released, "water_g": 50.0}

    # D — high-alkali stress: case A + 0.5 mol NaOH per kg water. Outside
    # normal OPC, inside AAM territory: where extended-DH parameterisations
    # of the two codes are EXPECTED to diverge. Diagnostic only.
    d_released = dict(a_released)
    cases["high_alkali_stress"] = {
        "description": "case A + 0.5 mol/kgw NaOH — activity-model stress case",
        "released_mol": d_released, "water_g": 50.0,
        "extra_element_mol": {"Na": 0.025, "O": 0.025, "H": 0.025}}
    return cases


def case_elements(case: Dict) -> Dict[str, float]:
    """Authoritative element-mol vector: released phases + free water (+extras)."""
    vec = element_vector({"H": 2, "O": 1}, case["water_g"] / H2O_G_MOL)
    for phase_id, mol in case["released_mol"].items():
        vec = vec + element_vector(_REG.get(phase_id).formula, mol)
    out = {el: float(v) for el, v in zip(ELEMENT_IDS, vec)}
    for el, v in case.get("extra_element_mol", {}).items():
        out[el] = out.get(el, 0.0) + v
    return out


# ---------------------------------------------------------------------------
# Element vector -> neutral reactants for PHREEQC (exact closure asserted)
# ---------------------------------------------------------------------------

_OXIDES: Tuple[Tuple[str, str, float, float], ...] = (
    # (reactant formula, element, element per formula, O per formula)
    ("CaO", "Ca", 1.0, 1.0), ("SiO2", "Si", 1.0, 2.0),
    ("Al2O3", "Al", 2.0, 3.0), ("Fe2O3", "Fe", 2.0, 3.0),
    ("SO3", "S", 1.0, 3.0), ("Na2O", "Na", 2.0, 1.0),
    ("K2O", "K", 2.0, 1.0), ("MgO", "Mg", 1.0, 1.0), ("CO2", "C", 1.0, 2.0),
)


def decompose_to_reactants(elements: Dict[str, float]) -> Dict[str, float]:
    """elements -> {reactant formula: mol} with H2O as solvent-forming water
    and any leftover O as O2. Raises if the vector is not oxide-decomposable
    (it always is for ledger inputs built from neutral formula units)."""
    reactants: Dict[str, float] = {}
    o_used = 0.0
    for formula, el, n_el, n_o in _OXIDES:
        amount = elements.get(el, 0.0)
        if amount <= 0.0:
            continue
        mol = amount / n_el
        reactants[formula] = mol
        o_used += mol * n_o
    h = elements.get("H", 0.0)
    if h < -1e-15:
        raise ValueError(f"negative H in input: {h}")
    h2o = h / 2.0
    o_left = elements.get("O", 0.0) - o_used - h2o
    scale = max(abs(v) for v in elements.values()) or 1.0
    if o_left < -1e-9 * scale:
        raise ValueError(
            f"element vector is not oxide-decomposable: O deficit {o_left} mol")
    reactants["H2O"] = h2o
    if o_left > 1e-15 * scale:
        reactants["O2"] = o_left / 2.0
    # exact closure audit: rebuild the vector from the reactants
    rebuilt = {el: 0.0 for el in elements}
    for formula, el, n_el, n_o in _OXIDES:
        mol = reactants.get(formula, 0.0)
        rebuilt[el] = rebuilt.get(el, 0.0) + mol * n_el
        rebuilt["O"] = rebuilt.get("O", 0.0) + mol * n_o
    rebuilt["H"] = rebuilt.get("H", 0.0) + 2.0 * reactants["H2O"]
    rebuilt["O"] = rebuilt.get("O", 0.0) + reactants["H2O"] \
        + 2.0 * reactants.get("O2", 0.0)
    for el, target in elements.items():
        if abs(rebuilt.get(el, 0.0) - target) > 1e-9 * scale + 1e-15:
            raise AssertionError(
                f"reactant decomposition broke {el}: {rebuilt.get(el)} != {target}")
    return reactants


# ---------------------------------------------------------------------------
# Parse phase formulas straight out of the .dat (nothing hand-copied)
# ---------------------------------------------------------------------------

_NUM = re.compile(r"[0-9]*\.?[0-9]+")
_SYM = re.compile(r"[A-Z][a-z]?")


def _formula_to_elements(token: str) -> Dict[str, float]:
    """Element mols of one formula-unit token. Handles arbitrary nesting with
    optional multipliers: ((CaO)1.25(SiO2)1(H2O)2.75)0.6667, Ca4Al2Fe2O10,
    ((H2O)2)Ca6Al2(SO4)3(OH)12(H2O)24 ... (recursive descent)."""
    token = token.strip()
    pos = 0

    def parse_seq(stop_at_close: bool) -> Dict[str, float]:
        nonlocal pos
        out: Dict[str, float] = {}

        def add(sub: Dict[str, float], factor: float) -> None:
            for el, n in sub.items():
                out[el] = out.get(el, 0.0) + n * factor

        while pos < len(token):
            ch = token[pos]
            if ch == ")":
                if stop_at_close:
                    return out
                raise ValueError(f"unbalanced ')' in {token!r}")
            if ch == "(":
                pos += 1
                inner = parse_seq(True)
                if pos >= len(token) or token[pos] != ")":
                    raise ValueError(f"unbalanced '(' in {token!r}")
                pos += 1
                mnum = _NUM.match(token, pos)
                factor = 1.0
                if mnum:
                    factor = float(mnum.group())
                    pos = mnum.end()
                add(inner, factor)
                continue
            msym = _SYM.match(token, pos)
            if msym:
                pos = msym.end()
                mnum = _NUM.match(token, pos)
                count = 1.0
                if mnum:
                    count = float(mnum.group())
                    pos = mnum.end()
                add({msym.group(): 1.0}, count)
                continue
            raise ValueError(f"cannot parse {token!r} at position {pos}")
        return out

    return parse_seq(False)


def parse_dat_phase_formulas(dat_path: Path) -> Dict[str, Dict[str, float]]:
    """{phase name: element mols per formula unit} from the PHASES block."""
    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    phases: Dict[str, Dict[str, float]] = {}
    in_phases = False
    name: Optional[str] = None
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.upper().startswith("PHASES"):
            in_phases = True
            continue
        if in_phases and line[0] not in " \t" and (
                stripped == "END" or (stripped.isupper() and "_" in stripped)):
            # END or the next KEYWORD block ends PHASES
            in_phases = False
            continue
        if not in_phases:
            continue
        if line[0] not in " \t":
            name = stripped
            continue
        if name and "=" in stripped and not stripped.startswith("-"):
            lhs = stripped.split("=", 1)[0].split()[0]
            try:
                phases[name] = _formula_to_elements(lhs)
            except Exception as exc:  # keep going; unparsed phases are reported
                phases[name] = {"__parse_error__": str(exc)}  # type: ignore
            name = None
    return phases


# ---------------------------------------------------------------------------
# PHREEQC side
# ---------------------------------------------------------------------------

# Curated candidate assemblage (opt-in mirrors GEMS clinker suppression: what
# is not listed cannot precipitate). Names are validated against the .dat at
# runtime. Si-hydrogarnet is excluded at 25 C by common practice (kinetics);
# quartz and zeolites likewise. Extend with --extra-phases.
CANDIDATE_PHASES = [
    "Portlandite", "Cal", "Amor-Sl", "Gbs", "FeOOHmic", "Brc",
    "Gp", "Anh", "hemihydrate", "syngenite", "K2SO4", "Na2SO4",
    "ettringite", "Fe-ettringite", "thaumasite",
    "monosulphate14", "monocarbonate", "hemicarbonate", "straetlingite",
    "C3AH6", "C3FH6", "C4AH13", "hydrotalcite",
]

# GEMS-side mirror of the CANDIDATE_PHASES exclusions above: siliceous
# hydrogarnet and zeolites are suppressed at 25 C by common practice.
# Filtered against the bundle at runtime (see run_gems_case).
GEMS_MIRROR_SUPPRESSED = ("C3(AF)S0.84H", "Chabazite", "Natrolite",
                          "ZeoliteP", "ZeoliteX", "ZeoliteY")

CSH_MODELS = {
    "cshq": ["CSHQ-TobH", "CSHQ-TobD", "CSHQ-JenH", "CSHQ-JenD",
             # alkali-uptake endmembers: present in cemdata18.dat under
             # the same names and formulas as the PC bundle CSHQ DCs
             "KSiOH", "NaSiOH"],
    "cnash": ["TobH-CNASHss", "T2C-CNASHss", "T5C-CNASHss",
              "5CA", "5CNA", "INFCA", "INFCN", "INFCNA"],
}


# majors punched for the RT-P0a speciation cross-check (phreeqc.dat names;
# the GEMS side reports its own DC names - the comparison joins on the
# dw-table aliases)
PUNCH_SPECIES = ["H+", "OH-", "Ca+2", "CaOH+", "Na+", "K+", "SO4-2",
                 "HSO4-", "CaSO4", "NaSO4-", "KSO4-", "H4SiO4", "H3SiO4-",
                 "Al(OH)4-", "CO3-2", "HCO3-"]


def build_phreeqc_input(case: Dict, elements: Dict[str, float],
                        candidate_phases: List[str], csh_model: str) -> str:
    reactants = decompose_to_reactants(elements)
    water_kg = reactants["H2O"] * H2O_G_MOL / 1000.0
    if water_kg <= 0.0:
        raise ValueError("case has no water")
    # REACTION coefficients are RELATIVE; the trailing "<total> moles" line
    # sets the absolute amount, so listing absolute mols and summing them
    # yields exactly the ledger amounts.
    reaction_lines = "\n".join(
        f"    {formula} {mol:.15g}"
        for formula, mol in reactants.items()
        if formula != "H2O" and mol > 0.0)
    eq_lines = "\n".join(f"    {p} 0 0" for p in candidate_phases)
    ss_comps = "\n".join(f"        -comp {p} 0" for p in CSH_MODELS[csh_model])
    totals = [el for el in ELEMENT_IDS if el not in ("H", "O")]
    punch_phases = " ".join(candidate_phases)
    punch_ss = " ".join(CSH_MODELS[csh_model])
    return f"""
KNOBS
    -iterations 400
    -convergence_tolerance 1e-10
SOLUTION 1  pure water, ledger free water
    temp {TEMPERATURE_K - 273.15:.2f}
    pH 7.0
    pe 8.0
    water {water_kg:.15g}
REACTION 1  ledger element inventory as neutral reactants (mol, absolute)
{reaction_lines}
    1.0 moles in 1 steps
EQUILIBRIUM_PHASES 1
{eq_lines}
SOLID_SOLUTIONS 1
    CSH  # ideal multicomponent — {csh_model} endmembers from the .dat
{ss_comps}
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
    -pH true
    -mu true
    -water true
    -totals {' '.join(totals)}
    -molalities {' '.join(PUNCH_SPECIES)}
    -equilibrium_phases {punch_phases}
    -solid_solutions {punch_ss}
END
"""


def run_phreeqc_case(name: str, case: Dict, dat_path: Path,
                     csh_model: str, extra_phases: List[str]) -> Dict:
    from phreeqpython import PhreeqPython
    elements = case_elements(case)
    formulas = parse_dat_phase_formulas(dat_path)
    available = set(formulas)
    candidates = [p for p in CANDIDATE_PHASES + extra_phases if p in available]
    missing = [p for p in CANDIDATE_PHASES + extra_phases if p not in available]
    for p in CSH_MODELS[csh_model]:
        if p not in available:
            raise RuntimeError(f"{csh_model} endmember {p!r} not in {dat_path.name}")
    pp = PhreeqPython(database=dat_path.name, database_directory=dat_path.parent)
    ip = pp.ip
    inp = build_phreeqc_input(case, elements, candidates, csh_model)
    ip.run_string(inp)
    rows = ip.get_selected_output_array()
    header, last = rows[0], rows[-1]
    col = {str(h).strip(): i for i, h in enumerate(header)}

    def _get(key: str) -> float:
        return float(last[col[key]])

    water_kg = _get("mass_H2O")
    aqueous = {el: _get(f"{el}(mol/kgw)") * water_kg
               for el in ELEMENT_IDS
               if el not in ("H", "O") and f"{el}(mol/kgw)" in col}
    phase_mol = {p: _get(p) for p in candidates if p in col}
    # solid-solution component columns are punched with an "s_" prefix
    ss_mol = {p: _get(f"s_{p}") for p in CSH_MODELS[csh_model]
              if f"s_{p}" in col}
    # element closure: solids (from .dat formulas) + aqueous vs input
    solid_elements = {el: 0.0 for el in ELEMENT_IDS}
    for p, mol in list(phase_mol.items()) + list(ss_mol.items()):
        if mol <= 0.0:
            continue
        for el, n in formulas[p].items():
            if el in solid_elements:
                solid_elements[el] += n * mol
    closure = {}
    scale = max(abs(v) for v in elements.values()) or 1.0
    for el in ELEMENT_IDS:
        if el in ("H", "O"):
            continue  # solvent partition handled by PHREEQC mole balance
        total = solid_elements[el] + aqueous.get(el, 0.0)
        closure[el] = (total - elements[el]) / max(abs(elements[el]), 1e-12 * scale)
    csh = {el: 0.0 for el in ("Ca", "Si", "Al", "Na")}
    csh_total = 0.0
    for p, mol in ss_mol.items():
        if mol <= 0.0:
            continue
        csh_total += mol
        for el in csh:
            csh[el] += formulas[p].get(el, 0.0) * mol
    return {
        "engine": "phreeqc-cemdata18",
        "case": name,
        "csh_model": csh_model,
        "candidate_phases_missing_from_dat": missing,
        "input_element_mol": elements,
        "ph": _get("pH"),
        "ionic_strength_mol_kgw": _get("mu"),
        "free_water_kg": water_kg,
        "aqueous_element_mol": aqueous,
        "aqueous_species_mol": {sp: _get(f"m_{sp}(mol/kgw)") * water_kg
                                for sp in PUNCH_SPECIES
                                if f"m_{sp}(mol/kgw)" in col},
        "solid_phase_mol": {p: m for p, m in phase_mol.items() if m > 1e-12},
        "csh_endmember_mol": {p: m for p, m in ss_mol.items() if m > 1e-12},
        "csh_ca_si": (csh["Ca"] / csh["Si"]) if csh["Si"] > 0 else None,
        "csh_al_si": (csh["Al"] / csh["Si"]) if csh["Si"] > 0 else None,
        "solid_bound_element_mol": solid_elements,
        "element_closure_rel": closure,
        "element_closure_max_rel": max(abs(v) for v in closure.values()),
    }


# ---------------------------------------------------------------------------
# GEMS side (user machine: needs the xgems interpreter)
# ---------------------------------------------------------------------------

def run_gems_case(name: str, case: Dict, bundle_lst: str,
                  gems_python: Optional[str],
                  gems_unsuppress: frozenset = frozenset()) -> Dict:
    from tinn.gems import GemsWorker, SUPPRESSED_CLINKER_PHASES
    elements = case_elements(case)
    # The production backend seeds a trace of O2 so the aqueous redox state is
    # well posed (gems.py O2_SEED_MOL_O); the cold-start 0D API does not, and
    # without it LPP AIA returns "Failure (no result)" on every case here.
    # Seed relatively so the spike stays scale-free (measured: identical pH
    # and ionic strength at 1x and 1/100 of the 100 g-binder basis).
    o2_seed = O2_SEED_REL * sum(elements.values())
    elements = dict(elements)
    elements["O"] = elements.get("O", 0.0) + o2_seed
    worker = GemsWorker(bundle_lst, python_executable=gems_python)
    try:
        # Mirror the PHREEQC candidate list on the GEMS side: only phases the
        # bundle actually declares may be named (the worker hard-errors on
        # unknown names, and a hydrates-only bundle has no clinker phases -
        # the engine path filters the same way, gems.py _suppressed).
        present = set(worker.info()["phase_names"])
        suppress = tuple(p for p in
                         tuple(SUPPRESSED_CLINKER_PHASES) + GEMS_MIRROR_SUPPRESSED
                         if p in present and p not in gems_unsuppress)
        res = worker.equilibrate_elements(elements, temperature_k=TEMPERATURE_K,
                                          suppressed_phases=suppress)
    finally:
        worker.close()
    return {
        "engine": f"gems3k:{bundle_lst}",
        "case": name,
        "input_element_mol": elements,
        "o2_seed_mol": o2_seed,
        "suppressed_phases": list(suppress),
        "status": res.status,
        "ph": res.ph,
        "ionic_strength": res.ionic_strength,
        "aqueous_h2o_mol": res.aqueous_h2o_mol,
        "phase_amounts_mol": res.phase_amounts_mol,
        "phase_elements_mol": res.phase_elements_mol,
        "phase_species_mol": res.phase_species_mol,
        "aqueous_species_mol": res.aqueous_species_mol,
        "element_closure_max_rel": res.element_closure_max_rel(),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _gems_aqueous_elements(gres: Dict) -> Dict[str, float]:
    for phase in ("aq_gen", "aq", "Aqua", "aqueous"):
        if phase in gres.get("phase_elements_mol", {}):
            return gres["phase_elements_mol"][phase]
    # fall back: the phase whose elements include most of H
    best, best_h = {}, -1.0
    for phase, els in gres.get("phase_elements_mol", {}).items():
        if els.get("H", 0.0) > best_h:
            best, best_h = els, els.get("H", 0.0)
    return best


def _speciation_rows(pres: Dict, gres: Dict) -> List[Dict]:
    """RT-P0a: per-species comparison joined on the dw-table aliases (GEMS DC
    name -> phreeqc name; identical names join directly)."""
    paq = pres.get("aqueous_species_mol") or {}
    gaq = gres.get("aqueous_species_mol") or {}
    if not paq or not gaq:
        return []
    aliases = {}
    dw_json = REPO / "gems_bundles" / "species_dw" / "species_dw.json"
    if dw_json.is_file():
        aliases = json.loads(dw_json.read_text(encoding="utf-8")).get(
            "aliases", {})
    rows = []
    for gdc, gmol in sorted(gaq.items(), key=lambda kv: -abs(kv[1])):
        if gdc == "H2O@":
            continue
        pname = gdc if gdc in paq else aliases.get(gdc)
        if pname is None or pname not in paq:
            continue
        pmol = paq[pname]
        denom = max(abs(pmol), abs(gmol), 1e-30)
        rows.append({"gems_dc": gdc, "phreeqc": pname,
                     "phreeqc_mol": pmol, "gems_mol": gmol,
                     "rel_diff": (pmol - gmol) / denom})
    return rows[:12]


def compare_case(pres: Dict, gres: Dict) -> Dict:
    gaq = _gems_aqueous_elements(gres)
    rows = []
    for el in ELEMENT_IDS:
        if el in ("H", "O"):
            continue
        p = pres["aqueous_element_mol"].get(el)
        g = gaq.get(el)
        if p is None or g is None:
            continue
        denom = max(abs(p), abs(g), 1e-30)
        rows.append({"element": el, "phreeqc_aq_mol": p, "gems_aq_mol": g,
                     "rel_diff": (p - g) / denom})
    ph_p, ph_g = pres.get("ph"), gres.get("ph")
    return {
        "case": pres["case"],
        "ph": {"phreeqc": ph_p, "gems": ph_g,
               "delta": None if (ph_p is None or ph_g is None) else ph_p - ph_g},
        "ionic_strength": {"phreeqc": pres.get("ionic_strength_mol_kgw"),
                           "gems": gres.get("ionic_strength")},
        "aqueous_elements": rows,
        "phreeqc_solids": pres.get("solid_phase_mol"),
        "phreeqc_csh": {"endmembers": pres.get("csh_endmember_mol"),
                        "ca_si": pres.get("csh_ca_si"),
                        "al_si": pres.get("csh_al_si")},
        "gems_solids": {p: m for p, m in gres.get("phase_amounts_mol", {}).items()
                        if m > 1e-12},
        "gems_csh_endmembers": gres.get("phase_species_mol"),
        "closure_max_rel": {"phreeqc": pres.get("element_closure_max_rel"),
                            "gems": gres.get("element_closure_max_rel")},
        "speciation": _speciation_rows(pres, gres),
    }


def render_markdown(comparisons: List[Dict]) -> str:
    out = ["# 0D cross-check: PHREEQC-Cemdata18 vs xGEMS bundle", ""]
    for c in comparisons:
        out += [f"## {c['case']}", ""]
        ph = c["ph"]
        out += [f"pH: PHREEQC {ph['phreeqc']:.3f} vs GEMS "
                + (f"{ph['gems']:.3f} (d={ph['delta']:+.3f})" if ph["gems"] is not None
                   and not (isinstance(ph["gems"], float) and math.isnan(ph["gems"]))
                   else "n/a"), ""]
        mu = c["ionic_strength"]
        out += [f"Ionic strength: PHREEQC {mu['phreeqc']:.4f} vs GEMS "
                + (f"{mu['gems']:.4f}" if mu["gems"] is not None else "n/a"), ""]
        out += ["| element | PHREEQC aq (mol) | GEMS aq (mol) | rel diff |",
                "|---|---|---|---|"]
        for r in c["aqueous_elements"]:
            out.append(f"| {r['element']} | {r['phreeqc_aq_mol']:.4e} "
                       f"| {r['gems_aq_mol']:.4e} | {r['rel_diff']:+.2%} |")
        out += ["", f"PHREEQC C-S-H Ca/Si: {c['phreeqc_csh']['ca_si']}", ""]
        out += ["PHREEQC solids (mol): "
                + json.dumps(c["phreeqc_solids"], sort_keys=True), ""]
        out += ["GEMS solids (mol): "
                + json.dumps(c["gems_solids"], sort_keys=True), ""]
        if c.get("speciation"):
            out += ["Aqueous speciation (top species, dw-alias join):", "",
                    "| GEMS DC | phreeqc | PHREEQC (mol) | GEMS (mol) "
                    "| rel diff |", "|---|---|---|---|---|"]
            for r in c["speciation"]:
                out.append(
                    f"| {r['gems_dc']} | {r['phreeqc']} "
                    f"| {r['phreeqc_mol']:.4e} | {r['gems_mol']:.4e} "
                    f"| {r['rel_diff']:+.2%} |")
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", default="phreeqc",
                    help="comma list: phreeqc,gems")
    ap.add_argument("--out", default="runs/spike_crosscheck")
    ap.add_argument("--dat", default=str(DEFAULT_DAT))
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--gems-python", default=None)
    ap.add_argument("--csh-model", choices=sorted(CSH_MODELS), default="cshq",
                    help="cshq matches the PC bundle; cnash the CNASH bundle")
    ap.add_argument("--extra-phases", default="",
                    help="comma list of additional candidate phases")
    ap.add_argument("--gems-unsuppress", default="",
                    help="comma list removed from the GEMS mirror-suppression "
                         "set (e.g. the CNASH Test bundle's ONLY Fe hydrate "
                         "is C3(AF)S0.84H - suppressing it forces all Fe "
                         "aqueous, measured 2026-09-02)")
    ap.add_argument("--cases", default="",
                    help="comma list to restrict cases (default: all)")
    ap.add_argument("--compare", metavar="OUT_DIR", default=None,
                    help="merge result files in OUT_DIR and write comparison")
    args = ap.parse_args()

    out = Path(args.compare or args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.compare:
        pj = out / "results_phreeqc.json"
        gj = out / "results_gems.json"
        if not (pj.exists() and gj.exists()):
            print(f"need both {pj.name} and {gj.name} in {out}", file=sys.stderr)
            return 2
        pres = json.loads(pj.read_text(encoding="utf-8"))
        gres = json.loads(gj.read_text(encoding="utf-8"))
        comparisons = [compare_case(pres[k], gres[k])
                       for k in pres if k in gres and not k.startswith("__")]
        (out / "comparison.json").write_text(
            json.dumps(comparisons, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(
            render_markdown(comparisons), encoding="utf-8")
        print(f"wrote {out / 'comparison.md'}")
        return 0

    cases = _build_cases()
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = keep - set(cases)
        if unknown:
            print(f"unknown cases: {sorted(unknown)}; have {sorted(cases)}",
                  file=sys.stderr)
            return 2
        cases = {k: v for k, v in cases.items() if k in keep}
    engines = {e.strip() for e in args.engines.split(",") if e.strip()}
    extra = [p.strip() for p in args.extra_phases.split(",") if p.strip()]

    if "phreeqc" in engines:
        dat = Path(args.dat).resolve()
        results = {}
        for name, case in cases.items():
            print(f"[phreeqc] {name} ...", flush=True)
            results[name] = run_phreeqc_case(name, case, dat,
                                             args.csh_model, extra)
        payload = {"__provenance__": {
            "dat": str(dat), "dat_sha256": _sha256(dat),
            "csh_model": args.csh_model, "temperature_k": TEMPERATURE_K}}
        payload.update(results)
        (out / "results_phreeqc.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out / 'results_phreeqc.json'}")

    if "gems" in engines:
        results = {}
        for name, case in cases.items():
            print(f"[gems] {name} ...", flush=True)
            results[name] = run_gems_case(
                name, case, args.bundle, args.gems_python,
                frozenset(x.strip() for x in args.gems_unsuppress.split(",")
                          if x.strip()))
        payload = {"__provenance__": {
            "bundle": args.bundle, "temperature_k": TEMPERATURE_K}}
        payload.update(results)
        (out / "results_gems.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out / 'results_gems.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
