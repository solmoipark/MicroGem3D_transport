# -*- coding: utf-8 -*-
"""0D activity/speciation crosscheck: GEMS vs PHREEQC on IDENTICAL real
pore solutions (2026-09-02, PHREEQC-main feasibility input).

Six element vectors taken from measured checkpoints - sealed OPC at
24/168/672 h (sorption_so4_calibrated) and a leach ladder case early/
mid/late (leach_w4_d0_2.0em09) - are speciated AQUEOUS-ONLY on both
engines (every solid suppressed / no EQUILIBRIUM_PHASES), canonical
joint scaling. Compared: pH, ionic strength, free-ion fractions
(Ca+2/SumCa, SO4-2/SumS), plus PHREEQC's activity coefficients for
OH-/Ca+2/SO4-2 as reference. This measures the ACTUAL gamma/I
discrepancy in our operating window - the datum the PHREEQC-main
decision needs (both codes are cemdata18 + extended Debye-Hueckel
variants; the port was built to reproduce GEMS, so the expectation is
small differences, but expectation is not a measurement).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tinn import storage                              # noqa: E402
from tinn.registry import ELEMENT_IDS, default_registry  # noqa: E402
from tinn.backend import _decompose_to_reactants, _SORB_OXIDES  # noqa: E402

DAT = REPO / "gems_bundles" / "PHREEQC-cemdata18" / "cemdata18.dat"
BUNDLE = str(REPO / "gems_bundles" / "PC" / "PC-dat.lst")
VM_W = 1.80686e-5          # m3/mol, engine liquid->mol conversion
VOX_M3 = (1.0e-6) ** 3     # both source configs: 1.0 um voxels
T_K = 296.15               # both source configs
O2_SEED_REL = 1e-8         # spike_phreeqc_crosscheck redox seed
CANON = 1e-2

CASES = {
    "sealed_24h":  "runs/sorption_so4_calibrated/ckpt_000",
    "sealed_168h": "runs/sorption_so4_calibrated/ckpt_001",
    "sealed_672h": "runs/sorption_so4_calibrated/ckpt_002",
    # leach states from the (v5) RT-W4(2) NP chain run - the older
    # scalar-ladder checkpoints are format v4 (no migration, PRD 0.3)
    "leach_168h":   "runs/leach_w4_dt0.5s_np_dwdef_1.0em09/ckpt_001",
    "leach_+0.1h":  "runs/leach_w4_dt0.5s_np_dwdef_1.0em09/ckpt_003",
    "leach_+0.2h":  "runs/leach_w4_dt0.5s_np_dwdef_1.0em09/ckpt_005",
}


def load_solution(ckpt: str):
    st = storage.load_checkpoint(str(REPO / ckpt), default_registry())
    e = np.clip(st.cluster_inventory.sum(axis=0), 0.0, None)
    water_mol = float(np.asarray(st.capillary_liquid).sum()) * VOX_M3 / VM_W
    return e, water_mol, float(st.time_h)


def canon_scale(e, water_mol):
    s = CANON / max(float(e.max()), 1e-300)
    return e * s, water_mol * s


def pad_oxide_frame(e):
    """The summed ledger inventory can sit ~0.1% outside the exact
    oxide/H2O frame (aqueous redox/hydroxide bookkeeping residue). Pad O
    minimally so the EXACT-closure decomposition accepts it, identically
    for BOTH engines (inputs stay identical; the pad is reported)."""
    idx = {el: i for i, el in enumerate(ELEMENT_IDS)}
    o_used = 0.0
    for formula, el, n_el, n_o in _SORB_OXIDES:
        v = float(e[idx[el]])
        if v > 0.0:
            o_used += v / n_el * n_o
    h2o = float(e[idx["H"]]) / 2.0
    o_left = float(e[idx["O"]]) - o_used - h2o
    pad = max(0.0, -o_left) * (1.0 + 1e-9)
    if pad > 0.0:
        e = e.copy()
        e[idx["O"]] += pad
    return e, pad


def run_phreeqc(e, water_mol):
    from phreeqpython import PhreeqPython
    pp = PhreeqPython(database=DAT.name, database_directory=DAT.parent)
    elements = {el: float(e[i]) for i, el in enumerate(ELEMENT_IDS)
                if e[i] > 0.0}
    reactants = _decompose_to_reactants(elements)
    lines = ["SELECTED_OUTPUT 1", "    -reset false",
             "    -high_precision true", "    -water true",
             "    -pH true", "    -mu true",
             "    -activities OH- Ca+2 SO4-2 K+",
             "    -molalities Ca+2 SO4-2 OH- K+ CaSO4 KSO4- CaOH+",
             "END",
             "SOLUTION 1", "    temp %.2f" % (T_K - 273.15),
             "    water %r kg" % (water_mol * 18.015 / 1000.0),
             "REACTION 1"]
    lines += ["    %s %r" % (f, m) for f, m in sorted(reactants.items())]
    lines += ["    1.0 moles", "END"]
    pp.ip.run_string("\n".join(lines))
    rows = pp.ip.get_selected_output_array()
    col = {str(h).strip(): i for i, h in enumerate(rows[0])}
    last = rows[-1]

    def g(k):
        return float(last[col[k]])
    kgw = g("mass_H2O")
    out = {"ph": g("pH"), "I": g("mu"), "kgw": kgw}
    for sp in ("Ca+2", "SO4-2", "OH-", "K+", "CaSO4", "KSO4-", "CaOH+"):
        out["m_" + sp] = g(f"m_{sp}(mol/kgw)") * kgw
    for sp in ("OH-", "Ca+2", "SO4-2", "K+"):
        la = g(f"la_{sp}")
        m = out["m_" + sp] / kgw
        out["gamma_" + sp] = 10.0 ** la / m if m > 0 else float("nan")
    return out


def run_gems(e, water_mol):
    import os
    from tinn.gems import GemsWorker
    elements = {el: float(e[i]) for i, el in enumerate(ELEMENT_IDS)
                if e[i] > 0.0}
    # water rides the element vector for the worker (H2O elements)
    elements["H"] = elements.get("H", 0.0) + 2.0 * water_mol
    elements["O"] = elements.get("O", 0.0) + water_mol
    # redox seed AFTER the water-add: sized on the solutes alone it is
    # ~400x too small for a water-dominated system and LPP AIA returns
    # "Failure (no result)" (measured; seed-size insensitive 1e-8..1e-4)
    elements["O"] = (elements.get("O", 0.0)
                     + O2_SEED_REL * sum(elements.values()))
    worker = GemsWorker(BUNDLE,
                        python_executable=os.environ.get("TINN_GEMS_PYTHON"))
    try:
        solids = [p for p in worker.info()["phase_names"]
                  if not p.lower().startswith(("aq", "gas"))
                  # CO3_SO4_AFt ignores suppress_multiple_phases (measured
                  # leak, see the worker's 0D suppression witness) - leave
                  # it unsuppressed DELIBERATELY; the aqueous-element
                  # denominators keep the speciation comparison valid and
                  # the leak report shows what precipitated.
                  and p != "CO3_SO4_AFt"]
        res = worker.equilibrate_elements(elements, temperature_k=T_K,
                                          suppressed_phases=tuple(solids))
    finally:
        worker.close()
    spec = dict(res.aqueous_species_mol or {})
    kgw = (res.aqueous_h2o_mol or 0.0) * 18.015 / 1000.0
    # KNOWN ISSUE (measured 2026-09-02): suppress_multiple_phases leaks
    # for at least the CO3_SO4_AFt solid solution on this 0D path - it
    # precipitated 45% of S despite being in the suppression tuple. The
    # engine path is unexposed (single-DC clinker suppressions + its own
    # suppression-failed witness, gems.py ~:636), but 0D spikes must
    # self-check: report aqueous ELEMENT totals as the speciation
    # denominators and surface any solid leak explicitly.
    aq_el, leak = {}, {}
    for ph, els in (res.phase_elements_mol or {}).items():
        if ph.startswith("aq"):
            aq_el = dict(els)
        elif any(v > 0.0 for v in els.values()):
            for k, v in els.items():
                if v > 0.0:
                    leak[ph] = leak.get(ph, 0.0) + v
    return {"ph": res.ph, "I": res.ionic_strength, "kgw": kgw,
            "status": res.status, "species": spec, "aq_el": aq_el,
            "leak": leak}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    idx = {el: i for i, el in enumerate(ELEMENT_IDS)}
    print(f"{'case':<12} {'pH_G':>6} {'pH_P':>6} {'dpH':>6} "
          f"{'I_G mM':>8} {'I_P mM':>8} {'dI%':>6}  free-ion fractions")
    for name, ckpt in CASES.items():
        e_raw, w_raw, t_h = load_solution(ckpt)
        e, w = canon_scale(e_raw, w_raw)
        e, o_pad = pad_oxide_frame(e)
        p = run_phreeqc(e, w)
        g = run_gems(e, w)
        di = (100.0 * (p["I"] - g["I"]) / g["I"]
              if g["I"] else float("nan"))
        # free-ion fractions: PHREEQC direct; GEMS from species dict
        tot_ca = e[idx["Ca"]]
        tot_s = e[idx["S"]]
        f_ca_p = p["m_Ca+2"] / tot_ca if tot_ca > 0 else float("nan")
        f_s_p = p["m_SO4-2"] / tot_s if tot_s > 0 else float("nan")
        gsp = g["species"]
        aq_ca = g["aq_el"].get("Ca", 0.0)
        aq_s = g["aq_el"].get("S", 0.0)
        f_ca_g = (gsp.get("Ca+2", 0.0) / aq_ca
                  if aq_ca > 0 else float("nan"))
        f_s_g = (gsp.get("SO4-2", 0.0) / aq_s
                 if aq_s > 0 else float("nan"))
        print(f"{name:<12} {g['ph']:6.3f} {p['ph']:6.3f} "
              f"{p['ph']-g['ph']:+6.3f} {g['I']*1e3:8.2f} "
              f"{p['I']*1e3:8.2f} {di:+6.1f}  "
              f"Ca2+ G {f_ca_g:.3f}/P {f_ca_p:.3f}  "
              f"SO4 G {f_s_g:.3f}/P {f_s_p:.3f}")
        print(f"{'':>12} gamma_P: OH {p['gamma_OH-']:.3f} "
              f"Ca {p['gamma_Ca+2']:.3f} SO4 {p['gamma_SO4-2']:.3f} "
              f"K {p['gamma_K+']:.3f}   (t={t_h:g} h, "
              f"water {w_raw:.2e} mol, O-pad {o_pad:.1e})")
        if g["leak"]:
            print(f"{'':>12} !! GEMS suppression leak (mol by phase): "
                  + ", ".join(f"{k} {v:.2e}" for k, v in g["leak"].items()))


if __name__ == "__main__":
    main()
