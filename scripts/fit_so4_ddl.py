# -*- coding: utf-8 -*-
"""RT-S2a: joint fit of the intrinsic SO4 log_K under the DDL surface
model against Divet & Randriambololona 1998 (CCR 28(3), 357): the four
NaOH isotherms (Figs. 2-3). The NaCl ionic-strength series at 0.2 M
NaOH (Fig. 4) is NOT fitted - it is the out-of-sample prediction test
(the ledger carries no Cl element yet, and predicting the trend the
no_edl model cannot even sign-match is the stronger check anyway).

Fixed, literature-owned pieces:
- sites 2.79 mmol/g and area 350 m2/g (Labbez 4.8 silanol/nm2, Divet
  BET) -> specific_area_m2_per_mol_site = 350/2.79e-3 = 1.2544e5
- charging: silanol deprotonation + Ca complexation, the PHREEQC-
  validated pair of Haas & Nonat 2015 (CCR 68, 124): pK_H 7.7 with
  log K_SiOCa -10.3 (sensitivity: their MC-linearization pair 9.8/-7).
Scanned: the sorbate reaction form (A releases OH-, B does not) and
its intrinsic log_K. Winner = lowest joint SSE over the NaOH fits.
"""
import sys
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(REPO) / "src"))
from tinn.config import SorptionConfig
from tinn.backend import SorptionOperator
from tinn.registry import ELEMENT_IDS

DAT = str(Path(REPO) / "gems_bundles" / "PHREEQC-cemdata18"
          / "cemdata18.dat")
MASS_G = 3.4
VOL_L = 0.25
WATER_MOL = 250.0 / 18.015
SITES_MOL = 2.79e-3 * MASS_G
AREA_PER_MOL = 350.0 / 2.79e-3          # m2 per mol sites
T_K = 298.15
IDX = {el: i for i, el in enumerate(ELEMENT_IDS)}

FORMS = {
    "A_OH_release": {"reaction": "Surf_sOH + SO4-2 = Surf_sSO4- + OH-",
                     "sorbed_elements": {"S": 1.0, "O": 3.0, "H": -1.0}},
    "B_no_release": {"reaction": "Surf_sOH + SO4-2 = Surf_sOHSO4-2",
                     "sorbed_elements": {"S": 1.0, "O": 4.0}},
}
CHARGING = {
    "haas_meanfield": [
        {"reaction": "Surf_sOH = Surf_sO- + H+", "log_k": -7.7,
         "sorbed_elements": {"H": -1.0}},
        {"reaction": "Surf_sOH + Ca+2 = Surf_sOCa+ + H+", "log_k": -10.3,
         "sorbed_elements": {"Ca": 1.0, "H": -1.0}}],
    "labbez_mc": [
        {"reaction": "Surf_sOH = Surf_sO- + H+", "log_k": -9.8,
         "sorbed_elements": {"H": -1.0}},
        {"reaction": "Surf_sOH + Ca+2 = Surf_sOCa+ + H+", "log_k": -7.0,
         "sorbed_elements": {"Ca": 1.0, "H": -1.0}}],
}

# (NaOH mol/L, NaCl mol/L, [(C_eq mmol/L, Cb mmol/g), ...]) - digitized
# from Divet Figs. 3-4 (600 dpi rendering), 0.5 M from the Fig. 2
# regression 1/Cb = 54.42/C - 0.053 (R2 0.9868).
DATASETS_FIT = {
    "0.05M NaOH":  (0.05, 0.0, [(20, 0.25), (30, 0.47), (47, 0.62),
                                (65, 0.93), (95, 1.03), (157, 1.50)]),
    "0.1M NaOH":   (0.10, 0.0, [(7.6, 0.06), (10.4, 0.09), (15, 0.16),
                                (22, 0.22), (30, 0.41), (45, 0.62),
                                (67, 0.81), (78, 0.83), (78, 1.02),
                                (93, 1.25), (125, 1.46)]),
    "0.2M NaOH":   (0.20, 0.0, [(22, 0.41), (28, 0.52), (38, 0.60),
                                (44, 0.72), (64, 1.00), (75, 1.06),
                                (79, 1.18), (94, 1.27), (121, 1.65)]),
    "0.5M NaOH":   (0.50, 0.0, [(c, 1.0 / (54.42 / c - 0.053))
                                for c in (10.0, 20.0, 40.0, 80.0, 120.0)]),
}
DATASETS_VAL = {
    "0.2M+0.5M NaCl": (0.20, 0.5, [(44, 0.90), (62, 1.27), (78, 1.47),
                                   (91, 1.67), (120, 2.00)]),
    "0.2M+2M NaCl":   (0.20, 2.0, [(23, 0.50), (43, 1.09), (59, 1.52),
                                   (74, 1.81), (88, 2.01), (115, 2.56)]),
}


def make_op(form_key, charge_key, log_k, log_k_na=None):
    f = FORMS[form_key]
    charging = [dict(c) for c in CHARGING[charge_key]]
    elements = ["S", "Ca"]
    kw = {}
    if log_k_na is not None:
        # Na+ pairing on the deprotonated site (Labbez 2006 titration
        # table measures the Na/SiO accumulation this represents) -
        # softens the potential at high NaOH, moderating the raw
        # Gouy-Chapman screening.
        charging.append({"reaction": "Surf_sO- + Na+ = Surf_sONa",
                         "log_k": log_k_na,
                         "sorbed_elements": {"Na": 1.0, "H": -1.0}})
        elements.append("Na")
        kw["alkali_exchange"] = True
    cfg = SorptionConfig(
        operator="phreeqc_surface", phreeqc_dat=DAT,
        surface_model="ddl",
        specific_area_m2_per_mol_site=AREA_PER_MOL,
        charging_reactions=charging,
        site_density_mol_per_mol={"CSHQ-TobH": 0.4766},
        surface_species=[{"reaction": f["reaction"], "log_k": log_k,
                          "sorbed_elements": dict(f["sorbed_elements"])}],
        elements=elements, **kw)
    return SorptionOperator(cfg, T_K)


def replay(op, m_naoh, data):
    rows = []
    n_naoh = m_naoh * VOL_L
    for c_eq, cb in data:
        n_s = c_eq * 1e-3 * VOL_L + cb * 1e-3 * MASS_G
        e = np.zeros(len(ELEMENT_IDS))
        e[IDX["Na"]] = n_naoh + 2.0 * n_s
        e[IDX["S"]] = n_s
        e[IDX["O"]] = n_naoh + 4.0 * n_s
        e[IDX["H"]] = n_naoh
        r = op.sorb(e, WATER_MOL, SITES_MOL)
        s_bound = float(r.sorbed_mol[IDX["S"]])
        rows.append((c_eq, cb, s_bound / MASS_G * 1e3,
                     (n_s - s_bound) / VOL_L * 1e3))
    return rows


def replay_nacl_standalone(form_key, charge_key, log_k,
                           m_naoh, m_nacl, data, log_k_na=None):
    """Out-of-sample NaCl prediction: identical reactions/constants/
    area, plain phreeqpython (the ledger carries no Cl element yet)."""
    from phreeqpython import PhreeqPython
    dat = Path(DAT)
    pp = PhreeqPython(database=dat.name, database_directory=dat.parent)
    f = FORMS[form_key]
    sp = f["reaction"].split("=", 1)[1].split(" + ")[0].strip()
    lines = ["SURFACE_MASTER_SPECIES", "    Surf_s Surf_sOH",
             "SURFACE_SPECIES", "    Surf_sOH = Surf_sOH",
             "        log_k 0",
             f"    {f['reaction']}", f"        log_k {log_k!r}"]
    for c in CHARGING[charge_key]:
        lines += [f"    {c['reaction']}", f"        log_k {c['log_k']!r}"]
    if log_k_na is not None:
        lines += ["    Surf_sO- + Na+ = Surf_sONa",
                  f"        log_k {log_k_na!r}"]
    lines += ["SELECTED_OUTPUT 1", "    -reset false",
              "    -high_precision true", "    -water true",
              f"    -molalities {sp}", "END"]
    pp.ip.run_string("\n".join(lines))
    area_m2 = SITES_MOL * AREA_PER_MOL
    rows = []
    for c_eq, cb in data:
        n_s = c_eq * 1e-3 * VOL_L + cb * 1e-3 * MASS_G
        n_naoh = m_naoh * VOL_L
        blk = ["SOLUTION 1", "    temp 25.0",
               f"    water {WATER_MOL * 18.015 / 1000.0!r} kg",
               "REACTION 1",
               f"    Na2O {n_naoh / 2.0!r}",
               f"    H2O {n_naoh / 2.0!r}",
               f"    NaCl {m_nacl * VOL_L!r}",
               f"    SO3 {n_s!r}",
               f"    Na2O {n_s!r}",
               "    1.0 moles",
               "SURFACE 1",
               f"    Surf_sOH {SITES_MOL!r} {area_m2!r} 1.0",
               "END"]
        pp.ip.run_string("\n".join(blk))
        out = pp.ip.get_selected_output_array()
        col = {str(h).strip(): i for i, h in enumerate(out[0])}
        kgw = float(out[-1][col["mass_H2O"]])
        bound = float(out[-1][col[f"m_{sp}(mol/kgw)"]]) * kgw
        rows.append((c_eq, cb, bound / MASS_G * 1e3,
                     (n_s - bound) / VOL_L * 1e3))
    return rows


def joint_sse(form_key, charge_key, log_k, log_k_na=None):
    op = make_op(form_key, charge_key, log_k, log_k_na)
    total = 0.0
    per = {}
    for name, (m_naoh, _nacl, data) in DATASETS_FIT.items():
        rows = replay(op, m_naoh, data)
        sse = sum((mod - meas) ** 2 for _, meas, mod, _ in rows)
        per[name] = (sse / len(rows)) ** 0.5
        total += sse
    return total, per


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    chg = "labbez_mc"       # 1D winner; haas pair was strictly worse
    best = None
    for form in FORMS:
        for lk_na in (None, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0):
            for lk in np.arange(-1.0, 5.01, 0.5):
                lk = round(float(lk), 2)
                try:
                    sse, _ = joint_sse(form, chg, lk, lk_na)
                except Exception as exc:
                    print(f"{form}/Na={lk_na} log_K {lk:+.1f}: FAILED "
                          f"{exc}"[:120])
                    continue
                if best is None or sse < best[3]:
                    best = (form, lk_na, lk, sse)
                    print(f"  new best {form}/Na={lk_na} "
                          f"log_K {lk:+.1f}: SSE {sse:.3f}")
    form, lk_na, lk0, _ = best
    for lk in np.arange(lk0 - 0.4, lk0 + 0.41, 0.1):
        lk = round(float(lk), 2)
        sse, _ = joint_sse(form, chg, lk, lk_na)
        if sse < best[3]:
            best = (form, lk_na, lk, sse)
    form, lk_na, lk, sse = best
    n_pts = sum(len(d[2]) for d in DATASETS_FIT.values())
    print(f"\n== WINNER: form {form}, charging {chg}, Na-pair "
          f"log_K {lk_na}, SO4 log_K {lk:+.2f}, joint SSE {sse:.3f} "
          f"over {n_pts} points ==")
    _, per = joint_sse(form, chg, lk, lk_na)
    op = make_op(form, chg, lk, lk_na)
    for name, (m_naoh, _nacl, data) in DATASETS_FIT.items():
        print(f"\n{name} (RMS {per[name]:.3f} mmol/g) "
              f"[C_data Cb_data | Cb_model C_model]:")
        for r in replay(op, m_naoh, data):
            print(f"  {r[0]:6.1f} {r[1]:5.2f} | {r[2]:5.2f} {r[3]:6.1f}")
    print("\n-- OUT-OF-SAMPLE: NaCl series (predicted, not fitted) --")
    for name, (m_naoh, m_nacl, data) in DATASETS_VAL.items():
        rows = replay_nacl_standalone(form, chg, lk, m_naoh, m_nacl,
                                      data, lk_na)
        rms = (sum((mo - me) ** 2 for _, me, mo, _ in rows)
               / len(rows)) ** 0.5
        print(f"\n{name} (RMS {rms:.3f} mmol/g):")
        for r in rows:
            print(f"  {r[0]:6.1f} {r[1]:5.2f} | {r[2]:5.2f} {r[3]:6.1f}")


if __name__ == "__main__":
    main()
