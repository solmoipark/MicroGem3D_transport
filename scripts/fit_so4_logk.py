# -*- coding: utf-8 -*-
"""Fit the RT-S1 SO4 surface log_K against Divet & Randriambololona
(CCR 28(3) 1998) by replaying their batch experiment through the
platform's own SorptionOperator (cemdata18 activities, no_edl).

Experiment: 3.4 g synthetic C-S-H (C/S 1.57, H/S 1.26, BET SSA 350
m2/g) in 250 mL NaOH + Na2SO4, 25 C. Site total FIXED from Labbez
et al. (J Phys Chem B 110, 2006): 4.8 silanol/nm2 x 350 m2/g =
2.79 mmol sites/g. Fit target: the 0.1 M NaOH isotherm (closest to
OPC pore solution), digitized from Fig. 3. Validation: the paper's
own 0.5 M Langmuir regression 1/Cb = 54.42/C - 0.053 (Fig. 2).
"""
import sys, json
import numpy as np
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(REPO) / "src"))
from tinn.config import SorptionConfig
from tinn.backend import SorptionOperator
from tinn.registry import ELEMENT_IDS

DAT = str(Path(REPO) / "gems_bundles" / "PHREEQC-cemdata18"
          / "cemdata18.dat")
SO4_REACTION = "Surf_sOH + SO4-2 = Surf_sSO4- + OH-"
SO4_ROW = {"S": 1.0, "O": 3.0, "H": -1.0}

MASS_G = 3.4
VOL_L = 0.25
WATER_MOL = 250.0 / 18.015
SITES_PER_G = 2.79e-3          # mol/g  (Labbez 4.8 /nm2 x 350 m2/g)
SITES_MOL = SITES_PER_G * MASS_G
T_K = 298.15

# Divet Fig.3, 0.1 M NaOH (open triangles), digitized (C mmol/L, Cb mmol/g)
DATA_01 = [(7.6, 0.06), (10.4, 0.09), (15.0, 0.16), (22.0, 0.22),
           (30.0, 0.41), (45.0, 0.62), (67.0, 0.81), (78.0, 0.83),
           (78.0, 1.02), (93.0, 1.25), (125.0, 1.46)]
# Divet Fig.2 regression (0.5 M NaOH): Cb = 1/(54.42/C - 0.053)
DATA_05 = [(c, 1.0 / (54.42 / c - 0.053)) for c in
           (10.0, 20.0, 40.0, 80.0, 120.0)]

IDX = {el: i for i, el in enumerate(ELEMENT_IDS)}

def make_op(log_k):
    cfg = SorptionConfig(
        operator="phreeqc_surface", phreeqc_dat=DAT,
        site_density_mol_per_mol={"CSHQ-TobH": 0.4766},
        surface_species=[{"reaction": SO4_REACTION, "log_k": log_k,
                          "sorbed_elements": dict(SO4_ROW)}],
        elements=["S"])
    return SorptionOperator(cfg, T_K)

def batch(op, n_naoh, n_s):
    e = np.zeros(len(ELEMENT_IDS))
    e[IDX["Na"]] = n_naoh + 2.0 * n_s
    e[IDX["S"]] = n_s
    e[IDX["O"]] = n_naoh + 4.0 * n_s
    e[IDX["H"]] = n_naoh
    r = op.sorb(e, WATER_MOL, SITES_MOL)
    return float(r.sorbed_mol[IDX["S"]])

def replay(op, data, n_naoh):
    rows = []
    for c_eq, cb in data:
        n_s = c_eq * 1e-3 * VOL_L + cb * 1e-3 * MASS_G   # total S mol
        sorbed = batch(op, n_naoh, n_s)
        cb_mod = sorbed / MASS_G * 1e3                    # mmol/g
        c_mod = (n_s - sorbed) / VOL_L * 1e3              # mmol/L
        rows.append((c_eq, cb, cb_mod, c_mod))
    return rows

def sse(rows):
    return sum((cb_mod - cb) ** 2 for _, cb, cb_mod, _ in rows)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    grid = [round(x, 2) for x in np.arange(-1.0, 1.51, 0.25)]
    best = None
    for lk in grid:
        s = sse(replay(make_op(lk), DATA_01, 0.1 * VOL_L))
        print(f"log_K {lk:+.2f}  SSE(0.1N) {s:.4f}")
        if best is None or s < best[1]:
            best = (lk, s)
    lo, hi = best[0] - 0.25, best[0] + 0.25
    for lk in np.arange(lo, hi + 1e-9, 0.05):
        lk = round(float(lk), 2)
        s = sse(replay(make_op(lk), DATA_01, 0.1 * VOL_L))
        print(f"  refine log_K {lk:+.2f}  SSE {s:.4f}")
        if s < best[1]:
            best = (lk, s)
    lk = best[0]
    print(f"\n== best log_K = {lk:+.2f} (SSE {best[1]:.4f}, sites fixed "
          f"{SITES_MOL*1e3:.2f} mmol = {SITES_PER_G*1e3:.2f} mmol/g) ==")
    op = make_op(lk)
    print("\n0.1 M NaOH fit (C_data, Cb_data | Cb_model, C_model):")
    for r in replay(op, DATA_01, 0.1 * VOL_L):
        print(f"  {r[0]:6.1f} {r[1]:5.2f} | {r[2]:5.2f} {r[3]:6.1f}")
    print("\n0.5 M NaOH validation vs Fig.2 regression:")
    for r in replay(op, DATA_05, 0.5 * VOL_L):
        print(f"  {r[0]:6.1f} {r[1]:5.2f} | {r[2]:5.2f} {r[3]:6.1f}")
    # distribution ratio at engine-like condition (C ~ 10 mmol/L, 0.1 M OH)
    rows = replay(op, [(10.0, 0.0)], 0.1 * VOL_L)
    rd = rows[0][2] / rows[0][3] if rows[0][3] > 0 else float("nan")
    print(f"\nRd at ~10 mmol/L, 0.1 M NaOH: {rd*1.0:.3f} L/g = {rd*1e3:.1f} L/kg")

if __name__ == "__main__":
    main()
