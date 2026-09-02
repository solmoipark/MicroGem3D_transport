# -*- coding: utf-8 -*-
"""Fit the RT-Cl-2 chloride surface log_K against Hirao, Yamada,
Takahashi & Zibara (J. Adv. Concr. Technol. 3(1) 2005, Fig. 7) by
replaying their batch experiment through the platform's own
SorptionOperator (cemdata18 activities, no_edl, CH-buffered).

Experiment: 1 g hydrate in 10 cm3 NaCl (0-5 M), 2 d, 20 C, suction
filtered. Sorbent: C-S-H from C3S hydration (w/s 10, 56 d; Ca/Si ~1.7
incl. the CH they later corrected out). Paper's own CH-corrected
Langmuir regression (Fig. 7): q = 0.61602 x 2.64947 C / (1 + 2.64947 C),
q in mmol/g pure C-S-H, C in mol/L.

Site model (RT-Cl-2): Cl- binds the Ca-decorated silanol pool Surf_c
(Elakneswaran 2009 eq. 6 family), whose total is FIXED by the Hirao
plateau: 0.616 mmol/g. The silanol pool Surf_s (0.4766 per Si, RT-S1c)
is present but idle (no sulfate in the batch). Fit target: C <= 1 M
(pore-solution range; the dat's activity model is not built for 5 M
NaCl) - the 2-5 M points are reported as an extrapolation check.

Cross-source band: Tang & Nilsson (CCR 23 1993) OPC-paste Freundlich
log Cb = 0.3788 log c + 1.140 (Cb mg/g C-S-H gel, c mol/L; c > 0.01 M).
"""
import sys
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
CL_REACTION = "Surf_cOH + Cl- = Surf_cOHCl-"
CL_ROW = {"Cl": 1.0}

MASS_G = 1.0
VOL_L = 0.010
WATER_MOL = 10.0 / 18.015
T_K = 293.15
# C-S-H at Ca/Si 1.7, ~2.1 H2O/Si (jennite-like, 11 % RH dried):
# M ~ 1.7*56.08 + 60.08 + 2.1*18.015 = 193.3 g per mol Si
M_PER_SI = 1.7 * 56.077 + 60.084 + 2.1 * 18.015
N_SI = MASS_G / M_PER_SI
SITES_S_MOL = 0.4766 * N_SI            # RT-S1c silanol pool (idle here)
Q_MAX_MMOL_G = 0.61602                 # Hirao Fig. 7 plateau
SITES_C_MOL = Q_MAX_MMOL_G * 1e-3 * MASS_G
K_L = 2.64947                          # Hirao Fig. 7, L/mol (concentration)
BUFFER_MOL = 2.0e-3                    # CH in excess (alite-derived sample)

C_FIT = (0.10, 0.25, 0.50, 1.00)
C_EXTRA = (2.0, 3.0, 5.0)


def hirao(c):
    return Q_MAX_MMOL_G * K_L * c / (1.0 + K_L * c)


def tang_nilsson_mmol_g(c):
    return 10 ** (0.3788 * np.log10(c) + 1.140) / 35.453


IDX = {el: i for i, el in enumerate(ELEMENT_IDS)}


def make_op(log_k):
    cfg = SorptionConfig(
        operator="phreeqc_surface", phreeqc_dat=DAT,
        site_density_mol_per_mol={"CSHQ-JenH": 0.4766},
        site_density_c_mol_per_mol={"CSHQ-JenH": 0.118},
        surface_species=[{"reaction": SO4_REACTION, "log_k": 0.50,
                          "sorbed_elements": dict(SO4_ROW)},
                         {"reaction": CL_REACTION, "log_k": log_k,
                          "sorbed_elements": dict(CL_ROW)}],
        elements=["S", "Cl"], buffer_phase="Portlandite")
    return SorptionOperator(cfg, T_K)


def batch(op, n_cl):
    e = np.zeros(len(ELEMENT_IDS))
    e[IDX["Na"]] = n_cl
    e[IDX["Cl"]] = n_cl
    r = op.sorb(e, WATER_MOL, SITES_S_MOL, buffer_mol=BUFFER_MOL,
                sites_c_mol=SITES_C_MOL)
    return float(r.sorbed_mol[IDX["Cl"]])


def replay(op, cs):
    rows = []
    for c_eq in cs:
        q = hirao(c_eq)
        n_cl = c_eq * VOL_L + q * 1e-3 * MASS_G       # total Cl mol
        sorbed = batch(op, n_cl)
        q_mod = sorbed / MASS_G * 1e3                 # mmol/g
        c_mod = (n_cl - sorbed) / VOL_L               # mol/L
        rows.append((c_eq, q, q_mod, c_mod))
    return rows


def sse(rows):
    return sum((q_mod - q) ** 2 for _, q, q_mod, _ in rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"sites: Surf_s {SITES_S_MOL*1e3:.3f} mmol (idle), Surf_c "
          f"{SITES_C_MOL*1e3:.3f} mmol = {Q_MAX_MMOL_G} mmol/g "
          f"= {SITES_C_MOL/N_SI:.4f} per Si = {SITES_C_MOL/N_SI/1.7:.4f} per Ca")
    print(f"Hirao K_L {K_L} L/mol -> log10 {np.log10(K_L):+.3f} "
          f"(concentration basis)")
    best = None
    for lk in [round(x, 2) for x in np.arange(-0.5, 1.51, 0.25)]:
        s = sse(replay(make_op(lk), C_FIT))
        print(f"log_K {lk:+.2f}  SSE(<=1 M) {s:.5f}")
        if best is None or s < best[1]:
            best = (lk, s)
    for lk in np.arange(best[0] - 0.25, best[0] + 0.25 + 1e-9, 0.05):
        lk = round(float(lk), 2)
        s = sse(replay(make_op(lk), C_FIT))
        print(f"  refine log_K {lk:+.2f}  SSE {s:.5f}")
        if s < best[1]:
            best = (lk, s)
    lk = best[0]
    print(f"\n== best log_K = {lk:+.2f} (SSE {best[1]:.5f}) ==")
    op = make_op(lk)
    print("\nC_eq (M)  q_Hirao  | q_model  C_model  | Tang&Nilsson paste-gel")
    for r in replay(op, C_FIT + C_EXTRA):
        tn = tang_nilsson_mmol_g(r[0])
        tag = "" if r[0] <= 1.0 else "  (extrapolation)"
        print(f"  {r[0]:5.2f}   {r[1]:6.3f}  | {r[2]:6.3f}   {r[3]:6.3f}  "
              f"| {tn:6.3f} mmol/g{tag}")
    # pore-solution-like point: 0.5 M NaCl bath front and 20 mM
    rows = replay(op, [0.02, 0.05])
    for r in rows:
        rd = r[2] / (r[3] * 1e3) if r[3] > 0 else float("nan")
        print(f"Rd at {r[0]:.2f} M: q {r[2]:.4f} mmol/g, C {r[3]*1e3:.1f} "
              f"mmol/L -> {rd*1e3:.2f} L/kg (Hirao {r[1]:.4f} mmol/g)")


if __name__ == "__main__":
    main()
