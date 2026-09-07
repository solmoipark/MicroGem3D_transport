# -*- coding: utf-8 -*-
"""Chloride ingress qualification (RVE scale, RT-Cl-2): per-checkpoint
observables of a `chloride_ingress_opc32` run - the boundary chloride
uptake, the surface-sorbed chloride (Surf_c pool), the free chloride,
and per-layer volumes of the chloride AFm phases (Friedel's / Kuzel's
salt) along the exposed axis (layer 0 = the bath face).

    py -3 scripts/analyze_chloride_ingress.py [runs/chloride_ingress_opc32]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.registry import ELEMENT_IDS, registry_for      # noqa: E402
from tinn.storage import load_checkpoint                  # noqa: E402
from tinn.config import TinnConfig                        # noqa: E402

TRACK = ("Portlandite", "CSHQ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("cfg=")]
    run_dir = Path(args[0] if args else REPO / "runs" / "chloride_ingress_opc32")
    cfg_path = next((a[4:] for a in sys.argv[1:] if a.startswith("cfg=")),
                    str(REPO / "examples" / "qualification"
                        / "chloride_ingress_opc32.json"))
    ckpts = sorted(run_dir.glob("ckpt_*"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    cfg = TinnConfig.model_validate(json.loads(
        Path(cfg_path).read_text(encoding="utf-8")))
    reg = registry_for(cfg)
    axis = {"z": 0, "y": 1, "x": 2}[cfg.transport.boundary.axis]
    other = tuple(a for a in range(3) if a != axis)
    cl_idx = ELEMENT_IDS.index("Cl")
    # cement mass in the RVE from the initial anhydrous clinker mol (the
    # four P&K phases carry 0.913 of the recipe mass; the salts ride the
    # fifth channel and are folded in by that ratio)
    CLINKER_G_MOL = (228.32, 172.24, 270.19, 485.96)      # C3S C2S C3A C4AF
    mf = cfg.binder.mass_fractions
    clinker_frac = sum(mf.get(k, 0.0) for k in ("C3S", "C2S", "C3A", "C4AF"))
    vox_m3 = (cfg.rve.voxel_size_um * 1e-6) ** 3
    rows = []
    for ck in ckpts:
        st = load_checkpoint(str(ck), reg)
        cement_g = float(np.dot(st.initial_phase_mol[:4], CLINKER_G_MOL)) / clinker_frac
        hyd = list(st.hydrate_ids)
        cl_phases = tuple(h for h in hyd
                          if "Friedel" in h or "Kuzel" in h or "Cl" in h)
        prof = {}
        for name in cl_phases + TRACK:
            if name in hyd:
                prof[name] = st.hydrate_fraction[hyd.index(name)].sum(
                    axis=other).tolist()
        sorbed = (float(st.domain_sorbed_mol[:, cl_idx].sum())
                  if st.domain_sorbed_mol.size else 0.0)
        cl_in = float(st.boundary_exchanged_elements[cl_idx])
        cl_aq = float(st.cluster_inventory[:, cl_idx].sum())
        liq_m3 = float(st.capillary_liquid.sum()) * vox_m3
        bound = cl_in - cl_aq                      # solids + sorbed, by balance
        # RT-06 (review): S partition, CH volume, and an approximate pH so the
        # S-R dt-convergence ladder can be judged on the same state.
        s_idx = ELEMENT_IDS.index("S")
        s_aq = float(st.cluster_inventory[:, s_idx].sum())
        s_sorbed = (float(st.domain_sorbed_mol[:, s_idx].sum())
                    if st.domain_sorbed_mol.size else 0.0)
        ch_vox = (float(st.hydrate_fraction[hyd.index("Portlandite")].sum())
                  if "Portlandite" in hyd else 0.0)
        # charge-balance pH (approximation, no speciation): the strong-ion
        # difference [Na]+[K]+2[Ca]-[Cl]-2[S(VI)] in the pore solution is
        # balanced by OH- (alkaline cement pore solution); activity, ion
        # pairing and CaOH+ are ignored, so this is indicative, not a GEMS 0D
        # pH. Reported alongside so the dt ladder tracks the same quantity.
        ph = None
        if liq_m3 > 0.0:
            def _molL(el):
                return float(st.cluster_inventory[:, ELEMENT_IDS.index(el)
                                                   ].sum()) / liq_m3 / 1e3
            sid = (_molL("Na") + _molL("K") + 2.0 * _molL("Ca")
                   - _molL("Cl") - 2.0 * _molL("S"))
            if sid > 0.0:
                ph = 14.0 + math.log10(sid)        # [OH-] = sid, pOH=-log10
            elif sid < 0.0:
                ph = -math.log10(-sid)             # [H+] = -sid
        rows.append({
            "ckpt": ck.name,
            "time_h": float(st.time_h),
            "Cl_in_from_bath_mol": cl_in,
            "Cl_sorbed_mol": sorbed,
            "Cl_aqueous_mol": cl_aq,
            "cement_g": cement_g,
            "free_Cl_mol_L": cl_aq / liq_m3 / 1e3 if liq_m3 > 0 else None,
            "bound_Cl_mg_per_g_cement": bound * 35.453 * 1e3 / cement_g,
            "sorbed_Cl_mg_per_g_cement": sorbed * 35.453 * 1e3 / cement_g,
            "S_aqueous_mol": s_aq,
            "S_sorbed_mol": s_sorbed,
            "free_S_mol_L": s_aq / liq_m3 / 1e3 if liq_m3 > 0 else None,
            "CH_volume_vox": ch_vox,
            "pH_charge_balance": ph,
            "cl_phases": list(cl_phases),
            "layer_profiles_vox": prof,
            "layer_liquid_vox": st.capillary_liquid.sum(axis=other).tolist(),
        })
    out = run_dir / "chloride_ingress_analysis.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    start = cfg.transport.boundary.start_h or 0.0
    print(f"{'t_h':>8} {'t_exp_h':>8} {'Cl_in':>10} {'Cl_sorbed':>10} "
          f"{'Cl_aq':>10}  Cl-AFm(layers 0-3 | 12-15 | 28-31)")
    for r in rows:
        afm = None
        for name in r["cl_phases"]:
            if name in r["layer_profiles_vox"]:
                v = np.asarray(r["layer_profiles_vox"][name])
                afm = v if afm is None else afm + v
        seg = ("n/a" if afm is None else
               f"{afm[0:4].sum():7.3f} | {afm[12:16].sum():7.3f} | "
               f"{afm[28:32].sum():7.3f}")
        print(f"{r['time_h']:8.2f} {r['time_h']-start:8.3f} "
              f"{r['Cl_in_from_bath_mol']:10.3e} {r['Cl_sorbed_mol']:10.3e} "
              f"{r['Cl_aqueous_mol']:10.3e}  {seg}")
    last = rows[-1]
    fc = last["free_Cl_mol_L"]
    print(f"final: cement {last['cement_g']:.3e} g, free Cl "
          f"{fc if fc is None else round(fc, 4)} mol/L, bound "
          f"{last['bound_Cl_mg_per_g_cement']:.2f} mg/g cement (sorbed "
          f"{last['sorbed_Cl_mg_per_g_cement']:.2f}; per g paste x1/(1+w/c) "
          f"= {last['bound_Cl_mg_per_g_cement']/(1+cfg.w_c):.2f})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
