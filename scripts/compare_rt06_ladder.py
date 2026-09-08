# -*- coding: utf-8 -*-
"""RT-06: tabulate the S-R dt-convergence ladder and judge it.

Reads the per-dt exposure runs produced by run_rt06_dt_ladder.py, extracts at
each output time the free/sorbed Cl, Friedel/Kuzel voxels, free/sorbed S, CH
volume and (charge-balance) pH, and reports for the finest comparison time
whether the change is first-order (dt/2->dt/4 smaller than dt->dt/2) and whether
the binding moves < 2 % across the ladder.

Usage:
  py scripts/compare_rt06_ladder.py out=runs/rt06_cl05 \
      cfg=chloride_binding_28d_0.5M anchor=672.4
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                  # noqa: E402
from tinn.registry import ELEMENT_IDS, registry_for  # noqa: E402
from tinn.storage import load_checkpoint            # noqa: E402

CLINKER_G_MOL = (228.32, 172.24, 270.19, 485.96)


def _extract(st, cfg, other):
    import numpy as np
    hyd = list(st.hydrate_ids)
    cl_i, s_i = ELEMENT_IDS.index("Cl"), ELEMENT_IDS.index("S")
    vox_m3 = (cfg.rve.voxel_size_um * 1e-6) ** 3
    liq_m3 = float(st.capillary_liquid.sum()) * vox_m3
    mf = cfg.binder.mass_fractions
    cfrac = sum(mf.get(k, 0.0) for k in ("C3S", "C2S", "C3A", "C4AF"))
    cement_g = float(np.dot(st.initial_phase_mol[:4], CLINKER_G_MOL)) / cfrac
    cl_aq = float(st.cluster_inventory[:, cl_i].sum())
    cl_in = float(st.boundary_exchanged_elements[cl_i])
    sorbed_cl = (float(st.domain_sorbed_mol[:, cl_i].sum())
                 if st.domain_sorbed_mol.size else 0.0)
    friedel = sum(float(st.hydrate_fraction[hyd.index(h)].sum())
                  for h in hyd if "Friedel" in h or "Kuzel" in h)
    s_aq = float(st.cluster_inventory[:, s_i].sum())
    sorbed_s = (float(st.domain_sorbed_mol[:, s_i].sum())
                if st.domain_sorbed_mol.size else 0.0)
    ch = (float(st.hydrate_fraction[hyd.index("Portlandite")].sum())
          if "Portlandite" in hyd else 0.0)
    ph = None
    if liq_m3 > 0.0:
        def ml(el):
            return float(st.cluster_inventory[:, ELEMENT_IDS.index(el)].sum()) / liq_m3 / 1e3
        sid = ml("Na") + ml("K") + 2 * ml("Ca") - ml("Cl") - 2 * ml("S")
        ph = 14.0 + math.log10(sid) if sid > 0 else (
            -math.log10(-sid) if sid < 0 else None)
    return {
        "bound_Cl_mg_g": (cl_in - cl_aq) * 35.453e3 / cement_g,
        "free_Cl_mmol_L": cl_aq / liq_m3 / 1e3 * 1e3 if liq_m3 > 0 else None,
        "sorbed_Cl_mol": sorbed_cl,
        "friedel_vox": friedel,
        "free_S_mmol_L": s_aq / liq_m3 / 1e3 * 1e3 if liq_m3 > 0 else None,
        "sorbed_S_mol": sorbed_s,
        "CH_vox": ch,
        "pH": ph,
    }


def main():
    import numpy as np
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out_root = REPO / args["out"]
    cfg_name = args.get("cfg", "chloride_binding_28d_0.5M")
    anchor = float(args.get("anchor", "672.4"))
    base = json.loads((REPO / "examples" / "qualification"
                       / f"{cfg_name}.json").read_text(encoding="utf-8"))
    cfg = TinnConfig.model_validate(base)
    reg = registry_for(cfg)
    axis = {"z": 0, "y": 1, "x": 2}[cfg.transport.boundary.axis]
    other = tuple(a for a in range(3) if a != axis)
    out_times = base["schedule"]["output_times_h"]

    dt_dirs = sorted([p for p in out_root.iterdir()
                      if p.is_dir() and p.name.startswith("dt_")],
                     key=lambda p: -float(p.name[3:]))   # coarse -> fine
    table = {}
    for d in dt_dirs:
        dt = float(d.name[3:])
        series = {}
        for k, t in enumerate(out_times):
            ck = d / f"ckpt_{k:03d}"
            if t <= cfg.transport.boundary.start_h + 1e-9 or not ck.exists():
                continue
            st = load_checkpoint(str(ck), reg)
            series[round(t - cfg.transport.boundary.start_h, 5)] = _extract(
                st, cfg, other)
        table[dt] = series

    # convergence judgment on bound Cl at the anchor time
    t_anchor = round(anchor - cfg.transport.boundary.start_h, 5)
    dts = sorted(table.keys(), reverse=True)            # dt, dt/2, dt/4
    verdict = {}
    if len(dts) >= 3 and all(t_anchor in table[x] for x in dts[:3]):
        v = [table[x][t_anchor]["bound_Cl_mg_g"] for x in dts[:3]]
        d_coarse = abs(v[1] - v[0])
        d_fine = abs(v[2] - v[1])
        rel = abs(v[2] - v[0]) / abs(v[0]) if v[0] else None
        verdict = {
            "anchor_h": anchor,
            "bound_Cl_mg_g": {str(dts[i]): v[i] for i in range(3)},
            "change_dt_to_dt2": d_coarse,
            "change_dt2_to_dt4": d_fine,
            "first_order_ok": bool(d_fine < d_coarse),
            "rel_spread_dt_to_dt4": rel,
            "under_2pct": bool(rel is not None and rel < 0.02),
        }
    result = {"cfg": cfg_name, "table": {str(k): v for k, v in table.items()},
              "verdict": verdict}
    (out_root / "rt06_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"[written] {out_root/'rt06_comparison.json'}")


if __name__ == "__main__":
    main()
