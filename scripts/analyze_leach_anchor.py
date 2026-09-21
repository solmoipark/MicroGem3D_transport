# -*- coding: utf-8 -*-
"""RT-W5: analyse one leaching-anchor run against a literature a (um/sqrt(day)).

Reads the run's checkpoints, and for every output AT/AFTER the exposure onset
(boundary.start_h from the config, NOT hardcoded) computes the CH depletion
front depth (analysis.boundary_profiles), Ca release, total CH volume, connected
porosity, and the per-layer phase profile at the front (CH -> AFm/AFt -> C-S-H
dissolution order). Fits the leaching coefficient a from x_front = a*sqrt(t) on
the wrap-guarded window (4 <= front <= n-(4+tile_z), t_leach >= 0.05 h), the same
rule as run_leach_w4. Also reports the CH front half-domain time (front = n/2).

Usage:
  py scripts/analyze_leach_anchor.py runs/<dir> examples/qualification/<cfg>.json \
     [target_a=169] [out=results_back/rt_w5/<name>.json]
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                   # noqa: E402
from tinn import analysis                            # noqa: E402
from tinn.config import TinnConfig                   # noqa: E402
from tinn.registry import ELEMENT_IDS, registry_for  # noqa: E402
from tinn.storage import load_checkpoint             # noqa: E402

GUARD_LO = 4
# phases whose front order we track (present-name tolerant)
TRACK = ["Portlandite", "ettringite", "Ettringite", "monosulfate", "Monosulfate",
         "Friedel", "Kuzel", "CSHQ", "C3AH6", "hemicarbonate", "monocarbonate"]


def main():
    args = [a for a in sys.argv[1:] if "=" not in a]
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    run_dir = REPO / args[0]
    cfg = TinnConfig.from_json_file(str(REPO / args[1]))
    target_a = float(kw["target_a"]) if "target_a" in kw else None
    reg = registry_for(cfg)
    axis = {"z": 0, "y": 1, "x": 2}[cfg.transport.boundary.axis]
    start_h = cfg.transport.boundary.start_h
    n = cfg.rve.grid_size
    h_um = cfg.rve.voxel_size_um
    tile_z = (cfg.transport.domains.tile_zyx[0]
              if cfg.transport.domains.tile_zyx else cfg.transport.domains.tile_vox)
    guard_hi = n - (GUARD_LO + tile_z)
    ch_i = None
    rows = []
    for k, t_out in enumerate(cfg.schedule.output_times_h):
        ck = run_dir / f"ckpt_{k:03d}"
        if not ck.exists():
            continue
        st = load_checkpoint(str(ck), reg)
        hyd = list(st.hydrate_ids)
        if ch_i is None and "Portlandite" in hyd:
            ch_i = hyd.index("Portlandite")
        prof = analysis.boundary_profiles(st, axis)
        split = analysis.porosity_split(st)
        # per-layer phase profiles at the front (dissolution order)
        other = tuple(a for a in range(3) if a != axis)
        phase_layer = {}
        for name in TRACK:
            if name in hyd:
                phase_layer[name] = st.hydrate_fraction[hyd.index(name)].sum(
                    axis=other).tolist()
        row = {
            "t_h": t_out, "t_leach_h": round(t_out - start_h, 6),
            "front_vox": prof["ch_front_depth_vox"],
            "ca_out_mol": float(-st.boundary_exchanged_elements[
                ELEMENT_IDS.index("Ca")]),
            "CH_total_vox": (float(st.hydrate_fraction[ch_i].sum())
                             if ch_i is not None else None),
            "porosity_connected": split["connected"],
            "liquid_vol_vox": float(st.capillary_liquid.sum()),
        }
        if t_out >= start_h - 1e-9:
            row["phase_layer_front8"] = {
                nm: [round(v, 3) for v in pl[:8]] for nm, pl in phase_layer.items()}
        rows.append(row)

    leach = [r for r in rows if r["t_leach_h"] >= -1e-9]
    # a fit: x_f = a sqrt(t), guarded window
    pts = [(r["t_leach_h"], r["front_vox"]) for r in leach
           if r["t_leach_h"] >= 0.05 and GUARD_LO <= r["front_vox"] <= guard_hi]
    a_um_sqrt_day = None
    if len(pts) >= 2:
        sx = sum(math.sqrt(t / 24.0) * x * h_um for t, x in pts)
        sxx = sum(t / 24.0 for t, _ in pts)
        a_um_sqrt_day = sx / sxx if sxx > 0 else None
    # front half-domain time: interpolate t where front reaches n/2
    t_half_domain = None
    prev = None
    for r in leach:
        if r["front_vox"] >= n / 2 and prev is not None:
            f0, t0 = prev["front_vox"], prev["t_leach_h"]
            f1, t1 = r["front_vox"], r["t_leach_h"]
            if f1 > f0:
                t_half_domain = t0 + (n / 2 - f0) * (t1 - t0) / (f1 - f0)
            break
        prev = r

    result = {
        "config": args[1], "grid": n, "w_c": cfg.w_c, "T_K": cfg.temperature_K,
        "start_h": start_h, "n_leach_pts": len(pts),
        "a_um_per_sqrt_day": a_um_sqrt_day, "target_a": target_a,
        "a_ratio": (a_um_sqrt_day / target_a if a_um_sqrt_day and target_a else None),
        "front_half_domain_h": t_half_domain,
        "front_trajectory": [(r["t_leach_h"], r["front_vox"]) for r in leach],
        "rows": rows,
    }
    out = kw.get("out")
    if out:
        p = REPO / out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[written] {p}")
    print(json.dumps({k: result[k] for k in (
        "config", "w_c", "a_um_per_sqrt_day", "target_a", "a_ratio",
        "front_half_domain_h", "n_leach_pts", "front_trajectory")}, indent=2))


if __name__ == "__main__":
    main()
