# -*- coding: utf-8 -*-
"""RT-W4 leaching qualification (PRD 3 RT-W4, external review rec. 5):
hydrate the Deschner OPC sealed to 7 d, expose the z-low face to renewed
pure water (start_h), and fit the CH-depletion front against sqrt(t) over
the wrap-guarded window - for a D0 sensitivity ladder spanning the ionic
self-diffusivity range. Literature anchor: a ~ 100-200 um/sqrt(day) for
mature paste (immature up to ~2x faster).

Usage:  py -3 scripts/run_leach_w4.py [cfg=<name>] [dt=<seconds>] [d0 ...]
        cfg names an examples/qualification/leach_w4_<name>.json base
        (default opc32); W4.2 front runs use cfg=opc64.
"""
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn import analysis                    # noqa: E402
from tinn.config import TinnConfig           # noqa: E402
from tinn.engine import Engine               # noqa: E402
from tinn.registry import ELEMENT_IDS, registry_for  # noqa: E402
from tinn.storage import load_checkpoint     # noqa: E402

START_H = 168.0
GUARD_LO_VOX = 4                 # exposed-face halo (PRD 4.6.3)
TILE_Z = 2

def base_path(name: str) -> Path:
    return REPO / "examples" / "qualification" / f"leach_w4_{name}.json"

def run_case(d0: float, out_dir: Path, dt_s: float = None,
             base: Path = None) -> dict:
    raw = json.loads((base or base_path("opc32")).read_text(encoding="utf-8"))
    raw["transport"]["domains"]["d0_m2_s"] = d0
    if dt_s is not None:
        # W4.1 dt ladder: replace the leach window's step (seconds)
        dt_h = dt_s / 3600.0
        raw["schedule"]["dt_windows"][-1]["dt_h"] = dt_h
        raw["schedule"]["dt_min_h"] = min(raw["schedule"]["dt_min_h"],
                                          dt_h / 4.0)
    cfg = TinnConfig.model_validate(raw)
    supply = []

    frozen = {"nonconv": 0.0, "water": 0.0, "surrendered": 0.0,
              "surrendered_mol": 0.0}

    def hook(ev):
        if ev.get("event") == "step_accepted":
            m = ev.get("metrics", {})
            if "boundary_supply_ratio" in m:
                supply.append(m["boundary_supply_ratio"])
            frozen["nonconv"] += m.get("nonconv_frozen_domains", 0.0)
            frozen["water"] += m.get("water_frozen_domains", 0.0)
            frozen["surrendered"] += m.get("dryout_surrendered_domains", 0.0)
            frozen["surrendered_mol"] += m.get("dryout_surrendered_mol", 0.0)

    if out_dir.exists():
        # storage refuses checkpoint overwrites; a rerun after an aborted
        # or completed ladder must start clean (review finding)
        import shutil
        assert out_dir.name.startswith("leach_w4_")
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    state, summary = Engine(cfg).run(out_dir=str(out_dir), audit_hook=hook)
    wall = time.perf_counter() - t0

    reg = registry_for(cfg)
    n = cfg.rve.grid_size
    h_um = cfg.rve.voxel_size_um
    guard_hi = n - (GUARD_LO_VOX + TILE_Z)
    rows = []
    for k, t_out in enumerate(cfg.schedule.output_times_h):
        if t_out < START_H:
            continue
        s = load_checkpoint(str(out_dir / f"ckpt_{k:03d}"), reg)
        prof = analysis.boundary_profiles(s, 0)
        rows.append({
            "t_h": t_out,
            "t_leach_h": t_out - START_H,
            "front_vox": prof["ch_front_depth_vox"],
            "ca_out_mol": float(-s.boundary_exchanged_elements[
                ELEMENT_IDS.index("Ca")]),
        })
    # sqrt(t) fit on the guarded window (exclude the alkali-washout
    # transient at t_leach < 0.05 h and fronts outside [guard_lo, guard_hi])
    pts = [(r["t_leach_h"], r["front_vox"]) for r in rows
           if r["t_leach_h"] >= 0.05
           and GUARD_LO_VOX <= r["front_vox"] <= guard_hi]
    a_um_sqrt_day = None
    if len(pts) >= 3:
        # least squares through the origin: x_f = a sqrt(t)
        sx = sum(math.sqrt(t / 24.0) * x * h_um for t, x in pts)
        sxx = sum(t / 24.0 for t, _ in pts)
        a_um_sqrt_day = sx / sxx if sxx > 0 else None
    return {
        "d0_m2_s": d0, "wall_s": round(wall, 1),
        "supply_ratio_max": max(supply) if supply else None,
        "nonconv_frozen_total": frozen["nonconv"],
        "water_frozen_total": frozen["water"],
        "dryout_surrendered_total": frozen["surrendered"],
        "dryout_surrendered_mol": frozen["surrendered_mol"],
        "supply_ratio_mean": (sum(supply) / len(supply)) if supply else None,
        "rows": rows,
        "a_um_per_sqrt_day": a_um_sqrt_day,
        "n_fit_points": len(pts),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    dt_s = None
    cfg_name = "opc32"
    while args and "=" in args[0]:
        key, _, val = args[0].partition("=")
        if key == "dt":
            dt_s = float(val)
        elif key == "cfg":
            cfg_name = val
        else:
            raise SystemExit(f"unknown option {args[0]!r}")
        args = args[1:]
    base = base_path(cfg_name)
    ladder = [float(x) for x in args] or [0.8e-9, 2.0e-9, 5.3e-9]
    results = {}
    for d0 in ladder:
        tag = f"d0_{d0:.1e}".replace("-", "m").replace("+", "")
        if dt_s is not None:
            tag = f"dt{dt_s:g}s_{tag}"
        if cfg_name != "opc32":
            tag = f"{cfg_name}_{tag}"
        out = REPO / "runs" / f"leach_w4_{tag}"
        print(f"== {tag} ==", flush=True)
        results[tag] = run_case(d0, out, dt_s, base)
        r = results[tag]
        print(f"  wall {r['wall_s']} s, a = {r['a_um_per_sqrt_day']} "
              f"um/sqrt(day) ({r['n_fit_points']} pts), supply "
              f"max/mean = {r['supply_ratio_max']}/{r['supply_ratio_mean']}",
              flush=True)
        fronts = [(row['t_leach_h'], row['front_vox']) for row in r['rows']]
        print(f"  fronts: {fronts}", flush=True)
    suffix = f"_dt{dt_s:g}s" if dt_s is not None else ""
    if cfg_name != "opc32":
        suffix = f"_{cfg_name}{suffix}"
    out_json = REPO / "runs" / f"leach_w4_results{suffix}.json"
    out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
