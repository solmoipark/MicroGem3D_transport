# -*- coding: utf-8 -*-
"""RT-W4 leaching qualification (PRD 3 RT-W4, external review rec. 5):
hydrate the Deschner OPC sealed to 7 d, expose the z-low face to renewed
pure water (start_h), and fit the CH-depletion front against sqrt(t) over
the wrap-guarded window - for a D0 sensitivity ladder spanning the ionic
self-diffusivity range. Literature anchor: a ~ 100-200 um/sqrt(day) for
mature paste (immature up to ~2x faster).

Usage:  py -3 scripts/run_leach_w4.py [d0_ladder ...]
"""
import io
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
from tinn.registry import ELEMENT_IDS, default_registry  # noqa: E402
from tinn.storage import load_checkpoint     # noqa: E402

BASE = REPO / "examples" / "qualification" / "leach_w4_opc32.json"
START_H = 168.0
GUARD_LO_VOX = 4                 # exposed-face halo (PRD 4.6.3)
TILE_Z = 2

def run_case(d0: float, out_dir: Path) -> dict:
    raw = json.loads(BASE.read_text(encoding="utf-8"))
    raw["transport"]["domains"]["d0_m2_s"] = d0
    cfg = TinnConfig.model_validate(raw)
    supply = []

    def hook(ev):
        if ev.get("event") == "step_accepted":
            m = ev.get("metrics", {})
            if "boundary_supply_ratio" in m:
                supply.append(m["boundary_supply_ratio"])

    t0 = time.perf_counter()
    state, summary = Engine(cfg).run(out_dir=str(out_dir), audit_hook=hook)
    wall = time.perf_counter() - t0

    reg = default_registry()
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
            "layer_CH": prof["layer_CH_vol"],
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
        "supply_ratio_mean": (sum(supply) / len(supply)) if supply else None,
        "rows": [{k: v for k, v in r.items() if k != "layer_CH"}
                 for r in rows],
        "a_um_per_sqrt_day": a_um_sqrt_day,
        "n_fit_points": len(pts),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ladder = [float(x) for x in sys.argv[1:]] or [0.8e-9, 2.0e-9, 5.3e-9]
    results = {}
    for d0 in ladder:
        tag = f"d0_{d0:.1e}".replace("-", "m").replace("+", "")
        out = REPO / "runs" / f"leach_w4_{tag}"
        print(f"== {tag} ==", flush=True)
        results[tag] = run_case(d0, out)
        r = results[tag]
        print(f"  wall {r['wall_s']} s, a = {r['a_um_per_sqrt_day']} "
              f"um/sqrt(day) ({r['n_fit_points']} pts), supply "
              f"max/mean = {r['supply_ratio_max']}/{r['supply_ratio_mean']}",
              flush=True)
        fronts = [(row['t_leach_h'], row['front_vox']) for row in r['rows']]
        print(f"  fronts: {fronts}", flush=True)
    out_json = REPO / "runs" / "leach_w4_results.json"
    io.open(out_json, "w", encoding="utf-8").write(
        json.dumps(results, indent=1))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
