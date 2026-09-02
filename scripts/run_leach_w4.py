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
             base: Path = None, until_h: float = None,
             np_mode: bool = False, resume: bool = False) -> dict:
    raw = json.loads((base or base_path("opc32")).read_text(encoding="utf-8"))
    if np_mode:
        # RT-W4(2) remeasure (PRD 4.6.4): species NP conductances replace the
        # scalar; the ladder argument becomes the EXPLICIT default dw for
        # unmapped species (recorded per species in the run summary)
        raw["transport"]["domains"].pop("d0_m2_s", None)
        raw["transport"]["domains"]["species"] = {
            "dw_table": "gems_bundles/species_dw/species_dw.json",
            "default_dw_m2_s": d0,
            "geometry_factor": 1.0,
            "phi_clamp_report": True,
        }
    else:
        raw["transport"]["domains"]["d0_m2_s"] = d0
    if until_h is not None:
        # W4.2 supply-clean confirmation: truncate the leach window so a
        # dt ~ 1/D0 run stays affordable (the front points that matter sit
        # in the first third of the window anyway)
        end = START_H + until_h
        raw["schedule"]["output_times_h"] = [
            t for t in raw["schedule"]["output_times_h"] if t <= end + 1e-9]
        raw["schedule"]["dt_windows"] = [
            w for w in raw["schedule"]["dt_windows"] if w["until_h"] < end]
        raw["schedule"]["dt_windows"].append(
            {"until_h": end, "dt_h": raw["schedule"]["dt_windows"][-1]["dt_h"]
             if raw["schedule"]["dt_windows"] else 0.0014})
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
    np_counts = {"np_phi_clamped": 0.0, "np_fick_clamped": 0.0,
                 "np_cap_clamped": 0.0, "np_smalldc": 0.0, "np_empty_el": 0.0,
                 "np_charge_flux_rel_max": 0.0}

    def hook(ev):
        if ev.get("event") == "step_accepted":
            m = ev.get("metrics", {})
            if "boundary_supply_ratio" in m:
                supply.append(m["boundary_supply_ratio"])
            frozen["nonconv"] += m.get("nonconv_frozen_domains", 0.0)
            frozen["water"] += m.get("water_frozen_domains", 0.0)
            frozen["surrendered"] += m.get("dryout_surrendered_domains", 0.0)
            frozen["surrendered_mol"] += m.get("dryout_surrendered_mol", 0.0)
            for k in np_counts:
                if k == "np_charge_flux_rel_max":
                    np_counts[k] = max(np_counts[k], m.get(k, 0.0))
                else:
                    np_counts[k] += m.get(k, 0.0)

    start_state = None
    if resume and out_dir.exists():
        # continue an interrupted run from its last checkpoint (the engine
        # is resume-complete: outputs filter on state.time_h and ckpt
        # indices are schedule-global). config_hash must match - a resume
        # across code/config drift is refused, never absorbed. NOTE: the
        # summary-derived counters (np clamps, frozen) then cover only the
        # resumed portion; checkpoint-based rows are unaffected.
        done = sorted(out_dir.glob("ckpt_*"))
        if done:
            from tinn.storage import load_checkpoint
            start_state = load_checkpoint(str(done[-1]), registry_for(cfg))
            if start_state.config_hash != cfg.config_hash():
                raise SystemExit(
                    f"resume refused: checkpoint config_hash "
                    f"{start_state.config_hash[:12]} != current config "
                    f"{cfg.config_hash()[:12]}")
            print(f"  [resume] from {done[-1].name} t={start_state.time_h} h",
                  flush=True)
    if out_dir.exists() and start_state is None:
        # storage refuses checkpoint overwrites; a rerun after an aborted
        # or completed ladder must start clean (review finding)
        import shutil
        assert out_dir.name.startswith("leach_w4_")
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    state, summary = Engine(cfg).run(state=start_state,
                                     out_dir=str(out_dir), audit_hook=hook)
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
        "np_mode": np_mode,
        "np_counters": (np_counts if np_mode else None),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    dt_s = None
    until_h = None
    cfg_name = "opc32"
    np_mode = False
    resume = False
    while args and "=" in args[0]:
        key, _, val = args[0].partition("=")
        if key == "dt":
            dt_s = float(val)
        elif key == "resume":
            resume = val not in ("0", "false", "")
        elif key == "cfg":
            cfg_name = val
        elif key == "until":
            until_h = float(val)
        elif key == "np":
            np_mode = val not in ("0", "false", "")
        else:
            raise SystemExit(f"unknown option {args[0]!r}")
        args = args[1:]
    base = base_path(cfg_name)
    ladder = [float(x) for x in args] or (
        [1.0e-9] if np_mode else [0.8e-9, 2.0e-9, 5.3e-9])
    results = {}
    tags = []
    for d0 in ladder:
        tag = (f"np_dwdef_{d0:.1e}" if np_mode
               else f"d0_{d0:.1e}").replace("-", "m").replace("+", "")
        if dt_s is not None:
            tag = f"dt{dt_s:g}s_{tag}"
        if until_h is not None:
            tag = f"{tag}_u{until_h:g}h"
        if cfg_name != "opc32":
            tag = f"{cfg_name}_{tag}"
        tags.append(tag)
        out = REPO / "runs" / f"leach_w4_{tag}"
        print(f"== {tag} ==", flush=True)
        results[tag] = run_case(d0, out, dt_s, base, until_h,
                                np_mode=np_mode, resume=resume)
        r = results[tag]
        print(f"  wall {r['wall_s']} s, a = {r['a_um_per_sqrt_day']} "
              f"um/sqrt(day) ({r['n_fit_points']} pts), supply "
              f"max/mean = {r['supply_ratio_max']}/{r['supply_ratio_mean']}",
              flush=True)
        fronts = [(row['t_leach_h'], row['front_vox']) for row in r['rows']]
        print(f"  fronts: {fronts}", flush=True)
    # The results file is named after the CASES it holds, not just the
    # config: concurrent single-case invocations of the same ladder used to
    # write the same path and the last finisher clobbered the others
    # (measured at W4.2 - two of three cases had to be recovered from logs).
    out_json = REPO / "runs" / f"leach_w4_results_{'__'.join(tags)}.json"
    out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
