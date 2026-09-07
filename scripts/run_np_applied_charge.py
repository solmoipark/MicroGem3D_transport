# -*- coding: utf-8 -*-
"""RT-01 measurement (external review 2026-09-07): run a 32^3 species-NP
leaching / ingress window and record the per-step distribution of
np_applied_charge_rel_max (the charge residual of the flux backward Euler
actually applied). Reuses the leach_w4 np_mode config construction.

Usage:
  py scripts/run_np_applied_charge.py cfg=leach_w4_opc32 dt=0.5 start=168.0 \
      until=168.2 out=runs/np_charge_w4 [d0=1.0e-9] [rtol=<val>]
  py scripts/run_np_applied_charge.py cfg=chloride_binding_28d_0.5M dt=0.5 \
      expose_only=1 out=runs/np_charge_cl05
The exposure window is [start, until] h; steps before `start` (sealed
hydration) are excluded from the distribution (NP is bath-inactive there).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig            # noqa: E402
from tinn.engine import Engine                # noqa: E402


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    cfg_name = args.get("cfg", "leach_w4_opc32")
    dt_s = float(args.get("dt", "0.5"))
    d0 = float(args.get("d0", "1.0e-9"))
    start_h = float(args.get("start", "168.0"))
    until_h = float(args.get("until", "168.2"))
    rtol = float(args["rtol"]) if "rtol" in args else None
    out_dir = REPO / args.get("out", "runs/np_charge_w4")

    raw = json.loads((REPO / "examples" / "qualification"
                      / f"{cfg_name}.json").read_text(encoding="utf-8"))
    dom = raw["transport"]["domains"]
    dom.pop("d0_m2_s", None)
    dom["species"] = {
        "dw_table": "gems_bundles/species_dw/species_dw.json",
        "default_dw_m2_s": d0, "geometry_factor": 1.0, "phi_clamp_report": True}
    if rtol is not None:
        dom["species"]["applied_charge_rtol"] = rtol

    # keep only outputs up to `until`, and drive the exposure window at dt_s
    dt_h = dt_s / 3600.0
    outs = [t for t in raw["schedule"]["output_times_h"] if t <= until_h + 1e-9]
    if until_h not in outs:
        outs.append(until_h)
    raw["schedule"]["output_times_h"] = sorted(set(outs))
    win = [w for w in raw["schedule"].get("dt_windows", [])
           if w["until_h"] <= start_h + 1e-9]
    win.append({"until_h": until_h, "dt_h": dt_h})
    raw["schedule"]["dt_windows"] = win
    raw["schedule"]["dt_min_h"] = min(raw["schedule"].get("dt_min_h", dt_h),
                                      dt_h / 4.0)

    cfg = TinnConfig.model_validate(raw)
    per_step = []           # (t_end_h, applied_rel, frozen_witness_rel)

    def hook(ev):
        if ev.get("event") == "step_accepted":
            m = ev.get("metrics", {})
            if "np_applied_charge_rel_max" in m and ev["time_end_h"] > start_h + 1e-9:
                per_step.append((float(ev["time_end_h"]),
                                 float(m["np_applied_charge_rel_max"]),
                                 float(m.get("np_charge_flux_rel_max", 0.0))))

    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    state, summary = Engine(cfg).run(out_dir=str(out_dir), audit_hook=hook)
    wall = time.perf_counter() - t0

    vals = [v for _, v, _ in per_step]
    frz = [w for _, _, w in per_step]
    result = {
        "measurement": "RT-01 applied-flux charge residual distribution",
        "config": cfg_name, "grid": cfg.rve.grid_size,
        "dt_s": dt_s, "default_dw_m2_s": d0,
        "window_h": [start_h, until_h], "applied_charge_rtol": rtol,
        "n_exposure_steps": len(vals),
        "np_applied_charge_rel_max": {
            "p50": _pct(vals, 50), "p95": _pct(vals, 95),
            "max": max(vals) if vals else None,
            "min": min(vals) if vals else None,
        },
        "np_charge_flux_rel_max_frozen_witness": {
            "p50": _pct(frz, 50), "p95": _pct(frz, 95),
            "max": max(frz) if frz else None,
        },
        "reject_counts": dict(state.reject_counts),
        "wall_s": round(wall, 1),
        "per_step_tail": per_step[-20:],
        "fallback_report": summary.get("fallback_report"),
        "gel_connectivity_warnings": summary.get("gel_connectivity_warnings"),
    }
    out_json = out_dir / "np_applied_charge_analysis.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[written] {out_json}")


if __name__ == "__main__":
    main()
