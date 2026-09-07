# -*- coding: utf-8 -*-
"""RT-06 (external review 2026-09-07): S-R splitting dt-convergence ladder.

Hydrate a sorption+bath example ONCE to the exposure onset, then resume the
SAME 672 h (or 168 h) state at several exposure dt values and compare the
free/sorbed Cl, Friedel, free/sorbed S, CH volume and (charge-balance) pH at
the output times. First-order convergence is expected: the dt/2 -> dt/4 change
should be smaller than the dt -> dt/2 change, and the 0.4 h binding should move
< 2 % across the ladder. run_config-style resume (no config_hash gate).

Usage:
  py scripts/run_rt06_dt_ladder.py cfg=chloride_binding_28d_0.5M start=672.0 \
      end=672.4 dts=0.0014,0.0007,0.00035 out=runs/rt06_cl05
  py scripts/run_rt06_dt_ladder.py cfg=sulfate_attack_opc32 start=168.0 \
      end=168.2 dts=0.0014,0.0007 out=runs/rt06_so4
"""
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                  # noqa: E402
from tinn.engine import Engine                      # noqa: E402
from tinn.registry import registry_for              # noqa: E402
from tinn.storage import load_checkpoint            # noqa: E402


def _cfg_from(base, outputs, windows):
    raw = json.loads(json.dumps(base))
    raw["schedule"]["output_times_h"] = outputs
    raw["schedule"]["dt_windows"] = windows
    dtmin = min([w["dt_h"] for w in windows] + [raw["schedule"]["dt_min_h"]])
    raw["schedule"]["dt_min_h"] = min(raw["schedule"]["dt_min_h"], dtmin / 4.0)
    return TinnConfig.model_validate(raw)


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    cfg_name = args["cfg"]
    start_h = float(args["start"])
    end_h = float(args.get("end", start_h + 0.4))
    dts = [float(x) for x in args["dts"].split(",")]
    out_root = REPO / args.get("out", "runs/rt06")
    out_root.mkdir(parents=True, exist_ok=True)

    base = json.loads((REPO / "examples" / "qualification"
                       / f"{cfg_name}.json").read_text(encoding="utf-8"))
    full_outputs = [t for t in base["schedule"]["output_times_h"]
                    if t <= end_h + 1e-9]
    if start_h not in full_outputs:
        raise SystemExit(f"{start_h} not an output time of {cfg_name}")
    start_idx = sorted(set(full_outputs)).index(start_h)

    # ---- phase 1: hydrate once to start_h ----
    hyd_dir = out_root / "hydrate"
    hyd_ckpt = hyd_dir / f"ckpt_{start_idx:03d}"
    if not hyd_ckpt.exists():
        if hyd_dir.exists():
            shutil.rmtree(hyd_dir)
        h_outputs = [t for t in full_outputs if t <= start_h + 1e-9]
        h_windows = [w for w in base["schedule"]["dt_windows"]
                     if w["until_h"] <= start_h + 1e-9]
        hcfg = _cfg_from(base, h_outputs, h_windows)
        print(f"[hydrate] {cfg_name} -> {start_h} h "
              f"(hash {hcfg.config_hash()[:12]})", flush=True)
        t0 = time.perf_counter()
        Engine(hcfg).run(out_dir=str(hyd_dir))
        print(f"[hydrate] done {time.perf_counter()-t0:.0f} s", flush=True)
    else:
        print(f"[hydrate] reuse {hyd_ckpt}", flush=True)

    # ---- phase 2: per-dt exposure resume from the shared checkpoint ----
    summary = {"cfg": cfg_name, "start_h": start_h, "end_h": end_h, "runs": {}}
    for dt in dts:
        tag = f"dt_{dt:g}"
        run_dir = out_root / tag
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        shutil.copytree(hyd_ckpt, run_dir / f"ckpt_{start_idx:03d}")
        windows = [w for w in base["schedule"]["dt_windows"]
                   if w["until_h"] <= start_h + 1e-9]
        windows.append({"until_h": end_h, "dt_h": dt})
        ecfg = _cfg_from(base, full_outputs, windows)
        state = load_checkpoint(str(run_dir / f"ckpt_{start_idx:03d}"),
                                registry_for(ecfg))
        print(f"[{tag}] resume t={state.time_h} h, expose to {end_h} h dt={dt}",
              flush=True)
        t0 = time.perf_counter()
        st, _ = Engine(ecfg).run(state=state, out_dir=str(run_dir))
        wall = time.perf_counter() - t0
        summary["runs"][tag] = {"dt_h": dt, "wall_s": round(wall, 1),
                                "reject_counts": dict(st.reject_counts)}
        print(f"[{tag}] done {wall:.0f} s rejects={dict(st.reject_counts)}",
              flush=True)

    (out_root / "rt06_ladder_runs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[written] {out_root/'rt06_ladder_runs.json'}", flush=True)


if __name__ == "__main__":
    main()
