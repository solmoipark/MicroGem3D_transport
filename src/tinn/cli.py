"""CLI: validate-config / run / restart / report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from .config import TinnConfig


def _load_config(path: str) -> TinnConfig:
    return TinnConfig.from_json_file(path)


def _validate_config(path: str) -> int:
    try:
        cfg = _load_config(path)
    except OSError as e:
        print(f"error: cannot read config file {path}: {e}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as e:
        print(f"error: config file {path} is not UTF-8: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in {path}: {e}", file=sys.stderr)
        return 2
    except ValidationError as e:
        print(f"error: invalid config {path}:\n{e}", file=sys.stderr)
        return 2

    print(f"config OK: {path}")
    print(f"  binder phases : {cfg.binder.mass_fractions} (unassigned {cfg.binder.unassigned:.3f})")
    print(f"  w/c           : {cfg.w_c}")
    print(f"  temperature   : {cfg.temperature_K} K")
    print(f"  kinetics      : {cfg.kinetics.kind}"
          + (f" ({cfg.kinetics.preset})" if cfg.kinetics.preset else ""))
    print(f"  RVE           : {cfg.rve.grid_size}^3 @ {cfg.rve.voxel_size_um} um, seed {cfg.rve.seed}")
    print(f"  backend       : {cfg.chemistry.backend}")
    print(f"  config_hash   : {cfg.config_hash()}")
    return 0


def _write_summary(summary: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _print_outputs(summary: dict) -> None:
    for row in summary["outputs"]:
        alpha = ", ".join(f"{p}={v:.4f}" for p, v in row["alpha"].items() if v > 0)
        print(f"  t={row['time_h']:9.3f} h  {alpha}  "
              f"cap.porosity={row['porosity_capillary']:.4f}  "
              f"accepts={row['accept_count']} rejects={sum(row['reject_counts'].values())}")


def _run(config_path: str, out_dir: str) -> int:
    from .engine import Engine, EngineError
    from .storage import StorageError
    try:
        cfg = _load_config(config_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    engine = Engine(cfg)
    try:
        _, summary = engine.run(out_dir=out_dir)
    except EngineError as e:
        print(f"error: run aborted ({e.reason}): {e}", file=sys.stderr)
        return 3
    except StorageError as e:
        print(f"error: checkpoint write failed: {e}", file=sys.stderr)
        return 3
    _write_summary(summary, out_dir)
    print(f"run complete -> {out_dir}")
    _print_outputs(summary)
    return 0


def _restart(ckpt_path: str, out_dir: str) -> int:
    from .engine import Engine, EngineError
    from .registry import default_registry
    from .storage import StorageError, load_checkpoint
    try:
        state = load_checkpoint(ckpt_path, default_registry())
    except StorageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    engine = Engine(state.config)
    try:
        _, summary = engine.run(state=state, out_dir=out_dir)
    except EngineError as e:
        print(f"error: restart aborted ({e.reason}): {e}", file=sys.stderr)
        return 3
    except StorageError as e:
        print(f"error: checkpoint write failed: {e}", file=sys.stderr)
        return 3
    _write_summary(summary, out_dir)
    print(f"restart complete -> {out_dir}")
    _print_outputs(summary)
    return 0


def _report(run_dir: str, out_dir: Optional[str],
            kc_constant_m2: Optional[float] = None) -> int:
    from .analysis import report
    from .storage import StorageError
    try:
        result = report(run_dir, out_dir, kc_constant_m2=kc_constant_m2)
    except (FileNotFoundError, StorageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    target = out_dir or run_dir
    print(f"report written -> {Path(target) / 'report.json'}")
    for row in result["outputs"]:
        perc = "percolating" if row["percolation"]["any"] else "isolated"
        viol = f" VIOLATIONS: {row['ledger_violations']}" if row["ledger_violations"] else ""
        print(f"  t={row['time_h']:9.3f} h  cap.por={row['porosity_capillary']:.4f}  "
              f"tot.por={row['porosity_total']:.4f}  liquid {perc}  "
              f"[{row['slice_png']}]{viol}")
    band = result["sanity_band"]
    n_warn = sum(1 for c in band["checks"] if c["status"] == "warn")
    print(f"  sanity band: {len(band['checks']) - n_warn} pass, {n_warn} warn "
          f"({band['note']})")
    for c in band["checks"]:
        if c["status"] == "warn":
            print(f"    [warn] {c['check']}: {c['value']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tinn", description="TINN cement hydration platform v2")
    sub = parser.add_subparsers(dest="command", required=True)
    p_val = sub.add_parser("validate-config", help="validate a config JSON file")
    p_val.add_argument("config_path")
    p_run = sub.add_parser("run", help="run a simulation from a config")
    p_run.add_argument("config_path")
    p_run.add_argument("--out", required=True, help="output directory for checkpoints/summary")
    p_res = sub.add_parser("restart", help="continue a run from a checkpoint directory")
    p_res.add_argument("checkpoint_path")
    p_res.add_argument("--out", required=True, help="output directory for new checkpoints/summary")
    p_rep = sub.add_parser("report", help="generate report.json + slice PNGs from a run directory")
    p_rep.add_argument("run_dir")
    p_rep.add_argument("--out", default=None, help="output directory (default: the run directory)")
    p_rep.add_argument("--kc-constant-m2", type=float, default=None,
                       help="Kozeny-Carman C override (default: d_mean^2/180)")
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        return _validate_config(args.config_path)
    if args.command == "run":
        return _run(args.config_path, args.out)
    if args.command == "restart":
        return _restart(args.checkpoint_path, args.out)
    return _report(args.run_dir, args.out, args.kc_constant_m2)


if __name__ == "__main__":
    sys.exit(main())
