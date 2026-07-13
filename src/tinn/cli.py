"""CLI entry point. M0 provides validate-config; run/restart/report arrive with M1/M5."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from pydantic import ValidationError

from .config import TinnConfig


def _validate_config(path: str) -> int:
    try:
        cfg = TinnConfig.from_json_file(path)
    except FileNotFoundError:
        print(f"error: config file not found: {path}", file=sys.stderr)
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tinn", description="TINN cement hydration platform v2")
    sub = parser.add_subparsers(dest="command", required=True)
    p_val = sub.add_parser("validate-config", help="validate a config JSON file")
    p_val.add_argument("config_path")
    args = parser.parse_args(argv)

    if args.command == "validate-config":
        return _validate_config(args.config_path)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
