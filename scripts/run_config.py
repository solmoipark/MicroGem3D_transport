# -*- coding: utf-8 -*-
"""Run one config from scratch - the generic driver behind the
qualification smokes (restart drivers live next to their studies).

    py -3 scripts/run_config.py <config.json> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                      # noqa: E402
from tinn.engine import Engine                          # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = TinnConfig.model_validate(json.loads(
        cfg_path.read_text(encoding="utf-8")))
    print(f"config {cfg_path} hash {cfg.config_hash()[:12]} -> {out}",
          flush=True)
    t0 = time.perf_counter()
    Engine(cfg).run(out_dir=str(out))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
