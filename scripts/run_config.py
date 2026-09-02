# -*- coding: utf-8 -*-
"""Run one config from scratch, or resume it from the latest checkpoint
in <out_dir> - the generic driver behind the qualification smokes.

    py -3 scripts/run_config.py <config.json> <out_dir> [resume=1]
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                      # noqa: E402
from tinn.engine import Engine                          # noqa: E402
from tinn.registry import registry_for                  # noqa: E402
from tinn.storage import load_checkpoint                # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = TinnConfig.model_validate(json.loads(
        cfg_path.read_text(encoding="utf-8")))
    print(f"config {cfg_path} hash {cfg.config_hash()[:12]} -> {out}",
          flush=True)
    t0 = time.perf_counter()
    state = None
    if any(a == "resume=1" for a in sys.argv[3:]):
        ckpts = sorted(out.glob("ckpt_*"))
        if not ckpts:
            raise SystemExit(f"resume=1 but no checkpoint under {out}")
        state = load_checkpoint(str(ckpts[-1]), registry_for(cfg))
        print(f"resuming from {ckpts[-1].name} at t={state.time_h} h",
              flush=True)
    Engine(cfg).run(state=state, out_dir=str(out))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
