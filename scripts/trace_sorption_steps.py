# -*- coding: utf-8 -*-
"""Step-by-step S-stage trace of one config (RT-Cl-3 diagnosis): prints
per accepted step the sorbed store, site total, buffer dissolution,
dry-reactor count and the sulfur split between store / solution.

    py -3 scripts/trace_sorption_steps.py <config.json> <out_dir>
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                      # noqa: E402
from tinn.engine import Engine                          # noqa: E402

KEYS = ("sorbed_total_mol", "sorbed_delta_mol", "sorption_sites_mol",
        "sorption_sites_c_mol", "sorption_buffer_dissolved_mol",
        "sorption_dry_reactors", "sorption_pool_coverage",
        "n_frozen_domains", "domains_selected", "domains_deferred")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = TinnConfig.model_validate(json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")))

    def hook(rec: dict) -> None:
        if rec.get("event") != "step_accepted":
            return
        m = rec.get("metrics") or {}
        flat = {}
        for k, v in m.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        found = {k: flat[k] for k in KEYS if k in flat}
        if not found:
            found = {k: v for k, v in flat.items() if "sorb" in k}
        t = rec.get("time_h", rec.get("time_end_h"))
        print(f"t={t} {json.dumps(found)}", flush=True)

    Engine(cfg).run(out_dir=sys.argv[2], audit_hook=hook)


if __name__ == "__main__":
    main()
