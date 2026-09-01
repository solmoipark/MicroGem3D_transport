# -*- coding: utf-8 -*-
"""RT-S1 SO4 sorption measured run (PRD 4.6.5): sealed Deschner OPC 32^3
to 28 d with the PHREEQC surface operator on CSHQ sites, sulfate
whitelist. Records the sorbed-S trajectory, site inventory and pool
coverage per output.

The surface constants default to the DEMONSTRATION values of the S1
gates (log_k 1.2, density 0.02/endmember) and the output is labeled
accordingly - the PRD-recorded run uses literature-calibrated values via
the CLI overrides once they are supplied:

    py -3 scripts/run_sorption_so4.py [log_k=<x>] [density=<mol/mol>]
                                      [tag=<name>]
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig            # noqa: E402
from tinn.engine import Engine                # noqa: E402
from tinn.registry import ELEMENT_IDS         # noqa: E402

BASE = REPO / "examples" / "qualification" / "deschner_opc_q32_dt06_28d.json"
SO4_REACTION = "Surf_sOH + SO4-2 = Surf_sSO4- + OH-"
SO4_ROW = {"S": 1.0, "O": 3.0, "H": -1.0}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    log_k = 1.2
    density = 0.02
    tag = "demo"
    for arg in sys.argv[1:]:
        key, _, val = arg.partition("=")
        if key == "log_k":
            log_k = float(val)
        elif key == "density":
            density = float(val)
        elif key == "tag":
            tag = val
        else:
            raise SystemExit(f"unknown option {arg!r}")

    raw = json.loads(BASE.read_text(encoding="utf-8"))
    raw["sorption"] = {
        "operator": "phreeqc_surface",
        "phreeqc_dat": str(REPO / "gems_bundles" / "PHREEQC-cemdata18"
                           / "cemdata18.dat"),
        "site_density_mol_per_mol": {dc: density for dc in
                                     ("CSHQ-TobH", "CSHQ-TobD",
                                      "CSHQ-JenH", "CSHQ-JenD")},
        "surface_species": [{"reaction": SO4_REACTION, "log_k": log_k,
                             "sorbed_elements": dict(SO4_ROW)}],
        "elements": ["S"],
    }
    cfg = TinnConfig.model_validate(raw)
    out_dir = REPO / "runs" / f"sorption_so4_{tag}"
    if out_dir.exists():
        import shutil
        assert out_dir.name.startswith("sorption_so4_")
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    state, summary = Engine(cfg).run(out_dir=str(out_dir))
    wall = time.perf_counter() - t0

    s_idx = ELEMENT_IDS.index("S")
    rows = []
    for out in summary["outputs"]:
        m = out.get("ledger_metrics", {})
        rows.append({
            "time_h": out.get("time_h"),
            "sorbed_total_mol": m.get("sorbed_total_mol"),
            "sorbed_delta_mol": m.get("sorbed_delta_mol"),
            "sorption_sites_mol": m.get("sorption_sites_mol"),
            "sorption_pool_coverage": m.get("sorption_pool_coverage"),
        })
    total_s = float(state.initial_elements[s_idx])
    sorbed_s = float(state.domain_sorbed_mol[:, s_idx].sum())
    aq_s = float(state.cluster_inventory[:, s_idx].sum())
    result = {
        "label": ("DEMONSTRATION constants - not literature-calibrated"
                  if tag == "demo" else tag),
        "log_k": log_k,
        "site_density_mol_per_mol": density,
        "wall_s": round(wall, 1),
        "outputs": rows,
        "final": {
            "time_h": state.time_h,
            "sorbed_S_mol": sorbed_s,
            "aqueous_S_mol": aq_s,
            "initial_S_mol": total_s,
            "sorbed_S_fraction_of_initial": (sorbed_s / total_s
                                             if total_s > 0 else None),
            "sites_mol": rows[-1]["sorption_sites_mol"] if rows else None,
            "occupancy": (sorbed_s / rows[-1]["sorption_sites_mol"]
                          if rows and rows[-1]["sorption_sites_mol"]
                          else None),
        },
    }
    out_json = REPO / "runs" / f"sorption_so4_results_{tag}.json"
    out_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["final"], indent=1))
    print(f"wall {wall:.0f} s; saved {out_json}")


if __name__ == "__main__":
    main()
