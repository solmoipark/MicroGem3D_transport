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
                                      [tag=<name>] | calibrated

`calibrated` uses the literature constants fitted on 2026-09-02:
log_K +0.50 from replaying Divet & Randriambololona (CCR 28(3) 1998,
Fig. 3 0.1 M NaOH isotherm, 3.4 g C-S-H / 250 mL, 25 C) through this
platform's SorptionOperator, with the site total FIXED from Labbez et
al. (J Phys Chem B 110, 2006: 4.8 silanol/nm2) x Divet's BET SSA
(350 m2/g) = 2.79 mmol/g = 0.4766 mol sites per mol Si (C/S 1.57,
H/S 1.26 -> 170.8 g/mol-Si). Per-endmember density = 0.4766 x Si
stoichiometry from the PC bundle DCH (TobH/JenH Si=1, TobD/JenD
Si=0.6667). Haas & Nonat (CCR 68, 2015) give 0.5-1.0 titratable
silanol per Si - same order. Caveat (mechanism table): the single
ligand-exchange reaction releases OH-, so its pH trend is OPPOSITE to
Divet's measured ionic-strength enhancement - constants are valid
near the calibration pH (~12.9-13.4, the OPC pore-solution regime)
and must not be extrapolated across pH. Fit script + digitized
points: scripts/fit_so4_logk.py (fitted 2026-09-02).
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

# literature-calibrated constants (see module docstring for provenance)
CAL_LOG_K = 0.50
CAL_SITES_PER_SI = 0.4766           # mol sites / mol Si
CAL_SI = {"CSHQ-TobH": 1.0, "CSHQ-TobD": 0.6667,
          "CSHQ-JenH": 1.0, "CSHQ-JenD": 0.6667}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    log_k = 1.2
    density = 0.02
    tag = "demo"
    density_map = None
    for arg in sys.argv[1:]:
        key, _, val = arg.partition("=")
        if arg == "calibrated":
            log_k = CAL_LOG_K
            density_map = {dc: CAL_SITES_PER_SI * si
                           for dc, si in CAL_SI.items()}
            tag = "calibrated"
        elif key == "log_k":
            log_k = float(val)
        elif key == "density":
            density = float(val)
        elif key == "tag":
            tag = val
        else:
            raise SystemExit(f"unknown option {arg!r}")
    if density_map is None:
        density_map = {dc: density for dc in CAL_SI}

    raw = json.loads(BASE.read_text(encoding="utf-8"))
    raw["sorption"] = {
        "operator": "phreeqc_surface",
        "phreeqc_dat": str(REPO / "gems_bundles" / "PHREEQC-cemdata18"
                           / "cemdata18.dat"),
        "site_density_mol_per_mol": density_map,
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
            "sorption_dry_reactors": m.get("sorption_dry_reactors"),
        })
    total_s = float(state.initial_elements[s_idx])
    sorbed_s = float(state.domain_sorbed_mol[:, s_idx].sum())
    aq_s = float(state.cluster_inventory[:, s_idx].sum())
    result = {
        "label": ("DEMONSTRATION constants - not literature-calibrated"
                  if tag == "demo" else
                  ("literature-calibrated: Divet 1998 log_K fit + "
                   "Labbez 2006 site density" if tag == "calibrated"
                   else tag)),
        "log_k": log_k,
        "site_density_mol_per_mol": density_map,
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
