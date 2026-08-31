"""Build the Section 4.1 fixed/log-timestep convergence evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_section_4_1 import _compare_row  # noqa: E402

RUNS = {
    "L0": "q32_log_l0_28d",
    "L1": "q32_log_l1_28d",
    "L2": "q32_log_l2_28d",
    "L3": "q32_log_l3_28d",
    "Fixed_DT1": "q32_fixed_dt1_log_28d",
    "Fixed_DT0.5": "q32_fixed_dt05_log_28d",
}
PAIRS = (
    ("L0", "L1"),
    ("L1", "L2"),
    ("L2", "L3"),
    ("Fixed_DT1", "Fixed_DT0.5"),
)
CLINKER = ("C3S", "C2S", "C3A", "C4AF")
PRIMARY_HYDRATES = ("CSHQ", "Portlandite", "ettringite", "C3(AF)S0.84H")
THRESHOLDS = {
    "clinker_alpha_absolute": 0.01,
    "primary_hydrate_relative": 0.02,
    "water_redistribution_normalized": 0.02,
    "capillary_porosity_absolute": 0.005,
    "CSHQ_ca_si_absolute": 0.02,
}


def _load_rows(path: Path) -> dict[float, dict]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {float(row["time_h"]): row for row in payload["outputs"]}


def _qualification(path: Path) -> dict:
    q = json.loads((path / "qualification.json").read_text(encoding="utf-8"))
    checks = {
        "no_rejected_trials": int(q["rejected_trials"]) == 0,
        "checkpoint_readback": bool(q["all_checkpoint_readbacks_match"]),
        "bundle_immutable": bool(q["bundle_sha256_unchanged"]),
        "boundary_exchange_zero": all(
            float(v) == 0.0
            for v in q["boundary_exchange_mol_by_element"].values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "accepted_steps": int(q["accepted_steps"]),
        "rejected_trials": int(q["rejected_trials"]),
        "max_over_accepted_steps": q["max_over_accepted_steps"],
    }


def _pair(a_name: str, b_name: str, rows: dict[str, dict]) -> dict:
    common = sorted(set(rows[a_name]) & set(rows[b_name]))
    ages = {}
    for time_h in common:
        a, b = rows[a_name][time_h], rows[b_name][time_h]
        raw = _compare_row(a, b)
        alpha = max(raw["alpha_absolute_difference"][p] for p in CLINKER)
        hydrate = {
            p: raw["hydrate_mol_relative_difference"].get(p, 0.0)
            for p in PRIMARY_HYDRATES if b["hydrate_mol"].get(p, 0.0) > 0.0
        }
        max_hydrate = max(hydrate.values(), default=0.0)
        water_total = sum(float(v) for v in b["water_mol"].values())
        water_norm = (
            sum(abs(float(a["water_mol"][p]) - float(b["water_mol"][p]))
                for p in ("free", "gel", "bound")) / water_total
            if water_total > 0.0 else 0.0)
        metrics = {
            "clinker_alpha_absolute": alpha,
            "primary_hydrate_relative": max_hydrate,
            "primary_hydrate_relative_by_phase": hydrate,
            "water_redistribution_normalized": water_norm,
            "capillary_porosity_absolute":
                raw["capillary_porosity_absolute_difference"],
            "CSHQ_ca_si_absolute":
                raw["CSHQ_ca_si_absolute_difference"] or 0.0,
        }
        checks = {
            key: float(metrics[key]) <= limit
            for key, limit in THRESHOLDS.items()
        }
        ages[str(time_h)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
        }
    worst = {}
    for metric in THRESHOLDS:
        time_h, value = max(
            ((float(t), float(row["metrics"][metric]))
             for t, row in ages.items()),
            key=lambda item: item[1])
        worst[metric] = {"value": value, "time_h": time_h}
    return {
        "candidate": a_name,
        "reference": b_name,
        "passed_all_ages": all(row["passed"] for row in ages.values()),
        "worst_over_ages": worst,
        "ages": ages,
    }


def main() -> int:
    root = REPO / "runs" / "section_4_1"
    paths = {name: root / rel for name, rel in RUNS.items()}
    rows = {name: _load_rows(path) for name, path in paths.items()}
    gates = {name: _qualification(path) for name, path in paths.items()}
    comparisons = {
        f"{a}_vs_{b}": _pair(a, b, rows) for a, b in PAIRS
    }
    log_pairs_pass = {
        key: value["passed_all_ages"] for key, value in comparisons.items()
        if key.startswith("L")
    }
    selected = None
    # The coarser member of the finest passing adjacent pair is acceptable
    # only when the trend reaches that pair. No fallback/guess on failure.
    if comparisons["L2_vs_L3"]["passed_all_ages"]:
        selected = "L2"
    result = {
        "schema": "tinn.section_4_1.log_timestep_convergence.v1",
        "evidence_level": "engineering_regression_only",
        "decision_authority": "relative_phase_screen_only",
        "production_timestep_decision": (
            "Superseded by log_timestep_absolute_convergence.json, which "
            "uses absolute phase quantities and voxel-equivalent volumes."
        ),
        "thresholds": THRESHOLDS,
        "qualification_gates": gates,
        "all_qualification_gates_pass": all(g["passed"] for g in gates.values()),
        "comparisons": comparisons,
        "log_pair_pass": log_pairs_pass,
        "fixed_pair_pass":
            comparisons["Fixed_DT1_vs_Fixed_DT0.5"]["passed_all_ages"],
        "convergence_status": (
            "converged" if selected is not None else "not_converged"),
        "selected_coarsest_schedule": selected,
        "decision": (
            "No production timestep schedule selected: L2-L3 and fixed "
            "DT1-DT0.5 do not pass all predeclared criteria. Further "
            "refinement or an increment-controlled timestep formulation is "
            "required before voxel/domain/seed production calculations."
            if selected is None else
            f"{selected} is the coarsest schedule passing against the next "
            "refinement at every output age."
        ),
    }
    out_json = root / "log_timestep_convergence.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Section 4.1 log-timestep convergence",
        "",
        "Evidence level: `engineering_regression_only`.",
        "",
        "> This relative-phase screen is retained as a diagnostic. The "
        "production timestep decision is controlled by "
        "`log_timestep_absolute_convergence.json`, because a small phase can "
        "otherwise dominate a percentage difference.",
        "",
        "## Outcome",
        "",
        f"- Conservation/checkpoint/bundle gates: "
        f"{'PASS' if result['all_qualification_gates_pass'] else 'FAIL'}",
        f"- Timestep convergence: `{result['convergence_status']}`",
        f"- Selected schedule: `{selected or 'none'}`",
        f"- Decision: {result['decision']}",
        "",
        "## Worst difference over all output ages",
        "",
        "| Comparison | clinker α abs | primary hydrate rel | water norm | "
        "porosity abs | CSHQ Ca/Si abs | All ages |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, comp in comparisons.items():
        w = comp["worst_over_ages"]
        lines.append(
            f"| {key} | {w['clinker_alpha_absolute']['value']:.6g} | "
            f"{w['primary_hydrate_relative']['value']:.3%} | "
            f"{w['water_redistribution_normalized']['value']:.3%} | "
            f"{w['capillary_porosity_absolute']['value']:.6g} | "
            f"{w['CSHQ_ca_si_absolute']['value']:.6g} | "
            f"{'PASS' if comp['passed_all_ages'] else 'FAIL'} |")
    lines.extend([
        "",
        "The primary-hydrate metric covers CSHQ, Portlandite, ettringite, "
        "and C3(AF)S0.84H. Trace phases are retained in the raw run outputs "
        "but do not control this convergence decision. Water redistribution "
        "is the L1 difference of free/gel/bound water normalized by initial "
        "total water, avoiding unstable relative errors for a tiny component.",
        "",
        "## Qualification steps",
        "",
        "| Run | Accepted | Rejected | Gate |",
        "|---|---:|---:|---|",
    ])
    for name, gate in gates.items():
        lines.append(
            f"| {name} | {gate['accepted_steps']} | "
            f"{gate['rejected_trials']} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |")
    (root / "log_timestep_convergence.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "json": str(out_json),
        "status": result["convergence_status"],
        "selected": selected,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
