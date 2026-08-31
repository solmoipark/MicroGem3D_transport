"""Read-only unmet-release and operator-splitting attribution for Section 4.1.

The analysis uses only existing Q32 qualification outputs.  Quantities that
were not recorded in the historical audit schema are reported as unavailable;
they are never inferred from absence of a rejection event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


RUNS = {
    "Fixed_DT12": "q32_dt12_28d",
    "Fixed_DT6": "q32_dt06_28d",
    "Fixed_DT3": "q32_dt03_28d",
    "Fixed_DT1.5": "q32_dt015_28d",
    "Fixed_DT1": "q32_fixed_dt1_log_28d",
    "Fixed_DT0.5": "q32_fixed_dt05_log_28d",
    "Log_L0": "q32_log_l0_28d",
    "Log_L1": "q32_log_l1_28d",
    "Log_L2": "q32_log_l2_28d",
    "Log_L3": "q32_log_l3_28d",
}
AGES_H = (24.0, 168.0, 672.0)
FOCUS_PHASES = ("Gypsum", "ettringite", "SO4_OH_AFm", "OH_SO4_AFm")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def _checkpoints(run: Path) -> dict[float, tuple[Path, dict]]:
    rows = {}
    for path in run.glob("ckpt_*"):
        header = _read(path / "header.json")
        rows[float(header["time_h"])] = (path, header)
    return rows


def _mmol_per_100g(value: float, binder_mass_g: float) -> float:
    return float(value) * 100_000.0 / binder_mass_g


def _cluster_row_at(audit: list[dict], age: float) -> dict | None:
    candidates = [row for row in audit
                  if row.get("event") == "step_accepted"
                  and float(row.get("time_end_h", -1.0)) <= age + 1.0e-10]
    if not candidates:
        return None
    row = max(candidates, key=lambda item: float(item["time_end_h"]))
    fractions = row.get("metrics", {}).get("cluster_liq_frac", {})
    ph = row.get("metrics", {}).get("cluster_ph", {})
    return {
        "sample_time_h": float(row["time_end_h"]),
        "cluster_count_from_liquid_fraction": len(fractions),
        "cluster_count_from_pH": len(ph),
        "largest_cluster_liquid_fraction": max(
            (float(value) for value in fractions.values()), default=None),
    }


def _phase_amounts(header: dict, binder_mass_g: float) -> dict[str, float]:
    return {
        phase: _mmol_per_100g(header["hydrate_mol"][i], binder_mass_g)
        for i, phase in enumerate(header["hydrate_phase_ids"])
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return None
    value = float(np.corrcoef(xa, ya)[0, 1])
    return value if math.isfinite(value) else None


def _write_report(out: Path, result: dict) -> None:
    lines = [
        "# Unmet-release and operator-splitting attribution",
        "",
        "Evidence level: `engineering_regression_only`.",
        "",
        "This is a read-only analysis of the ten completed Q32 schedules.",
        "Historical audit files do not contain explicit per-step trace-water-freeze",
        "or spatial-fill-freeze counters, so those requested frequencies are marked",
        "`not_recorded` rather than inferred.",
        "",
        "## Run coverage",
        "",
        "| Schedule | Accepted steps | Rejections | Qualification |",
        "|---|---:|---:|---:|",
    ]
    for label, row in result["runs"].items():
        lines.append(
            f"| {label} | {row['accepted_steps']} | {row['trial_rejections']} | "
            f"{row['qualification_passed']} |")
    lines += ["", "## Unmet release at 672 h", ""]
    phases = result["kinetic_phase_ids"]
    lines += ["| Schedule | " + " | ".join(phases) + " |",
              "|---|" + "---:|" * len(phases)]
    for label, run in result["runs"].items():
        row = run["ages"]["672"]
        values = " | ".join(
            f"{float(row['unmet_mmol_per_100g'].get(phase, 0.0)):.6g}"
            for phase in phases)
        lines.append(f"| {label} | {values} |")
    lines += [
        "",
        "## Attribution limits",
        "",
        "- Unmet inventories and cluster-count evolution are directly observable.",
        "- Sulfate phase inventories are observable only at written checkpoint ages.",
        "- Trace-water and fill-freeze event frequencies are not present in the old audit schema.",
        "- Correlations across ten schedules are diagnostics, not causal estimates.",
        "",
    ]
    (out / "UNMET_ATTRIBUTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.runs_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite attribution evidence: {out}")

    result_runs: dict[str, dict] = {}
    source_hashes: dict[str, dict] = {}
    kinetic_ids: list[str] | None = None
    for label, relative in RUNS.items():
        run = root / relative
        qualification_path = run / "qualification.json"
        summary_path = run / "summary.json"
        audit_path = run / "step_audit.jsonl"
        qualification = _read(qualification_path)
        audit = _audit(audit_path)
        checkpoints = _checkpoints(run)
        if not all(age in checkpoints for age in AGES_H):
            raise RuntimeError(f"{label} lacks a common-age checkpoint")
        first_header = checkpoints[min(checkpoints)][1]
        ids = [str(value) for value in first_header["kinetic_phase_ids"]]
        if kinetic_ids is None:
            kinetic_ids = ids
        elif kinetic_ids != ids:
            raise RuntimeError(f"kinetic channel mismatch in {label}")
        binder_mass_g = 0.0
        config = first_header["config"]
        solid = config["binder"]["mass_fractions"]
        # The rasterized binder mass is derived from exact checkpoint initial mol
        # and registry-independent molar masses are already embedded in the
        # phase mol definition. Use the known as-built reference when available;
        # otherwise normalize unmet by total initial kinetic mol and mark it.
        reference = root / "zero_d_as_built_q32_v1_20260805" / "reference_0d_as_built.json"
        if reference.is_file():
            binder_mass_g = float(_read(reference)["initial_as_built_inventory"][
                "binder_mass_g_per_rve"])
        if binder_mass_g <= 0.0:
            raise RuntimeError("as-built binder mass reference is unavailable")

        age_rows = {}
        for age in AGES_H:
            path, header = checkpoints[age]
            unmet = {
                phase: _mmol_per_100g(header["unmet_mol"][i], binder_mass_g)
                for i, phase in enumerate(ids)
            }
            phases = _phase_amounts(header, binder_mass_g)
            age_rows[str(int(age))] = {
                "checkpoint": str(path),
                "unmet_mmol_per_100g": unmet,
                "total_unmet_mmol_per_100g": float(sum(unmet.values())),
                "sulfate_phase_mmol_per_100g": {
                    phase: float(phases.get(phase, 0.0)) for phase in FOCUS_PHASES
                },
                "cluster_diagnostics": _cluster_row_at(audit, age),
            }

        rejection_events = [row for row in audit if row.get("event") == "trial_rejected"]
        result_runs[label] = {
            "run": str(run),
            "accepted_steps": int(qualification["accepted_steps"]),
            "trial_rejections": len(rejection_events),
            "rejection_reasons": {
                reason: sum(1 for row in rejection_events if row.get("reason") == reason)
                for reason in sorted({row.get("reason") for row in rejection_events})
            },
            "qualification_passed": (
                float(qualification["final_time_h"]) == 672.0
                and int(qualification["rejected_trials"]) == 0
                and bool(qualification["all_checkpoint_readbacks_match"])
                and bool(qualification["bundle_sha256_unchanged"])
            ),
            "ages": age_rows,
            "event_observability": {
                "trace_water_freeze_frequency": "not_recorded",
                "spatial_fill_freeze_frequency": "not_recorded",
                "cluster_count": "recorded_in_step_accepted.metrics",
                "unmet_inventory": "recorded_at_checkpoints",
            },
        }
        source_hashes[label] = {
            "qualification": _sha256(qualification_path),
            "summary": _sha256(summary_path),
            "step_audit": _sha256(audit_path),
            "checkpoints": {
                str(int(age)): _sha256(checkpoints[age][0] / "header.json")
                for age in AGES_H
            },
        }

    assert kinetic_ids is not None
    correlations = {}
    for phase in FOCUS_PHASES:
        unmet = [result_runs[label]["ages"]["672"]["total_unmet_mmol_per_100g"]
                 for label in RUNS]
        amount = [result_runs[label]["ages"]["672"]["sulfate_phase_mmol_per_100g"][phase]
                  for label in RUNS]
        clusters = [float(result_runs[label]["ages"]["672"]["cluster_diagnostics"][
            "cluster_count_from_pH"]) for label in RUNS]
        correlations[phase] = {
            "pearson_total_unmet_vs_phase_amount": _pearson(unmet, amount),
            "pearson_cluster_count_vs_phase_amount": _pearson(clusters, amount),
            "n_schedules": len(RUNS),
        }

    result = {
        "schema": "tinn.section_4_1.unmet_attribution.v1",
        "evidence_level": "engineering_regression_only",
        "kinetic_phase_ids": kinetic_ids,
        "common_ages_h": list(AGES_H),
        "runs": result_runs,
        "cross_schedule_correlations_at_672h": correlations,
        "source_sha256": source_hashes,
        "limitations": [
            "Historical audits lack explicit trace-water-freeze counters.",
            "Historical audits lack explicit spatial-fill-freeze counters.",
            "Sulfate transition times are bounded by checkpoint output ages.",
            "Cross-schedule correlations are descriptive and not causal.",
        ],
    }
    out.mkdir(parents=True)
    (out / "unmet_attribution.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    _write_report(out, result)
    print(json.dumps({
        "out": str(out),
        "runs": len(result_runs),
        "all_qualification_gates_pass": all(
            row["qualification_passed"] for row in result_runs.values()),
        "not_recorded": ["trace_water_freeze_frequency", "spatial_fill_freeze_frequency"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
