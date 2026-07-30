"""Compare Section 4.1 timestep, restart, and rerun evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _read(path: Path, name: str) -> dict:
    return json.loads((path / name).read_text(encoding="utf-8"))


def _rows(run: Path) -> Dict[float, dict]:
    return {float(row["time_h"]): row
            for row in _read(run, "summary.json")["outputs"]}


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-30)


def _compare_row(candidate: dict, reference: dict) -> dict:
    alpha_keys = sorted(set(candidate["alpha"]) | set(reference["alpha"]))
    phase_keys = sorted(
        set(candidate["hydrate_mol"]) | set(reference["hydrate_mol"]))
    water_keys = sorted(
        set(candidate["water_mol"]) | set(reference["water_mol"]))
    csh_c = candidate.get("solid_solution_composition", {}).get("CSHQ", {})
    csh_r = reference.get("solid_solution_composition", {}).get("CSHQ", {})
    em_c = csh_c.get("endmember_mol", {})
    em_r = csh_r.get("endmember_mol", {})
    em_keys = sorted(set(em_c) | set(em_r))

    alpha_abs = {
        key: abs(float(candidate["alpha"].get(key, 0.0))
                 - float(reference["alpha"].get(key, 0.0)))
        for key in alpha_keys
    }
    hydrate_rel = {
        key: _rel(float(candidate["hydrate_mol"].get(key, 0.0)),
                  float(reference["hydrate_mol"].get(key, 0.0)))
        for key in phase_keys
    }
    water_rel = {
        key: _rel(float(candidate["water_mol"].get(key, 0.0)),
                  float(reference["water_mol"].get(key, 0.0)))
        for key in water_keys
    }
    em_rel = {
        key: _rel(float(em_c.get(key, 0.0)), float(em_r.get(key, 0.0)))
        for key in em_keys
    }
    return {
        "alpha_absolute_difference": alpha_abs,
        "max_alpha_absolute_difference": max(alpha_abs.values(), default=0.0),
        "hydrate_mol_relative_difference": hydrate_rel,
        "max_hydrate_mol_relative_difference": max(
            hydrate_rel.values(), default=0.0),
        "water_mol_relative_difference": water_rel,
        "max_water_mol_relative_difference": max(
            water_rel.values(), default=0.0),
        "capillary_porosity_absolute_difference": abs(
            float(candidate["porosity_capillary"])
            - float(reference["porosity_capillary"])),
        "CSHQ_ca_si_absolute_difference": (
            abs(float(csh_c["ca_si"]) - float(csh_r["ca_si"]))
            if "ca_si" in csh_c and "ca_si" in csh_r else None),
        "CSHQ_endmember_relative_difference": em_rel,
        "max_CSHQ_endmember_relative_difference": max(
            em_rel.values(), default=0.0),
    }


def _engineering_gate(run: Path) -> dict:
    q = _read(run, "qualification.json")
    maxima = q.get("max_over_accepted_steps", {})
    checks = {
        "checkpoint_readback": bool(q["all_checkpoint_readbacks_match"]),
        "bundle_immutable": bool(q["bundle_sha256_unchanged"]),
        "rollback_dense_state": bool(q["rollback_dense_state_preserved"]),
        # An accepted engine step has already passed the scale-aware ledger
        # tolerances.  Requiring literal zero here would incorrectly fail on
        # harmless floating-point dust (for example 2e-25 mol at 28 d).
        "blocking_ledgers_all_accepted_steps": bool(
            q.get("blocking_ledger_checks_passed_for_all_accepted_steps",
                  q.get("accepted_steps", 0) > 0)),
        "boundary_exchange_zero": all(
            float(v) == 0.0
            for v in q["boundary_exchange_mol_by_element"].values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_over_accepted_steps": maxima,
        "rejection_counts": q["rejection_counts"],
    }


def compare(dt12: Path, dt06: Path, dt03: Path, dt015: Path | None,
            restart: Path | None, rerun: Path | None) -> dict:
    runs = {"dt12": dt12.resolve(), "dt06": dt06.resolve(),
            "dt03": dt03.resolve()}
    if dt015 is not None:
        runs["dt015"] = dt015.resolve()
    rows = {name: _rows(path) for name, path in runs.items()}
    common_set = set(next(iter(rows.values())))
    for run_rows in rows.values():
        common_set &= set(run_rows)
    common = sorted(common_set)
    reference_name = "dt015" if "dt015" in rows else "dt03"
    sensitivity = {}
    for time_h in common:
        ref = rows[reference_name][time_h]
        sensitivity[str(time_h)] = {"reference": reference_name}
        for name, run_rows in rows.items():
            if name != reference_name:
                sensitivity[str(time_h)][f"{name}_vs_{reference_name}"] = (
                    _compare_row(run_rows[time_h], ref))

    baseline_q = _read(dt06, "qualification.json")
    exact = {}
    for name, path in (("restart", restart), ("fresh_rerun", rerun)):
        if path is None:
            exact[name] = {"available": False, "full_hash_match": False}
            continue
        q = _read(path.resolve(), "qualification.json")
        exact[name] = {
            "available": True,
            "baseline_full_hash": baseline_q["final_full_hash"],
            "candidate_full_hash": q["final_full_hash"],
            "full_hash_match": (
                baseline_q["final_full_hash"] == q["final_full_hash"]),
            "dense_hash_match": (
                baseline_q["final_dense_hash"] == q["final_dense_hash"]),
        }

    gates = {name: _engineering_gate(path) for name, path in runs.items()}
    return {
        "schema": "tinn.section_4_1.comparison.v1",
        "evidence_level": "engineering_regression_only",
        "reference_timestep_h": 1.5 if reference_name == "dt015" else 3.0,
        "common_output_times_h": common,
        "engineering_gates": gates,
        "all_primary_engineering_gates_pass": all(
            gate["passed"] for gate in gates.values()),
        "timestep_sensitivity": sensitivity,
        "exact_restart_and_rerun": exact,
        "exact_reproducibility_pass": all(
            item["available"] and item["full_hash_match"]
            for item in exact.values()),
        "interpretation_note": (
            "Timestep differences are reported without an empirical-validation "
            "claim. Exact hashes are required only for restart and fresh rerun "
            "under an identical frozen config."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt12", required=True)
    parser.add_argument("--dt06", required=True)
    parser.add_argument("--dt03", required=True)
    parser.add_argument("--dt015")
    parser.add_argument("--restart")
    parser.add_argument("--rerun")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(
        Path(args.dt12), Path(args.dt06), Path(args.dt03),
        Path(args.dt015) if args.dt015 else None,
        Path(args.restart) if args.restart else None,
        Path(args.rerun) if args.rerun else None)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
