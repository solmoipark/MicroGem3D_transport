"""Qualify and compare the Q32 single-672 h-step diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from compare_timestep_to_zero_d import (  # noqa: E402
    _checkpoint_rows,
    _read,
    _row,
    _summary_rows,
)
from tinn.config import TinnConfig  # noqa: E402
from tinn.registry import KINETIC_PHASE_IDS, registry_for  # noqa: E402


FOCUS_PHASES = (
    "CSHQ",
    "Portlandite",
    "ettringite",
    "C3(AF)S0.84H",
    "Gypsum",
)


def _focus(row: dict) -> dict:
    phases = row["phase_amount_mmol_per_100g"]
    return {
        "phase_amount_mmol_per_100g": {
            phase: {
                "three_d": float(phases["three_d"].get(phase, 0.0)),
                "zero_d": float(phases["zero_d"].get(phase, 0.0)),
                "three_d_minus_zero_d": float(
                    phases["three_d_minus_zero_d"].get(phase, 0.0)
                ),
            }
            for phase in FOCUS_PHASES
        },
        "total_solid_skeleton_volume_cm3_per_100g": row[
            "total_solid_skeleton_volume_cm3_per_100g"
        ],
        "free_or_aqueous_h2o_g_per_100g": row[
            "free_or_aqueous_h2o_g_per_100g"
        ],
        "clinker_reaction_degree": row["clinker_reaction_degree"],
        "CSHQ_bulk_ca_si": row["CSHQ_bulk_ca_si"],
        "three_d_only": row["three_d_only"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-run", required=True)
    parser.add_argument("--zero-d", required=True)
    parser.add_argument("--baseline-comparison", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.diagnostic_run).resolve()
    zero_dir = Path(args.zero_d).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {out}")

    qualification = _read(run / "qualification.json")
    checkpoints = _checkpoint_rows(run)
    summaries = _summary_rows(run)
    zero_reference = _read(zero_dir / "reference_0d_as_built.json")
    zero_rows = {
        float(item["time_h"]): item for item in zero_reference["rows"]
    }
    header = checkpoints[672.0][1]
    cfg = TinnConfig.model_validate(header["config"])
    binder_mass_g = float(
        zero_reference["initial_as_built_inventory"]["binder_mass_g_per_rve"]
    )
    reference_phase_mol = zero_reference["initial_as_built_inventory"][
        "phase_mol_per_rve"
    ]
    reference_water_mol = float(
        zero_reference["initial_as_built_inventory"]["water_mol_per_rve"]
    )
    initial_phase_mol = {
        phase: float(header["initial_phase_mol"][i])
        for i, phase in enumerate(KINETIC_PHASE_IDS)
    }
    boundary_zero = all(
        float(value) == 0.0
        for value in qualification["boundary_exchange_mol_by_element"].values()
    )
    gates = {
        "final_time_672h": float(qualification["final_time_h"]) == 672.0,
        "exactly_one_accepted_step": int(qualification["accepted_steps"]) == 1,
        "accepted_step_is_672h": qualification["accepted_dt_h"]["unique"]
        == [672.0],
        "zero_rejected_trials": int(qualification["rejected_trials"]) == 0,
        "rollback_state_preserved": bool(
            qualification["rollback_dense_state_preserved"]
        ),
        "checkpoint_readback": bool(
            qualification["all_checkpoint_readbacks_match"]
        ),
        "bundle_immutable": bool(qualification["bundle_sha256_unchanged"]),
        "zero_boundary_exchange": boundary_zero,
        "initial_inventory_matches_zero_d": (
            all(
                math.isclose(
                    initial_phase_mol[phase],
                    float(reference_phase_mol[phase]),
                    rel_tol=2.0e-15,
                    abs_tol=1.0e-30,
                )
                for phase in KINETIC_PHASE_IDS
            )
            and math.isclose(
                float(header["initial_water_mol"]),
                reference_water_mol,
                rel_tol=2.0e-15,
                abs_tol=1.0e-30,
            )
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"single-step diagnostic gate failed: {gates}")

    diagnostic_row = _row(
        header,
        summaries[672.0],
        zero_rows[672.0],
        cfg,
        binder_mass_g,
        registry_for(cfg),
    )
    baseline = _read(Path(args.baseline_comparison).resolve())
    baseline_rows = {
        label: next(
            row for row in payload["rows"] if float(row["time_h"]) == 672.0
        )
        for label, payload in baseline["runs"].items()
    }

    output = {
        "schema": "tinn.section_4_1.single_step_672_diagnostic.v1",
        "evidence_level": "engineering_regression_only",
        "interpretation": (
            "This deliberately suppresses the repeated 3D temporal path. "
            "Closeness to 0D diagnoses operator-splitting/path dependence; it "
            "does not qualify a 672 h production timestep."
        ),
        "diagnostic_run": str(run),
        "gates": {"passed": all(gates.values()), "checks": gates},
        "diagnostic_672h": _focus(diagnostic_row),
        "existing_672h": {
            label: _focus(row) for label, row in baseline_rows.items()
        },
    }
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "gates_passed": output["gates"]["passed"],
                "accepted_steps": qualification["accepted_steps"],
                "accepted_dt_h": qualification["accepted_dt_h"]["unique"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
