"""Compare every completed Q32 timestep run with one common as-built 0D GEMS reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.analysis import gel_porosity_vector  # noqa: E402
from tinn.config import TinnConfig  # noqa: E402
from tinn.registry import KINETIC_PHASE_IDS, registry_for  # noqa: E402


RUNS = {
    "Fixed_DT12": "q32_dt12_28d",
    "Fixed_DT6": "q32_dt06_28d",
    "Fixed_DT3": "q32_dt03_28d",
    "Fixed_DT1.5": "q32_dt015_28d",
    "Log_L0": "q32_log_l0_28d",
    "Log_L1": "q32_log_l1_28d",
    "Log_L2": "q32_log_l2_28d",
    "Log_L3": "q32_log_l3_28d",
    "Fixed_DT1": "q32_fixed_dt1_log_28d",
    "Fixed_DT0.5": "q32_fixed_dt05_log_28d",
}
AGES_H = (24.0, 168.0, 672.0)
CLINKER_PHASES = ("C3S", "C2S", "C3A", "C4AF")
FOCUS_PHASES = (
    "CSHQ",
    "Portlandite",
    "ettringite",
    "C3(AF)S0.84H",
    "Gypsum",
    "SO4_OH_AFm",
    "OH_SO4_AFm",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_rows(run: Path) -> dict[float, tuple[Path, dict]]:
    rows = {}
    for checkpoint in run.glob("ckpt_*"):
        header = _read(checkpoint / "header.json")
        rows[float(header["time_h"])] = (checkpoint, header)
    return rows


def _summary_rows(run: Path) -> dict[float, dict]:
    return {
        float(row["time_h"]): row
        for row in _read(run / "summary.json")["outputs"]
    }


def _mmol_per_100g(mol: float, binder_mass_g: float) -> float:
    return float(mol) * 100_000.0 / binder_mass_g


def _safe(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _row(
    header: dict,
    summary: dict,
    zero: dict,
    cfg: TinnConfig,
    binder_mass_g: float,
    registry,
) -> dict:
    hydrate_ids = list(header["hydrate_phase_ids"])
    kinetic_ids = list(header["kinetic_phase_ids"])
    phase_3d = {
        phase: _mmol_per_100g(header["hydrate_mol"][i], binder_mass_g)
        for i, phase in enumerate(hydrate_ids)
    }
    phase_0d = {
        phase: float(value)
        for phase, value in zero[
            "phase_amount_mmol_per_100g_initial_as_built_binder"
        ].items()
    }
    phase_delta = {
        phase: phase_3d.get(phase, 0.0) - phase_0d.get(phase, 0.0)
        for phase in sorted(set(phase_3d) | set(phase_0d))
    }

    eps = gel_porosity_vector(cfg, tuple(hydrate_ids), registry)
    voxel_cm3 = (float(cfg.rve.voxel_size_um) * 1.0e-4) ** 3
    skeleton_cm3_per_100g = sum(
        float(header["hydrate_env_vol_vox"][i]) * (1.0 - float(eps[i]))
        for i in range(len(hydrate_ids))
    ) * voxel_cm3 * 100.0 / binder_mass_g
    zero_skeleton_cm3_per_100g = sum(
        float(value)
        for value in zero[
            "phase_volume_cm3_per_100g_initial_as_built_binder"
        ].values()
    )

    mw_h2o = registry.get("H2O").molar_mass_g_mol
    free_water_g_per_100g = (
        float(header["water_free_mol"]) * mw_h2o * 100.0 / binder_mass_g
    )
    zero_water_g_per_100g = float(
        zero["aqueous_h2o_g_per_100g_initial_as_built_binder"]
    )

    alpha_3d = {}
    alpha_delta = {}
    zero_alpha = zero["kinetic_alpha"]
    for phase in CLINKER_PHASES:
        i = kinetic_ids.index(phase)
        initial = float(header["initial_phase_mol"][i])
        remaining = float(header["phase_mol"][i])
        alpha_3d[phase] = 1.0 - remaining / initial if initial > 0.0 else 0.0
        alpha_delta[phase] = alpha_3d[phase] - float(zero_alpha[phase])

    ca_si_3d = (
        summary.get("solid_solution_composition", {})
        .get("CSHQ", {})
        .get("ca_si")
    )
    ca_si_0d = zero.get("CSHQ", {}).get("ca_si")
    ca_si_delta = (
        float(ca_si_3d) - float(ca_si_0d)
        if ca_si_3d is not None and ca_si_0d is not None else None
    )
    max_phase = max(
        ((abs(value), phase) for phase, value in phase_delta.items()),
        default=(0.0, None),
    )
    max_alpha = max(
        ((abs(value), phase) for phase, value in alpha_delta.items()),
        default=(0.0, None),
    )
    return {
        "time_h": float(header["time_h"]),
        "phase_amount_mmol_per_100g": {
            "three_d": phase_3d,
            "zero_d": phase_0d,
            "three_d_minus_zero_d": phase_delta,
            "focus_three_d_minus_zero_d": {
                phase: phase_delta.get(phase, 0.0) for phase in FOCUS_PHASES
            },
            "max_absolute_difference": {
                "phase": max_phase[1],
                "value_mmol_per_100g": max_phase[0],
            },
        },
        "total_solid_skeleton_volume_cm3_per_100g": {
            "three_d": skeleton_cm3_per_100g,
            "zero_d": zero_skeleton_cm3_per_100g,
            "three_d_minus_zero_d": (
                skeleton_cm3_per_100g - zero_skeleton_cm3_per_100g
            ),
        },
        "free_or_aqueous_h2o_g_per_100g": {
            "three_d_free": free_water_g_per_100g,
            "zero_d_aqueous": zero_water_g_per_100g,
            "three_d_minus_zero_d": free_water_g_per_100g - zero_water_g_per_100g,
        },
        "clinker_reaction_degree": {
            "three_d": alpha_3d,
            "zero_d_kinetic_target": {
                phase: float(zero_alpha[phase]) for phase in CLINKER_PHASES
            },
            "three_d_minus_zero_d": alpha_delta,
            "max_absolute_difference": {
                "phase": max_alpha[1],
                "value": max_alpha[0],
            },
        },
        "CSHQ_bulk_ca_si": {
            "three_d": _safe(ca_si_3d),
            "zero_d": _safe(ca_si_0d),
            "three_d_minus_zero_d": _safe(ca_si_delta),
        },
        "three_d_only": {
            "capillary_porosity": _safe(summary.get("porosity_capillary")),
            "gel_water_g_per_100g": (
                float(header["water_gel_mol"]) * mw_h2o * 100.0 / binder_mass_g
            ),
            "bound_water_g_per_100g": (
                float(header["water_bound_mol"]) * mw_h2o * 100.0 / binder_mass_g
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--zero-d", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    zero_dir = Path(args.zero_d).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {out}")

    zero_reference = _read(zero_dir / "reference_0d_as_built.json")
    zero_qualification = _read(zero_dir / "qualification.json")
    zero_rows = {float(row["time_h"]): row for row in zero_reference["rows"]}
    binder_mass_g = float(
        zero_reference["initial_as_built_inventory"]["binder_mass_g_per_rve"]
    )
    reference_phase_mol = zero_reference["initial_as_built_inventory"][
        "phase_mol_per_rve"
    ]
    reference_water_mol = float(
        zero_reference["initial_as_built_inventory"]["water_mol_per_rve"]
    )

    result_runs = {}
    run_gates = {}
    for label, relative in RUNS.items():
        run = runs_root / relative
        qualification = _read(run / "qualification.json")
        checkpoints = _checkpoint_rows(run)
        summaries = _summary_rows(run)
        first_header = checkpoints[min(checkpoints)][1]
        cfg = TinnConfig.model_validate(first_header["config"])
        registry = registry_for(cfg)
        initial = {
            phase: float(first_header["initial_phase_mol"][i])
            for i, phase in enumerate(KINETIC_PHASE_IDS)
        }
        inventory_matches = initial == {
            phase: float(reference_phase_mol[phase])
            for phase in KINETIC_PHASE_IDS
        } and float(first_header["initial_water_mol"]) == reference_water_mol
        ages_present = all(
            age in checkpoints and age in summaries and age in zero_rows
            for age in AGES_H
        )
        gates = {
            "qualification_final_time_672h": float(qualification["final_time_h"])
                == 672.0,
            "qualification_no_rejected_trials": int(
                qualification["rejected_trials"]
            ) == 0,
            "qualification_checkpoint_readback": bool(
                qualification["all_checkpoint_readbacks_match"]
            ),
            "qualification_bundle_immutable": bool(
                qualification["bundle_sha256_unchanged"]
            ),
            "initial_as_built_inventory_matches_zero_d": inventory_matches,
            "common_ages_present": ages_present,
        }
        run_gates[label] = {"passed": all(gates.values()), "checks": gates}
        if not all(gates.values()):
            raise RuntimeError(f"comparison gate failed for {label}: {gates}")
        result_runs[label] = {
            "run_dir": str(run),
            "config_hash": first_header["config_hash"],
            "accepted_steps": int(qualification["accepted_steps"]),
            "accepted_dt_h": qualification["accepted_dt_h"],
            "rows": [
                _row(
                    checkpoints[age][1],
                    summaries[age],
                    zero_rows[age],
                    cfg,
                    binder_mass_g,
                    registry,
                )
                for age in AGES_H
            ],
        }

    output = {
        "schema": "tinn.section_4_1.timestep_vs_zero_d.v1",
        "evidence_level": "engineering_regression_only",
        "interpretation": (
            "The 3D-minus-0D gap contains spatial constraint and morphology "
            "effects. Timestep sensitivity is the change of that gap across "
            "schedules, not the absolute gap itself."
        ),
        "normalization_basis": "100 g initial rasterized reactive-plus-salt binder",
        "common_ages_h": list(AGES_H),
        "zero_d_reference": str(zero_dir / "reference_0d_as_built.json"),
        "zero_d_qualification_passed": bool(zero_qualification["passed"]),
        "run_gates": run_gates,
        "runs": result_runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "runs": len(result_runs),
        "ages_h": list(AGES_H),
        "all_run_gates_pass": all(row["passed"] for row in run_gates.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
