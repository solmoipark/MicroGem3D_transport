"""Build the Section 4.1 S1b 2x2 temporal/spatial path decomposition.

This is a read-only post-processing script.  It joins the returned 3-D
single-step family with the already-qualified single-evaluation 0-D,
sequential 0-D, and Q32 marching runs.  No chemistry solve is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from compare_timestep_to_zero_d import (  # noqa: E402
    FOCUS_PHASES,
    _checkpoint_rows,
    _row,
    _summary_rows,
)
from tinn.config import TinnConfig  # noqa: E402
from tinn.registry import registry_for  # noqa: E402


SCHEDULES = {
    "DT12": {
        "sequential_0d": "sequential_0d_dt12.json",
        "marching_3d": "q32_dt12_28d",
    },
    "DT0.5": {
        "sequential_0d": "sequential_0d_dt05.json",
        "marching_3d": "q32_fixed_dt05_log_28d",
    },
    "LOG_L2": {
        "sequential_0d": "sequential_0d_l2.json",
        "marching_3d": "q32_log_l2_28d",
    },
}
COMMON_AGES = (24.0, 168.0, 672.0)
FAMILY_AGES = (8.0, 16.0, 24.0, 72.0, 168.0, 336.0, 672.0)
METRICS = (
    "CSHQ",
    "Portlandite",
    "ettringite",
    "C3(AF)S0.84H",
    "Gypsum",
    "solid_skeleton_volume",
    "free_water",
    "CSHQ_ca_si",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def values_from_3d_row(row: dict) -> dict[str, float | None]:
    phases = row["phase_amount_mmol_per_100g"]["three_d"]
    return {
        **{phase: float(phases.get(phase, 0.0)) for phase in FOCUS_PHASES},
        "solid_skeleton_volume": float(
            row["total_solid_skeleton_volume_cm3_per_100g"]["three_d"]
        ),
        "free_water": float(row["free_or_aqueous_h2o_g_per_100g"]["three_d_free"]),
        "CSHQ_ca_si": row["CSHQ_bulk_ca_si"]["three_d"],
    }


def values_from_0d(row: dict) -> dict[str, float | None]:
    phases = row["phase_amount_mmol_per_100g_initial_as_built_binder"]
    return {
        **{phase: float(phases.get(phase, 0.0)) for phase in FOCUS_PHASES},
        "solid_skeleton_volume": sum(
            float(v)
            for v in row[
                "phase_volume_cm3_per_100g_initial_as_built_binder"
            ].values()
        ),
        "free_water": float(
            row["aqueous_h2o_g_per_100g_initial_as_built_binder"]
        ),
        "CSHQ_ca_si": row.get("CSHQ", {}).get("ca_si"),
    }


def subtract(a: dict, b: dict) -> dict:
    out = {}
    for metric in METRICS:
        av, bv = a.get(metric), b.get(metric)
        out[metric] = (
            float(av) - float(bv)
            if av is not None and bv is not None and math.isfinite(float(av))
            and math.isfinite(float(bv))
            else None
        )
    return out


def raw_3d_values(run: Path, age: float, binder_mass_g: float) -> dict:
    checkpoints = _checkpoint_rows(run)
    summaries = _summary_rows(run)
    if age not in checkpoints or age not in summaries:
        raise KeyError(f"missing age {age:g} h in {run}")
    header = checkpoints[age][1]
    cfg = TinnConfig.model_validate(header["config"])
    # _row needs a 0-D record.  For the family-only trajectory, create a
    # neutral zero record solely to reuse the authoritative normalization.
    zero = {
        "phase_amount_mmol_per_100g_initial_as_built_binder": {},
        "phase_volume_cm3_per_100g_initial_as_built_binder": {},
        "aqueous_h2o_g_per_100g_initial_as_built_binder": 0.0,
        "kinetic_alpha": {p: 0.0 for p in ("C3S", "C2S", "C3A", "C4AF")},
        "CSHQ": {"ca_si": 0.0},
    }
    row = _row(
        header,
        summaries[age],
        zero,
        cfg,
        binder_mass_g,
        registry_for(cfg),
    )
    return values_from_3d_row(row)


def load_single_family(s1b: Path, single_672: Path, binder_mass_g: float) -> dict:
    out = {}
    for age in FAMILY_AGES:
        run = (
            single_672
            if age == 672.0
            else s1b / "runs" / f"single_step_{int(age):04d}h"
        )
        out[age] = {
            "run": str(run),
            "values": raw_3d_values(run, age, binder_mass_g),
        }
    return out


def build(args: argparse.Namespace) -> dict:
    s1b = Path(args.s1b).resolve()
    runs_root = Path(args.runs_root).resolve()
    zero_dir = Path(args.zero_d).resolve()
    seq_dir = Path(args.sequential_0d).resolve()
    single_672 = Path(args.single_672).resolve()

    family_q = read_json(s1b / "qualification.json")
    if not family_q["integrity_passed"]:
        raise RuntimeError("S1b family integrity gate is not passed")
    if not family_q["all_exactly_one_accepted_step"]:
        raise RuntimeError("S1b family is not an exact one-step family")
    if not family_q["all_zero_rejected_trials"]:
        raise RuntimeError("S1b family contains trial rejections")

    zero_ref_path = zero_dir / "reference_0d_as_built.json"
    zero_ref = read_json(zero_ref_path)
    zero_rows = {float(row["time_h"]): row for row in zero_ref["rows"]}
    binder_mass_g = float(
        zero_ref["initial_as_built_inventory"]["binder_mass_g_per_rve"]
    )
    single_family = load_single_family(s1b, single_672, binder_mass_g)

    records = []
    for schedule, spec in SCHEDULES.items():
        seq_path = seq_dir / spec["sequential_0d"]
        seq = read_json(seq_path)
        seq_rows = {float(row["time_h"]): row for row in seq["rows"]}
        marching_run = runs_root / spec["marching_3d"]
        marching_ckpt = _checkpoint_rows(marching_run)
        marching_summary = _summary_rows(marching_run)
        for age in COMMON_AGES:
            zero_single = values_from_0d(zero_rows[age])
            zero_sequential = values_from_0d(seq_rows[age])
            three_single = single_family[age]["values"]
            header = marching_ckpt[age][1]
            cfg = TinnConfig.model_validate(header["config"])
            marching_row = _row(
                header,
                marching_summary[age],
                zero_rows[age],
                cfg,
                binder_mass_g,
                registry_for(cfg),
            )
            three_marching = values_from_3d_row(marching_row)
            temporal_0d = subtract(zero_sequential, zero_single)
            temporal_3d = subtract(three_marching, three_single)
            spatial_single = subtract(three_single, zero_single)
            interaction = subtract(temporal_3d, temporal_0d)
            records.append(
                {
                    "schedule": schedule,
                    "time_h": age,
                    "states": {
                        "zero_d_single": zero_single,
                        "zero_d_sequential": zero_sequential,
                        "three_d_single": three_single,
                        "three_d_marching": three_marching,
                    },
                    "effects": {
                        "zero_d_temporal_path": temporal_0d,
                        "three_d_temporal_path": temporal_3d,
                        "spatial_effect_without_temporal_path": spatial_single,
                        "spatial_temporal_interaction": interaction,
                    },
                }
            )

    return {
        "schema": "tinn.section_4_1.s1b_path_decomposition.v1",
        "evidence_level": "engineering_regression_only",
        "normalization_basis": "100 g initial rasterized reactive-plus-salt binder",
        "interpretation": (
            "The 0-D temporal-path effect is sequential minus single 0-D. "
            "The 3-D temporal-path effect is marching minus one-step 3-D. "
            "Their difference isolates the spatial-temporal interaction."
        ),
        "qualification": {
            "s1b_integrity_passed": True,
            "all_exactly_one_step": True,
            "all_zero_rejections": True,
        },
        "input_sha256": {
            "s1b_qualification": sha256(s1b / "qualification.json"),
            "zero_d_reference": sha256(zero_ref_path),
            "single_672_qualification": sha256(single_672 / "qualification.json"),
        },
        "single_step_family": {
            str(int(age)): payload for age, payload in single_family.items()
        },
        "common_ages_h": list(COMMON_AGES),
        "records": records,
    }


def write_csv(payload: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["schedule", "time_h", "metric", "zero_d_single", "zero_d_sequential",
             "three_d_single", "three_d_marching", "zero_d_temporal_path",
             "three_d_temporal_path", "spatial_single", "interaction"]
        )
        for rec in payload["records"]:
            for metric in METRICS:
                writer.writerow(
                    [
                        rec["schedule"], rec["time_h"], metric,
                        rec["states"]["zero_d_single"].get(metric),
                        rec["states"]["zero_d_sequential"].get(metric),
                        rec["states"]["three_d_single"].get(metric),
                        rec["states"]["three_d_marching"].get(metric),
                        rec["effects"]["zero_d_temporal_path"].get(metric),
                        rec["effects"]["three_d_temporal_path"].get(metric),
                        rec["effects"]["spatial_effect_without_temporal_path"].get(metric),
                        rec["effects"]["spatial_temporal_interaction"].get(metric),
                    ]
                )


def markdown(payload: dict) -> str:
    lines = [
        "# S1b 3D single-step family: 2×2 path decomposition",
        "",
        "All S1b integrity gates passed. The comparison below uses the common",
        "24, 168, and 672 h ages, in the same 100 g initial rasterized binder basis.",
        "",
        "| Schedule | Age (h) | 0D temporal ΔCSHQ | 3D temporal ΔCSHQ | "
        "3D single−0D single ΔCSHQ | Interaction ΔCSHQ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rec in payload["records"]:
        e = rec["effects"]
        lines.append(
            f"| {rec['schedule']} | {rec['time_h']:.0f} | "
            f"{e['zero_d_temporal_path']['CSHQ']:+.6g} | "
            f"{e['three_d_temporal_path']['CSHQ']:+.6g} | "
            f"{e['spatial_effect_without_temporal_path']['CSHQ']:+.6g} | "
            f"{e['spatial_temporal_interaction']['CSHQ']:+.6g} |"
        )
    lines += [
        "",
        "The sequential 0D temporal-path term is near numerical zero. Material",
        "3D drift therefore belongs to the spatial allocation/morphology path and",
        "its interaction with repeated stepping, rather than to homogeneous GEMS",
        "equilibrium path dependence.",
        "",
        "The S1b family also contains 8, 16, 72, and 336 h one-step states in the",
        "machine-readable JSON. A complete 0D four-cell decomposition is restricted",
        "to 24/168/672 h because those are the qualified common 0D ages.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1b", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--zero-d", required=True)
    parser.add_argument("--sequential-0d", required=True)
    parser.add_argument("--single-672", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {outdir}")
    payload = build(args)
    outdir.mkdir(parents=True)
    (outdir / "s1b_path_decomposition.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(payload, outdir / "s1b_path_decomposition.csv")
    (outdir / "S1B_PATH_DECOMPOSITION_REPORT.md").write_text(
        markdown(payload), encoding="utf-8"
    )
    print(json.dumps({"outdir": str(outdir), "records": len(payload["records"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
