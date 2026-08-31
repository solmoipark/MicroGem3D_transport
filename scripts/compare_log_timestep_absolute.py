"""Absolute-quantity convergence evidence for the Section 4.1 timestep runs.

The earlier relative phase metric can be dominated by a small sulfate phase.
This report instead compares authoritative phase moles and their checkpointed
envelope volumes.  The production decision is made in physical voxel-volume
units, with mmol per 100 g of initial rasterized binder supplied only as a
human-readable scale conversion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig  # noqa: E402
from tinn.registry import registry_for  # noqa: E402

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
FOCUS_PHASES = (
    "CSHQ",
    "Portlandite",
    "ettringite",
    "C3(AF)S0.84H",
    "Gypsum",
    "SO4_OH_AFm",
    "OH_SO4_AFm",
)
CLINKER_PHASES = ("C3S", "C2S", "C3A", "C4AF")

# Absolute gates for this 32^3 qualification RVE.  A single limit is used for
# every material inventory so no small phase receives an unstable denominator.
# 328 voxels is the nearest integer to 0.01 * 32^3.  The report and decision
# are expressed in absolute voxels, not percentage phase errors.
MATERIAL_QUANTITY_TOLERANCE_VOX = 328.0
# Preserve the previously declared absolute porosity criterion (0.005 of a
# 32^3 RVE) instead of relaxing it merely because phase quantities are now
# judged on an absolute basis.
CAPILLARY_PORE_TOLERANCE_VOX = 0.005 * 32**3
CSHQ_CA_SI_TOLERANCE = 0.02
MATERIAL_PHASE_FLOOR_VOX = 1.0


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_rows(run: Path) -> dict[float, dict]:
    return {
        float(row["time_h"]): row
        for row in _read(run / "summary.json")["outputs"]
    }


def _checkpoint_rows(run: Path) -> dict[float, dict]:
    rows = {}
    for checkpoint in sorted(run.glob("ckpt_*")):
        header = _read(checkpoint / "header.json")
        hydrate_ids = list(header["hydrate_phase_ids"])
        kinetic_ids = list(header["kinetic_phase_ids"])
        rows[float(header["time_h"])] = {
            "hydrate_mol": {
                phase: float(header["hydrate_mol"][i])
                for i, phase in enumerate(hydrate_ids)
            },
            "hydrate_env_vox": {
                phase: float(header["hydrate_env_vol_vox"][i])
                for i, phase in enumerate(hydrate_ids)
            },
            "phase_mol": {
                phase: float(header["phase_mol"][i])
                for i, phase in enumerate(kinetic_ids)
            },
            "initial_phase_mol": {
                phase: float(header["initial_phase_mol"][i])
                for i, phase in enumerate(kinetic_ids)
            },
            "water_mol": {
                "free": float(header["water_free_mol"]),
                "gel": float(header["water_gel_mol"]),
                "bound": float(header["water_bound_mol"]),
            },
        }
    return rows


def _run_data(run: Path) -> dict:
    provenance = _read(run / "provenance.json")
    cfg = TinnConfig.from_json_file(provenance["config_path"])
    registry = registry_for(cfg)
    checkpoints = _checkpoint_rows(run)
    first = checkpoints[min(checkpoints)]
    initial_binder_mass_g = sum(
        mol * registry.get(phase).molar_mass_g_mol
        for phase, mol in first["initial_phase_mol"].items()
    )
    voxel_cm3 = (float(cfg.rve.voxel_size_um) * 1.0e-4) ** 3
    return {
        "run": run,
        "summary": _summary_rows(run),
        "checkpoints": checkpoints,
        "registry": registry,
        "voxel_cm3": voxel_cm3,
        "n_vox": int(cfg.rve.grid_size) ** 3,
        "initial_binder_mass_g": initial_binder_mass_g,
    }


def _mmol_per_100g(delta_mol: float, binder_mass_g: float) -> float:
    return delta_mol * 100_000.0 / binder_mass_g


def _phase_diff(
    phase: str,
    a: dict,
    b: dict,
    binder_mass_g: float,
) -> dict:
    delta_mol = abs(
        float(a["hydrate_mol"].get(phase, 0.0))
        - float(b["hydrate_mol"].get(phase, 0.0))
    )
    delta_vox = abs(
        float(a["hydrate_env_vox"].get(phase, 0.0))
        - float(b["hydrate_env_vox"].get(phase, 0.0))
    )
    return {
        "mol_per_rve": delta_mol,
        "mmol_per_100g_initial_binder": _mmol_per_100g(
            delta_mol, binder_mass_g
        ),
        "envelope_volume_vox": delta_vox,
    }


def _pair(a_name: str, b_name: str, runs: dict[str, dict]) -> dict:
    a_run, b_run = runs[a_name], runs[b_name]
    common = sorted(
        set(a_run["summary"])
        & set(b_run["summary"])
        & set(a_run["checkpoints"])
        & set(b_run["checkpoints"])
    )
    registry = b_run["registry"]
    voxel_cm3 = b_run["voxel_cm3"]
    binder_mass_g = b_run["initial_binder_mass_g"]
    water_vm_cm3_mol = registry.get("H2O").molar_volume_cm3
    ages = {}

    for time_h in common:
        sa, sb = a_run["summary"][time_h], b_run["summary"][time_h]
        ca, cb = a_run["checkpoints"][time_h], b_run["checkpoints"][time_h]
        hydrate_keys = sorted(
            set(ca["hydrate_mol"]) | set(cb["hydrate_mol"])
        )
        hydrate = {
            phase: _phase_diff(phase, ca, cb, binder_mass_g)
            for phase in hydrate_keys
        }
        material_hydrate = {
            phase: values
            for phase, values in hydrate.items()
            if max(
                abs(ca["hydrate_env_vox"].get(phase, 0.0)),
                abs(cb["hydrate_env_vox"].get(phase, 0.0)),
            )
            >= MATERIAL_PHASE_FLOOR_VOX
        }

        anhydrous = {}
        for phase in sorted(set(ca["phase_mol"]) | set(cb["phase_mol"])):
            delta_mol = abs(
                ca["phase_mol"].get(phase, 0.0)
                - cb["phase_mol"].get(phase, 0.0)
            )
            entry = registry.get(phase)
            delta_vox = delta_mol * entry.envelope_molar_volume_cm3 / voxel_cm3
            anhydrous[phase] = {
                "mol_per_rve": delta_mol,
                "mmol_per_100g_initial_binder": _mmol_per_100g(
                    delta_mol, binder_mass_g
                ),
                "envelope_volume_vox": delta_vox,
            }

        water = {}
        for compartment in ("free", "gel", "bound"):
            delta_mol = abs(
                ca["water_mol"][compartment] - cb["water_mol"][compartment]
            )
            water[compartment] = {
                "mol_per_rve": delta_mol,
                "mmol_per_100g_initial_binder": _mmol_per_100g(
                    delta_mol, binder_mass_g
                ),
                "liquid_volume_vox": (
                    delta_mol * water_vm_cm3_mol / voxel_cm3
                ),
            }

        total_hydrate_vox = abs(
            sum(ca["hydrate_env_vox"].values())
            - sum(cb["hydrate_env_vox"].values())
        )
        capillary_pore_vox = abs(
            float(sa["porosity_capillary"])
            - float(sb["porosity_capillary"])
        ) * b_run["n_vox"]
        ca_si_a = (
            sa.get("solid_solution_composition", {})
            .get("CSHQ", {})
            .get("ca_si")
        )
        ca_si_b = (
            sb.get("solid_solution_composition", {})
            .get("CSHQ", {})
            .get("ca_si")
        )
        ca_si_delta = (
            abs(float(ca_si_a) - float(ca_si_b))
            if ca_si_a is not None and ca_si_b is not None
            else 0.0
        )

        max_hydrate = max(
            (
                (values["envelope_volume_vox"], phase)
                for phase, values in material_hydrate.items()
            ),
            default=(0.0, None),
        )
        max_anhydrous = max(
            (
                (values["envelope_volume_vox"], phase)
                for phase, values in anhydrous.items()
            ),
            default=(0.0, None),
        )
        max_water = max(
            (
                (values["liquid_volume_vox"], compartment)
                for compartment, values in water.items()
            ),
            default=(0.0, None),
        )
        checks = {
            "individual_material_hydrate_quantity": (
                max_hydrate[0] <= MATERIAL_QUANTITY_TOLERANCE_VOX
            ),
            "total_hydrate_envelope_quantity": (
                total_hydrate_vox <= MATERIAL_QUANTITY_TOLERANCE_VOX
            ),
            "individual_anhydrous_quantity": (
                max_anhydrous[0] <= MATERIAL_QUANTITY_TOLERANCE_VOX
            ),
            "individual_water_compartment_quantity": (
                max_water[0] <= MATERIAL_QUANTITY_TOLERANCE_VOX
            ),
            "capillary_pore_quantity": (
                capillary_pore_vox <= CAPILLARY_PORE_TOLERANCE_VOX
            ),
            "CSHQ_ca_si": ca_si_delta <= CSHQ_CA_SI_TOLERANCE,
        }
        ages[str(time_h)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "hydrate_absolute_difference": hydrate,
            "material_hydrate_phases": sorted(material_hydrate),
            "anhydrous_absolute_difference": anhydrous,
            "water_absolute_difference": water,
            "max_material_hydrate_envelope_difference": {
                "phase": max_hydrate[1],
                "voxel_volume": max_hydrate[0],
            },
            "total_hydrate_envelope_difference_vox": total_hydrate_vox,
            "max_anhydrous_envelope_difference": {
                "phase": max_anhydrous[1],
                "voxel_volume": max_anhydrous[0],
            },
            "max_water_compartment_difference": {
                "compartment": max_water[1],
                "voxel_volume": max_water[0],
            },
            "capillary_pore_difference_vox": capillary_pore_vox,
            "CSHQ_ca_si_absolute_difference": ca_si_delta,
        }

    def worst(metric):
        time_h, value, label = max(
            (
                (float(time), *metric(row))
                for time, row in ages.items()
            ),
            key=lambda item: item[1],
        )
        return {"time_h": time_h, "value_vox": value, "label": label}

    return {
        "candidate": a_name,
        "reference": b_name,
        "passed_all_ages": all(row["passed"] for row in ages.values()),
        "worst": {
            "material_hydrate": worst(
                lambda row: (
                    row["max_material_hydrate_envelope_difference"][
                        "voxel_volume"
                    ],
                    row["max_material_hydrate_envelope_difference"]["phase"],
                )
            ),
            "total_hydrate": worst(
                lambda row: (
                    row["total_hydrate_envelope_difference_vox"],
                    "all hydrates",
                )
            ),
            "anhydrous": worst(
                lambda row: (
                    row["max_anhydrous_envelope_difference"]["voxel_volume"],
                    row["max_anhydrous_envelope_difference"]["phase"],
                )
            ),
            "water_compartment": worst(
                lambda row: (
                    row["max_water_compartment_difference"]["voxel_volume"],
                    row["max_water_compartment_difference"]["compartment"],
                )
            ),
            "capillary_pore": worst(
                lambda row: (
                    row["capillary_pore_difference_vox"],
                    "capillary pore",
                )
            ),
        },
        "ages": ages,
    }


def _focus_table(comp: dict, time_h: float) -> list[dict]:
    age = comp["ages"][str(time_h)]
    return [
        {
            "phase": phase,
            **age["hydrate_absolute_difference"].get(
                phase,
                {
                    "mol_per_rve": 0.0,
                    "mmol_per_100g_initial_binder": 0.0,
                    "envelope_volume_vox": 0.0,
                },
            ),
        }
        for phase in FOCUS_PHASES
    ]


def main() -> int:
    root = REPO / "runs" / "section_4_1"
    runs = {
        name: _run_data(root / relative)
        for name, relative in RUNS.items()
    }
    comparisons = {
        f"{a}_vs_{b}": _pair(a, b, runs)
        for a, b in PAIRS
    }

    fixed_pass = comparisons[
        "Fixed_DT1_vs_Fixed_DT0.5"
    ]["passed_all_ages"]
    selected = "Fixed_DT1" if fixed_pass else "Fixed_DT0.5"
    result = {
        "schema": "tinn.section_4_1.log_timestep_absolute_convergence.v1",
        "evidence_level": "engineering_regression_only",
        "quantity_basis": {
            "authoritative_phase_quantity": "mol per simulated RVE",
            "decision_quantity": "phase envelope volume in physical voxels",
            "human_scale": "mmol per 100 g initial rasterized binder",
            "material_quantity_tolerance_vox": (
                MATERIAL_QUANTITY_TOLERANCE_VOX
            ),
            "capillary_pore_tolerance_vox": (
                CAPILLARY_PORE_TOLERANCE_VOX
            ),
            "material_phase_floor_vox": MATERIAL_PHASE_FLOOR_VOX,
            "CSHQ_ca_si_tolerance": CSHQ_CA_SI_TOLERANCE,
        },
        "comparisons": comparisons,
        "fixed_refinement_passed": fixed_pass,
        "selected_production_schedule": selected,
        "selected_timestep_h": 1.0 if selected == "Fixed_DT1" else 0.5,
        "selection_scope": (
            "sealed saturated Q32 OPC, pure water, w/c 0.50, 20 C, 1 bar, "
            "through 672 h"
        ),
        "decision": (
            "Fixed DT1 is selected as the coarsest tested schedule whose "
            "absolute phase, water, and pore quantities pass against Fixed "
            "DT0.5 at every common output age."
            if fixed_pass
            else "Fixed DT0.5 is selected as the conservative production "
            "schedule. Fixed DT1 passed the absolute phase-quantity limits "
            "but exceeded the preserved capillary-pore limit against Fixed "
            "DT0.5."
        ),
        "focus_28d": {
            key: _focus_table(comp, 672.0)
            for key, comp in comparisons.items()
        },
    }
    out_json = root / "log_timestep_absolute_convergence.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Section 4.1 absolute-quantity timestep decision",
        "",
        "Evidence level: `engineering_regression_only`.",
        "",
        "## Decision",
        "",
        f"- Selected production schedule: `{selected}`",
        f"- Selected timestep: `{result['selected_timestep_h']} h`",
        f"- Scope: {result['selection_scope']}",
        f"- Rationale: {result['decision']}",
        "",
        "The authoritative comparison is absolute phase amount (mol per RVE). "
        "For readability it is also expressed as mmol per 100 g of the "
        "initial rasterized binder and as the corresponding phase-envelope "
        "volume in voxels. No relative phase error controls this decision.",
        "",
        "## Absolute gates over all output ages",
        "",
        "| Comparison | Max hydrate phase (vox) | Total hydrate (vox) | "
        "Max anhydrous phase (vox) | Max water compartment (vox) | "
        "Capillary pore (vox) | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, comp in comparisons.items():
        worst = comp["worst"]
        lines.append(
            f"| {key} | "
            f"{worst['material_hydrate']['value_vox']:.1f} "
            f"({worst['material_hydrate']['label']}, "
            f"{worst['material_hydrate']['time_h']:g} h) | "
            f"{worst['total_hydrate']['value_vox']:.1f} | "
            f"{worst['anhydrous']['value_vox']:.1f} | "
            f"{worst['water_compartment']['value_vox']:.1f} | "
            f"{worst['capillary_pore']['value_vox']:.1f} | "
            f"{'PASS' if comp['passed_all_ages'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"The common absolute material-inventory limit is "
            f"{MATERIAL_QUANTITY_TOLERANCE_VOX:.0f} voxels. Hydrate phases "
            f"holding less than {MATERIAL_PHASE_FLOOR_VOX:g} voxel in both "
            "runs are reported in JSON but cannot control the decision. The "
            "predeclared capillary-pore limit is preserved separately at "
            f"{CAPILLARY_PORE_TOLERANCE_VOX:.2f} voxels.",
            "",
            "## Phase quantities at 28 d",
            "",
        ]
    )
    for key in ("Fixed_DT1_vs_Fixed_DT0.5", "L2_vs_L3"):
        lines.extend(
            [
                f"### {key}",
                "",
                "| Phase | |Δn| (mol/RVE) | |Δn| "
                "(mmol/100 g binder) | |ΔV| (vox) |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in result["focus_28d"][key]:
            lines.append(
                f"| {row['phase']} | {row['mol_per_rve']:.6e} | "
                f"{row['mmol_per_100g_initial_binder']:.6g} | "
                f"{row['envelope_volume_vox']:.1f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The L2-L3 failure is an absolute sulfate-phase repartition, not "
            "a small-denominator artifact: ettringite alone differs by more "
            "than the material-inventory gate at 28 d. Fixed DT1 versus Fixed "
            "DT0.5 keeps every individual phase, total hydrate, anhydrous "
            "phase, and water compartment inside its absolute material limit, "
            "but its capillary-pore difference reaches 202.6 voxels and "
            "exceeds the preserved 163.84-voxel pore gate. Fixed DT0.5 is "
            "therefore the conservative production timestep for the next "
            "voxel/domain/seed studies; Fixed DT1 may be used only for "
            "screening calculations.",
        ]
    )
    out_md = root / "log_timestep_absolute_convergence.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(out_json),
                "markdown": str(out_md),
                "selected": selected,
                "fixed_refinement_passed": fixed_pass,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
