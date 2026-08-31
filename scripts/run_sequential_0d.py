"""Sequential as-built 0D schedule test for Section 4.1.

Each accepted 3D schedule boundary is reproduced without spatial allocation.
The complete previous equilibrium assemblage is converted back to its element
inventory, the next kinetic release increment is added, and the closed system
is re-equilibrated.  Soluble salt carriers are offered in full on the first
step, matching the globally accessible 0D interpretation.

This is an engineering-regression diagnostic.  It does not emulate unmet
spatial release, morphology placement, cluster splitting, or transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig  # noqa: E402
from tinn.gems import (  # noqa: E402
    AQUEOUS_PHASE,
    CANONICAL_MAX_ELEMENT_MOL,
    GAS_PHASE,
    O2_SEED_MOL_O,
    SUPPRESSED_CLINKER_PHASES,
    GemsWorker,
    audit_bundle,
)
from tinn.kinetics import make_kinetics  # noqa: E402
from tinn.registry import (  # noqa: E402
    ELEMENT_IDS,
    KINETIC_PHASE_IDS,
    SALT_PHASE_IDS,
    element_vector,
    registry_for,
)


AGES_H = (24.0, 168.0, 672.0)
DIRECT_CLOSURE_LIMIT = 1.0e-10
FLOOR_ADJUST_LIMIT = 1.0e-9
PATH_ELEMENT_DRIFT_LIMIT = 2.0e-9
TIME_TOL = 1.0e-10


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_true(values: Iterable[bool]) -> bool:
    return all(bool(value) for value in values)


def _schedule_boundaries(cfg: TinnConfig, horizon_h: float) -> list[float]:
    """Reproduce a rejection-free Engine boundary sequence."""
    schedule = cfg.schedule
    outputs = [float(t) for t in schedule.output_times_h if t <= horizon_h + TIME_TOL]
    if not outputs or abs(outputs[-1] - horizon_h) > TIME_TOL:
        outputs.append(float(horizon_h))
    outputs = sorted(set(outputs))
    boundaries: list[float] = []
    time_h = 0.0
    output_index = 0
    while time_h < horizon_h - TIME_TOL:
        while output_index < len(outputs) and outputs[output_index] <= time_h + TIME_TOL:
            output_index += 1
        next_output = outputs[output_index] if output_index < len(outputs) else horizon_h
        dt = min(float(schedule.dt_cap_at(time_h)), next_output - time_h)
        window_end = schedule.next_window_end_after(time_h)
        if window_end is not None:
            dt = min(dt, float(window_end) - time_h)
        if dt <= TIME_TOL:
            raise RuntimeError(f"non-positive schedule step at {time_h} h")
        time_h = min(horizon_h, time_h + dt)
        if abs(time_h - next_output) <= TIME_TOL:
            time_h = next_output
        boundaries.append(float(time_h))
    return boundaries


def _scale_elements(elements: np.ndarray, *, add_o2_seed: bool) -> tuple[dict, float]:
    peak = float(np.max(elements))
    if peak <= 0.0:
        raise ValueError("empty sequential 0D element inventory")
    scale = CANONICAL_MAX_ELEMENT_MOL / peak
    payload = {
        element: float(elements[i] * scale)
        for i, element in enumerate(ELEMENT_IDS)
        if elements[i] > 0.0
    }
    if add_o2_seed:
        payload["O"] = payload.get("O", 0.0) + O2_SEED_MOL_O
    return payload, scale


def _output_elements(result, scale: float) -> np.ndarray:
    values = np.zeros(len(ELEMENT_IDS), dtype=np.float64)
    index = {element: i for i, element in enumerate(ELEMENT_IDS)}
    for row in result.phase_elements_mol.values():
        for element, value in row.items():
            if element in index:
                values[index[element]] += float(value) / scale
    return values


def _solid_maps(result, scale: float) -> tuple[dict, dict, dict, dict, dict]:
    excluded = {AQUEOUS_PHASE, GAS_PHASE, *SUPPRESSED_CLINKER_PHASES}
    amounts = {
        phase: float(value) / scale
        for phase, value in result.phase_amounts_mol.items()
        if phase not in excluded and float(value) > 0.0
    }
    masses = {
        phase: float(result.phase_masses_kg.get(phase, 0.0)) / scale
        for phase in amounts
    }
    volumes = {
        phase: float(result.phase_volumes_m3.get(phase, 0.0)) / scale
        for phase in amounts
    }
    elements = {
        phase: {
            element: float(value) / scale
            for element, value in result.phase_elements_mol.get(phase, {}).items()
        }
        for phase in amounts
    }
    species = {
        phase: {
            name: float(value) / scale
            for name, value in result.phase_species_mol.get(phase, {}).items()
        }
        for phase in amounts
    }
    return amounts, masses, volumes, elements, species


def _normalize(values: Mapping[str, float], binder_mass_g: float, factor: float) -> dict:
    return {
        str(key): float(value) * factor * 100.0 / binder_mass_g
        for key, value in values.items()
    }


def _normalize_nested(values: Mapping[str, Mapping[str, float]], binder_mass_g: float) -> dict:
    return {
        str(key): _normalize(row, binder_mass_g, 1000.0)
        for key, row in values.items()
    }


def _cshq(elements: Mapping[str, Mapping[str, float]], species: Mapping[str, Mapping[str, float]]) -> dict:
    row = elements.get("CSHQ", {})
    ca = float(row.get("Ca", 0.0))
    si = float(row.get("Si", 0.0))
    endmembers = species.get("CSHQ", {})
    total = float(sum(endmembers.values()))
    return {
        "ca_si": ca / si if si > 0.0 else None,
        "endmember_mole_fraction": {
            key: float(value) / total for key, value in endmembers.items()
        } if total > 0.0 else {},
    }


def _snapshot_row(result, scale: float, time_h: float, alpha: np.ndarray,
                  binder_mass_g: float, registry) -> dict:
    amounts, masses, volumes, elements, species = _solid_maps(result, scale)
    aqueous = {
        element: float(value) / scale
        for element, value in result.phase_elements_mol.get(AQUEOUS_PHASE, {}).items()
    }
    water_mol = float(result.aqueous_h2o_mol) / scale
    return {
        "time_h": float(time_h),
        "kinetic_alpha": {
            phase: float(alpha[i]) for i, phase in enumerate(KINETIC_PHASE_IDS)
        },
        "phase_amount_mmol_per_100g_initial_as_built_binder":
            _normalize(amounts, binder_mass_g, 1000.0),
        "phase_mass_g_per_100g_initial_as_built_binder":
            _normalize(masses, binder_mass_g, 1000.0),
        "phase_volume_cm3_per_100g_initial_as_built_binder":
            _normalize(volumes, binder_mass_g, 1.0e6),
        "phase_element_mmol_per_100g_initial_as_built_binder":
            _normalize_nested(elements, binder_mass_g),
        "phase_endmember_mmol_per_100g_initial_as_built_binder":
            _normalize_nested(species, binder_mass_g),
        "aqueous_element_mmol_per_100g_initial_as_built_binder":
            _normalize(aqueous, binder_mass_g, 1000.0),
        "aqueous_h2o_mol_per_100g_initial_as_built_binder":
            water_mol * 100.0 / binder_mass_g,
        "aqueous_h2o_g_per_100g_initial_as_built_binder":
            water_mol * registry.get("H2O").molar_mass_g_mol * 100.0 / binder_mass_g,
        "pH": float(result.ph),
        "ionic_strength_molal": float(result.ionic_strength),
        "CSHQ": _cshq(elements, species),
        "element_closure_max_rel": float(result.element_closure_max_rel()),
        "floor_adjust_max_rel": float(result.floor_adjust_max_rel()),
        "xgems_version": result.xgems_version,
    }


def _metric_delta(sequential: dict, single: dict) -> dict:
    seq_phase = sequential["phase_amount_mmol_per_100g_initial_as_built_binder"]
    one_phase = single["phase_amount_mmol_per_100g_initial_as_built_binder"]
    phases = sorted(set(seq_phase) | set(one_phase))
    return {
        "time_h": float(sequential["time_h"]),
        "phase_mmol_per_100g_sequential_minus_single": {
            phase: float(seq_phase.get(phase, 0.0)) - float(one_phase.get(phase, 0.0))
            for phase in phases
        },
        "solid_skeleton_volume_cm3_per_100g": {
            "sequential": float(sum(sequential[
                "phase_volume_cm3_per_100g_initial_as_built_binder"].values())),
            "single": float(sum(single[
                "phase_volume_cm3_per_100g_initial_as_built_binder"].values())),
        },
        "aqueous_h2o_g_per_100g": {
            "sequential": float(sequential[
                "aqueous_h2o_g_per_100g_initial_as_built_binder"]),
            "single": float(single[
                "aqueous_h2o_g_per_100g_initial_as_built_binder"]),
        },
        "bulk_cshq_ca_si": {
            "sequential": sequential.get("CSHQ", {}).get("ca_si"),
            "single": single.get("CSHQ", {}).get("ca_si"),
        },
        "pH": {"sequential": sequential["pH"], "single": single["pH"]},
        "ionic_strength_molal": {
            "sequential": sequential["ionic_strength_molal"],
            "single": single["ionic_strength_molal"],
        },
    }


def _run_schedule(label: str, cfg: TinnConfig, reference: dict,
                  worker: GemsWorker, bundle_hash: dict) -> tuple[dict, dict]:
    registry = registry_for(cfg)
    inventory = reference["initial_as_built_inventory"]
    initial_phase_mol = {
        phase: float(inventory["phase_mol_per_rve"][phase])
        for phase in KINETIC_PHASE_IDS
    }
    binder_mass_g = float(inventory["binder_mass_g_per_rve"])
    water_mol = float(inventory["water_mol_per_rve"])
    kinetics = make_kinetics(cfg)
    boundaries = _schedule_boundaries(cfg, max(AGES_H))
    present = set(worker.info()["phase_names"])
    suppressed = tuple(
        phase for phase in SUPPRESSED_CLINKER_PHASES if phase in present
    )

    state_elements: np.ndarray | None = None
    previous_alpha = kinetics.alpha_at(0.0)
    expected_elements = element_vector(registry.get("H2O").formula, water_mol)
    recorded: list[dict] = []
    max_closure = 0.0
    max_floor = 0.0
    max_path_drift = 0.0
    previous_time = 0.0
    for step_index, time_h in enumerate(boundaries):
        alpha = kinetics.alpha_at(time_h)
        released: dict[str, float] = {}
        for i, phase in enumerate(KINETIC_PHASE_IDS):
            if phase in SALT_PHASE_IDS:
                amount = initial_phase_mol[phase] if step_index == 0 else 0.0
            else:
                amount = initial_phase_mol[phase] * max(
                    0.0, float(alpha[i] - previous_alpha[i]))
            if amount > 0.0:
                released[phase] = amount
                expected_elements += element_vector(registry.get(phase).formula, amount)

        call_elements = expected_elements.copy() if state_elements is None else state_elements.copy()
        if state_elements is not None:
            for phase, amount in released.items():
                call_elements += element_vector(registry.get(phase).formula, amount)
        payload, scale = _scale_elements(call_elements, add_o2_seed=step_index == 0)
        result = worker.equilibrate_elements(
            payload, cfg.temperature_K, suppressed_phases=suppressed)
        state_elements = _output_elements(result, scale)
        effective_elements = call_elements.copy()
        element_index = {element: i for i, element in enumerate(ELEMENT_IDS)}
        if step_index == 0:
            effective_elements[element_index["O"]] += O2_SEED_MOL_O / scale
        for element, adjustment in result.element_input.get(
                "floor_adjustments_mol", {}).items():
            if element in element_index:
                effective_elements[element_index[element]] += float(adjustment) / scale
        for element, slack in result.element_input.get(
                "verification_slack_mol", {}).items():
            if element in element_index:
                effective_elements[element_index[element]] += float(slack) / scale
        call_norm = max(float(np.max(np.abs(effective_elements))), 1.0e-300)
        path_drift = float(
            np.max(np.abs(state_elements - effective_elements))) / call_norm
        max_path_drift = max(max_path_drift, path_drift)
        max_closure = max(max_closure, float(result.element_closure_max_rel()))
        max_floor = max(max_floor, float(result.floor_adjust_max_rel()))

        if any(abs(time_h - age) <= TIME_TOL for age in AGES_H):
            row = _snapshot_row(
                result, scale, time_h, alpha, binder_mass_g, registry)
            row["step_index"] = int(step_index + 1)
            row["time_start_h"] = float(previous_time)
            row["path_element_drift_rel_this_step"] = path_drift
            recorded.append(row)
        previous_alpha = alpha
        previous_time = time_h

    gates = {
        "final_time_672h": abs(boundaries[-1] - 672.0) <= TIME_TOL,
        "requested_ages_present": [row["time_h"] for row in recorded] == list(AGES_H),
        "direct_element_closure": max_closure <= DIRECT_CLOSURE_LIMIT,
        "direct_floor_adjustment": max_floor <= FLOOR_ADJUST_LIMIT,
        "path_element_drift": max_path_drift <= PATH_ELEMENT_DRIFT_LIMIT,
        "bundle_identity_available": bool(bundle_hash),
    }
    result = {
        "schema": "tinn.section_4_1.sequential_0d.v1",
        "evidence_level": "engineering_regression_only",
        "schedule": label,
        "normalization_basis": "100 g initial rasterized reactive-plus-salt binder",
        "step_count": len(boundaries),
        "boundaries_h": boundaries,
        "salt_policy": "full globally accessible salt inventory offered at first step",
        "rows": recorded,
    }
    qualification = {
        "schema": "tinn.section_4_1.sequential_0d_qualification.v1",
        "schedule": label,
        "gates": gates,
        "passed": _all_true(gates.values()),
        "max_direct_element_closure_rel": max_closure,
        "max_direct_floor_adjustment_rel": max_floor,
        "max_path_element_drift_rel": max_path_drift,
        "tolerances": {
            "direct_element_closure_rel": DIRECT_CLOSURE_LIMIT,
            "direct_floor_adjustment_rel": FLOOR_ADJUST_LIMIT,
            "path_element_drift_rel": PATH_ELEMENT_DRIFT_LIMIT,
        },
    }
    return result, qualification


def _write_report(out: Path, comparisons: dict, qualifications: dict) -> None:
    lines = [
        "# Sequential 0D schedule report",
        "",
        "Evidence level: `engineering_regression_only`.",
        "",
        "This calculation re-equilibrates the complete previous homogeneous 0D",
        "assemblage after every schedule increment. It contains no spatial",
        "allocation, morphology placement, cluster splitting, or unmet release.",
        "",
        "## Qualification",
        "",
        "| Schedule | Steps | Passed | Max path element drift |",
        "|---|---:|---:|---:|",
    ]
    for label, row in qualifications.items():
        lines.append(
            f"| {label} | {row['step_count']} | {row['passed']} | "
            f"{row['max_path_element_drift_rel']:.3e} |")
    lines += ["", "## Sequential-minus-single phase differences", ""]
    focus = ("CSHQ", "Portlandite", "ettringite", "Gypsum", "C3(AF)S0.84H")
    for label, schedule in comparisons["schedules"].items():
        lines += [f"### {label}", "", "| Age (h) | " + " | ".join(focus) + " |",
                  "|---:|" + "---:|" * len(focus)]
        for row in schedule["rows"]:
            phase = row["phase_mmol_per_100g_sequential_minus_single"]
            values = " | ".join(f"{float(phase.get(name, 0.0)):.6g}" for name in focus)
            lines.append(f"| {row['time_h']:.0f} | {values} |")
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "Near-identity with the single-age equilibrium means homogeneous closed-system",
        "equilibrium is path-independent at the tested numerical tolerance. Any remaining",
        "3D separation must then arise from spatial accessibility, allocation, morphology,",
        "unmet release, or their coupling. A difference here would instead demonstrate an",
        "intrinsic sequential chemistry/path effect and must be reported without forcing a",
        "spatial interpretation.",
        "",
    ]
    (out / "SEQUENTIAL_0D_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config-dt12", required=True)
    parser.add_argument("--config-dt05", required=True)
    parser.add_argument("--config-l2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--only",
        choices=("SEQ-DT12", "SEQ-DT05", "SEQ-L2"),
        help="Run one schedule only (intended for package smoke verification).",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference).resolve()
    config_paths = {
        "SEQ-DT12": Path(args.config_dt12).resolve(),
        "SEQ-DT05": Path(args.config_dt05).resolve(),
        "SEQ-L2": Path(args.config_l2).resolve(),
    }
    if args.only is not None:
        config_paths = {args.only: config_paths[args.only]}
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite sequential 0D evidence: {out}")

    reference = _read(reference_path)
    single_rows = {float(row["time_h"]): row for row in reference["rows"]}
    configs = {label: TinnConfig.from_json_file(str(path))
               for label, path in config_paths.items()}
    bundle_paths = {
        (REPO / str(cfg.chemistry.gems_bundle_lst)).resolve()
        for cfg in configs.values()
    }
    if len(bundle_paths) != 1:
        raise RuntimeError(f"sequential schedules use different bundles: {bundle_paths}")
    bundle = next(iter(bundle_paths))
    bundle_before = audit_bundle(str(bundle))
    worker_python = os.environ.get("TINN_GEMS_PYTHON") or next(
        cfg.chemistry.gems_worker_python for cfg in configs.values())
    worker = GemsWorker(str(bundle), python_executable=worker_python)
    results: dict[str, dict] = {}
    quals: dict[str, dict] = {}
    try:
        for label, cfg in configs.items():
            result, qualification = _run_schedule(
                label, cfg, reference, worker, bundle_before)
            results[label] = result
            qualification["step_count"] = result["step_count"]
            quals[label] = qualification
    finally:
        worker.close()
    bundle_after = audit_bundle(str(bundle))

    comparison = {
        "schema": "tinn.section_4_1.sequential_0d_comparison.v1",
        "single_0d_reference": str(reference_path),
        "schedules": {},
    }
    for label, result in results.items():
        comparison["schedules"][label] = {
            "step_count": result["step_count"],
            "rows": [
                _metric_delta(row, single_rows[float(row["time_h"])])
                for row in result["rows"]
            ],
        }

    common_gates = {
        "all_schedule_qualifications_pass": _all_true(
            row["passed"] for row in quals.values()),
        "bundle_sha256_unchanged": bundle_before == bundle_after,
        "reference_ages_present": all(age in single_rows for age in AGES_H),
    }
    qualification = {
        "schema": "tinn.section_4_1.sequential_0d_combined_qualification.v1",
        "gates": common_gates,
        "schedules": quals,
        "passed": _all_true(common_gates.values()),
    }
    provenance = {
        "schema": "tinn.section_4_1.sequential_0d_provenance.v1",
        "reference": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "configs": {
            label: {"path": str(path), "sha256": _sha256(path),
                    "config_hash": configs[label].config_hash()}
            for label, path in config_paths.items()
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "bundle_sha256_before": bundle_before,
        "bundle_sha256_after": bundle_after,
        "bundle_sha256_unchanged": bundle_before == bundle_after,
        "worker_python": worker_python,
    }

    out.mkdir(parents=True)
    filenames = {
        "SEQ-DT12": "sequential_0d_dt12.json",
        "SEQ-DT05": "sequential_0d_dt05.json",
        "SEQ-L2": "sequential_0d_l2.json",
    }
    for label, result in results.items():
        (out / filenames[label]).write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    (out / "sequential_0d_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")
    (out / "qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    _write_report(out, comparison, {
        label: {**row, "step_count": results[label]["step_count"]}
        for label, row in quals.items()
    })
    print(json.dumps({
        "out": str(out),
        "steps": {label: result["step_count"] for label, result in results.items()},
        "qualification_passed": qualification["passed"],
    }, indent=2))
    return 0 if qualification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
