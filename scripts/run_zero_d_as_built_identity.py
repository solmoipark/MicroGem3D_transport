"""Q32 as-built standalone-GEMS reference and one-domain adapter identity.

The source checkpoint supplies the actual rasterized phase and water inventory.
At each requested age, cumulative kinetic release is equilibrated independently.
The same canonical element input is sent through standalone ``GemsWorker`` and
through ``GemsBackend.react``.  Results are reported on the physical RVE basis
and normalized to 100 g of initial as-built binder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from run_section_4_1 import _git_state, _sha256, _source_hashes  # noqa: E402
from tinn.config import TinnConfig  # noqa: E402
from tinn.gems import (  # noqa: E402
    AQUEOUS_PHASE,
    CANONICAL_MAX_ELEMENT_MOL,
    GAS_PHASE,
    O2_SEED_MOL_O,
    SUPPRESSED_CLINKER_PHASES,
    GemsBackend,
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
from tinn.storage import load_checkpoint  # noqa: E402


TIMES_H = (24.0, 168.0, 672.0)
IDENTITY_RTOL = 2.0e-12
IDENTITY_ATOL_MOL = 1.0e-24
PH_ATOL = 1.0e-12
IONIC_STRENGTH_ATOL = 1.0e-12
DIRECT_CLOSURE_LIMIT = 1.0e-10
FLOOR_ADJUST_LIMIT = 1.0e-9


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_checkpoint(run: Path) -> Path:
    rows = []
    for checkpoint in run.glob("ckpt_*"):
        header = _read(checkpoint / "header.json")
        rows.append((float(header["time_h"]), checkpoint))
    if not rows:
        raise FileNotFoundError(f"no checkpoint found under {run}")
    return min(rows)[1]


def _scaled_input(
    released_mol: Mapping[str, float],
    water_mol: float,
    registry,
) -> tuple[dict[str, float], float, np.ndarray]:
    elements = np.zeros(len(ELEMENT_IDS), dtype=np.float64)
    for phase, mol in released_mol.items():
        if mol > 0.0:
            elements += element_vector(registry.get(phase).formula, mol)
    elements += element_vector(registry.get("H2O").formula, water_mol)
    peak = float(elements.max())
    if peak <= 0.0:
        raise ValueError("empty 0D element inventory")
    scale = CANONICAL_MAX_ELEMENT_MOL / peak
    scaled = {
        element: float(elements[i] * scale)
        for i, element in enumerate(ELEMENT_IDS)
        if elements[i] > 0.0
    }
    scaled["O"] = scaled.get("O", 0.0) + O2_SEED_MOL_O
    return scaled, scale, elements


def _rescale_map(values: Mapping[str, float], scale: float) -> dict[str, float]:
    return {str(key): float(value) / scale for key, value in values.items()}


def _rescale_nested(
    values: Mapping[str, Mapping[str, float]], scale: float
) -> dict[str, dict[str, float]]:
    return {
        str(outer): _rescale_map(inner, scale)
        for outer, inner in values.items()
    }


def _normalize_map(
    values: Mapping[str, float], binder_mass_g: float, factor: float
) -> dict[str, float]:
    return {
        key: float(value) * factor * 100.0 / binder_mass_g
        for key, value in values.items()
    }


def _normalize_nested(
    values: Mapping[str, Mapping[str, float]],
    binder_mass_g: float,
    factor: float,
) -> dict[str, dict[str, float]]:
    return {
        outer: _normalize_map(inner, binder_mass_g, factor)
        for outer, inner in values.items()
    }


def _max_abs_rel(
    a: Mapping[str, float], b: Mapping[str, float]
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    label = None
    for key in sorted(set(a) | set(b)):
        av, bv = float(a.get(key, 0.0)), float(b.get(key, 0.0))
        delta = abs(av - bv)
        rel = delta / max(abs(av), abs(bv), IDENTITY_ATOL_MOL)
        if delta > max_abs:
            max_abs, label = delta, key
        max_rel = max(max_rel, rel)
    return max_abs, max_rel, label


def _flatten(values: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        f"{outer}/{inner}": float(value)
        for outer, row in values.items()
        for inner, value in row.items()
    }


def _adapter_maps(result) -> tuple[dict, dict, dict]:
    phase_amounts: dict[str, float] = {}
    phase_elements: dict[str, dict[str, float]] = {}
    phase_species: dict[str, dict[str, float]] = {}
    for parcel in result.parcels:
        phase_amounts[parcel.phase_id] = (
            phase_amounts.get(parcel.phase_id, 0.0) + float(parcel.mol)
        )
        elements = phase_elements.setdefault(parcel.phase_id, {})
        for i, element in enumerate(ELEMENT_IDS):
            value = float(parcel.elements[i])
            if value != 0.0:
                elements[element] = elements.get(element, 0.0) + value
        species = phase_species.setdefault(parcel.phase_id, {})
        for name, value in (parcel.endmember_mol or {}).items():
            species[name] = species.get(name, 0.0) + float(value)
    return phase_amounts, phase_elements, phase_species


def _solid_direct_maps(direct, scale: float) -> tuple[dict, dict, dict]:
    excluded = {AQUEOUS_PHASE, GAS_PHASE, *SUPPRESSED_CLINKER_PHASES}
    amounts = {
        phase: float(mol) / scale
        for phase, mol in direct.phase_amounts_mol.items()
        if phase not in excluded and float(mol) > 0.0
    }
    elements = {
        phase: _rescale_map(direct.phase_elements_mol.get(phase, {}), scale)
        for phase in amounts
    }
    species = {
        phase: _rescale_map(direct.phase_species_mol.get(phase, {}), scale)
        for phase in amounts
    }
    return amounts, elements, species


def _cshq_row(
    phase_elements: Mapping[str, Mapping[str, float]],
    phase_species: Mapping[str, Mapping[str, float]],
) -> dict:
    elements = phase_elements.get("CSHQ", {})
    ca, si = float(elements.get("Ca", 0.0)), float(elements.get("Si", 0.0))
    species = phase_species.get("CSHQ", {})
    total = float(sum(species.values()))
    return {
        "ca_si": ca / si if si > 0.0 else None,
        "endmember_mole_fraction": {
            name: float(value) / total for name, value in species.items()
        } if total > 0.0 else {},
    }


def _all_true(values: Iterable[bool]) -> bool:
    return all(bool(value) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--times", nargs="+", type=float, default=list(TIMES_H))
    args = parser.parse_args()

    anchor_run = Path(args.anchor_run).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite 0D evidence: {out}")

    checkpoint = _first_checkpoint(anchor_run)
    header = _read(checkpoint / "header.json")
    cfg = TinnConfig.model_validate(header["config"])
    registry = registry_for(cfg)
    state = load_checkpoint(str(checkpoint), registry)
    if state.config_hash != header["config_hash"]:
        raise RuntimeError("checkpoint state/config identity mismatch")

    initial_phase_mol = {
        phase: float(state.initial_phase_mol[i])
        for i, phase in enumerate(KINETIC_PHASE_IDS)
    }
    binder_mass_g = sum(
        mol * registry.get(phase).molar_mass_g_mol
        for phase, mol in initial_phase_mol.items()
    )
    if binder_mass_g <= 0.0:
        raise RuntimeError("as-built binder mass is zero")

    bundle = (REPO / str(cfg.chemistry.gems_bundle_lst)).resolve()
    bundle_before = audit_bundle(str(bundle))
    worker = GemsWorker(
        str(bundle), python_executable=cfg.chemistry.gems_worker_python
    )
    backend = GemsBackend(worker, cfg.temperature_K, registry=registry)
    info = worker.info()
    present = set(info["phase_names"])
    suppressed = tuple(
        phase for phase in SUPPRESSED_CLINKER_PHASES if phase in present
    )
    kinetics = make_kinetics(cfg)

    reference_rows = []
    identity_rows = []
    try:
        for time_h in sorted(set(float(value) for value in args.times)):
            alpha = kinetics.alpha_at(time_h)
            released_mol = {}
            for i, phase in enumerate(KINETIC_PHASE_IDS):
                initial = initial_phase_mol[phase]
                released = initial if phase in SALT_PHASE_IDS else initial * float(alpha[i])
                if released > 0.0:
                    released_mol[phase] = released

            scaled_input, scale, physical_elements = _scaled_input(
                released_mol, state.initial_water_mol, registry
            )
            direct = worker.equilibrate_elements(
                scaled_input,
                cfg.temperature_K,
                suppressed_phases=suppressed,
            )
            adapter = backend.react(
                released_mol,
                float(state.initial_water_mol),
                np.zeros(len(ELEMENT_IDS)),
            )

            direct_amounts, direct_elements, direct_species = _solid_direct_maps(
                direct, scale
            )
            adapter_amounts, adapter_elements, adapter_species = _adapter_maps(adapter)

            amount_abs, amount_rel, amount_label = _max_abs_rel(
                direct_amounts, adapter_amounts
            )
            element_abs, element_rel, element_label = _max_abs_rel(
                _flatten(direct_elements), _flatten(adapter_elements)
            )
            species_abs, species_rel, species_label = _max_abs_rel(
                _flatten(direct_species), _flatten(adapter_species)
            )

            direct_aq = _rescale_map(
                direct.phase_elements_mol.get(AQUEOUS_PHASE, {}), scale
            )
            adapter_aq = {
                element: float(adapter.aqueous_elements[i])
                for i, element in enumerate(ELEMENT_IDS)
                if float(adapter.aqueous_elements[i]) != 0.0
            }
            aq_abs, aq_rel, aq_label = _max_abs_rel(direct_aq, adapter_aq)
            direct_h2o = float(direct.aqueous_h2o_mol) / scale
            h2o_abs = abs(direct_h2o - float(adapter.aqueous_h2o_mol))
            h2o_rel = h2o_abs / max(abs(direct_h2o), IDENTITY_ATOL_MOL)
            ph_abs = abs(float(direct.ph) - float(adapter.ph))
            ionic_abs = abs(
                float(direct.ionic_strength) - float(adapter.ionic_strength)
            )

            direct_masses_kg = {
                phase: float(direct.phase_masses_kg.get(phase, 0.0)) / scale
                for phase in direct_amounts
            }
            direct_volumes_m3 = {
                phase: float(direct.phase_volumes_m3.get(phase, 0.0)) / scale
                for phase in direct_amounts
            }
            direct_phase_elements = {
                phase: direct_elements[phase] for phase in direct_amounts
            }
            direct_phase_species = {
                phase: direct_species[phase] for phase in direct_amounts
            }
            reference_rows.append({
                "time_h": time_h,
                "kinetic_alpha": {
                    phase: float(alpha[i])
                    for i, phase in enumerate(KINETIC_PHASE_IDS)
                },
                "released_phase_mmol_per_100g_initial_as_built_binder":
                    _normalize_map(released_mol, binder_mass_g, 1000.0),
                "phase_amount_mmol_per_100g_initial_as_built_binder":
                    _normalize_map(direct_amounts, binder_mass_g, 1000.0),
                "phase_mass_g_per_100g_initial_as_built_binder":
                    _normalize_map(direct_masses_kg, binder_mass_g, 1000.0),
                "phase_volume_cm3_per_100g_initial_as_built_binder":
                    _normalize_map(direct_volumes_m3, binder_mass_g, 1.0e6),
                "phase_element_mmol_per_100g_initial_as_built_binder":
                    _normalize_nested(
                        direct_phase_elements, binder_mass_g, 1000.0
                    ),
                "phase_endmember_mmol_per_100g_initial_as_built_binder":
                    _normalize_nested(
                        direct_phase_species, binder_mass_g, 1000.0
                    ),
                "aqueous_element_mmol_per_100g_initial_as_built_binder":
                    _normalize_map(direct_aq, binder_mass_g, 1000.0),
                "aqueous_h2o_mol_per_100g_initial_as_built_binder":
                    direct_h2o * 100.0 / binder_mass_g,
                "aqueous_h2o_g_per_100g_initial_as_built_binder":
                    direct_h2o * registry.get("H2O").molar_mass_g_mol
                    * 100.0 / binder_mass_g,
                "pH": float(direct.ph),
                "ionic_strength_molal": float(direct.ionic_strength),
                "CSHQ": _cshq_row(direct_elements, direct_species),
                "element_closure_max_rel": direct.element_closure_max_rel(),
                "floor_adjust_max_rel": direct.floor_adjust_max_rel(),
                "xgems_version": direct.xgems_version,
            })

            phase_sets_match = set(direct_amounts) == set(adapter_amounts)
            gates = {
                "phase_assemblage_match": phase_sets_match,
                "phase_amount_match": amount_abs <= IDENTITY_ATOL_MOL
                    or amount_rel <= IDENTITY_RTOL,
                "phase_element_match": element_abs <= IDENTITY_ATOL_MOL
                    or element_rel <= IDENTITY_RTOL,
                "endmember_match": species_abs <= IDENTITY_ATOL_MOL
                    or species_rel <= IDENTITY_RTOL,
                "aqueous_element_match": aq_abs <= IDENTITY_ATOL_MOL
                    or aq_rel <= IDENTITY_RTOL,
                "aqueous_h2o_match": h2o_abs <= IDENTITY_ATOL_MOL
                    or h2o_rel <= IDENTITY_RTOL,
                "pH_match": ph_abs <= PH_ATOL,
                "ionic_strength_match": ionic_abs <= IONIC_STRENGTH_ATOL,
                "direct_element_closure": direct.element_closure_max_rel()
                    <= DIRECT_CLOSURE_LIMIT,
                "direct_floor_adjustment": direct.floor_adjust_max_rel()
                    <= FLOOR_ADJUST_LIMIT,
            }
            identity_rows.append({
                "time_h": time_h,
                "gates": gates,
                "passed": _all_true(gates.values()),
                "differences": {
                    "phase_amount_max_abs_mol_per_rve": amount_abs,
                    "phase_amount_max_rel": amount_rel,
                    "phase_amount_worst": amount_label,
                    "phase_element_max_abs_mol_per_rve": element_abs,
                    "phase_element_max_rel": element_rel,
                    "phase_element_worst": element_label,
                    "endmember_max_abs_mol_per_rve": species_abs,
                    "endmember_max_rel": species_rel,
                    "endmember_worst": species_label,
                    "aqueous_element_max_abs_mol_per_rve": aq_abs,
                    "aqueous_element_max_rel": aq_rel,
                    "aqueous_element_worst": aq_label,
                    "aqueous_h2o_abs_mol_per_rve": h2o_abs,
                    "aqueous_h2o_rel": h2o_rel,
                    "pH_abs": ph_abs,
                    "ionic_strength_abs": ionic_abs,
                },
                "standalone_phase_assemblage": sorted(direct_amounts),
                "adapter_phase_assemblage": sorted(adapter_amounts),
            })
    finally:
        worker.close()

    bundle_after = audit_bundle(str(bundle))
    qualification_gates = {
        "source_checkpoint_manifest_readback": True,
        "source_checkpoint_config_identity": state.config_hash == header["config_hash"],
        "requested_ages_present": [row["time_h"] for row in reference_rows]
            == sorted(set(float(value) for value in args.times)),
        "all_adapter_identity_rows_pass": _all_true(
            row["passed"] for row in identity_rows
        ),
        "bundle_sha256_unchanged": bundle_before == bundle_after,
    }
    qualification = {
        "schema": "tinn.section_4_1.zero_d_as_built_qualification.v1",
        "gates": qualification_gates,
        "passed": _all_true(qualification_gates.values()),
        "identity_tolerances": {
            "relative": IDENTITY_RTOL,
            "absolute_mol_per_rve": IDENTITY_ATOL_MOL,
            "pH_absolute": PH_ATOL,
            "ionic_strength_absolute": IONIC_STRENGTH_ATOL,
            "direct_element_closure_relative": DIRECT_CLOSURE_LIMIT,
            "direct_floor_adjustment_relative": FLOOR_ADJUST_LIMIT,
        },
    }

    result = {
        "schema": "tinn.section_4_1.zero_d_as_built_reference.v1",
        "evidence_level": "engineering_regression_only",
        "claim_scope": (
            "standalone PC/xGEMS bulk-equilibrium reference for the actual "
            "rasterized Q32 inventory; common chemistry baseline for subsequent "
            "3D timestep comparisons, not a spatial or experimental validation"
        ),
        "age_policy": "independent cumulative-release equilibrium at each age",
        "normalization_basis": "100 g initial rasterized reactive-plus-salt binder",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_time_h": float(state.time_h),
        "source_checkpoint_dense_hash": state.dense_hash(),
        "source_checkpoint_full_hash": state.full_hash(),
        "config_hash": cfg.config_hash(),
        "initial_as_built_inventory": {
            "binder_mass_g_per_rve": binder_mass_g,
            "phase_mol_per_rve": initial_phase_mol,
            "phase_mmol_per_100g_initial_as_built_binder":
                _normalize_map(initial_phase_mol, binder_mass_g, 1000.0),
            "water_mol_per_rve": float(state.initial_water_mol),
            "water_g_per_100g_initial_as_built_binder":
                float(state.initial_water_mol)
                * registry.get("H2O").molar_mass_g_mol
                * 100.0 / binder_mass_g,
            "input_element_mol_per_rve_at_full_initial_state": {
                element: float(state.initial_elements[i])
                for i, element in enumerate(ELEMENT_IDS)
            },
        },
        "rows": reference_rows,
    }
    identity = {
        "schema": "tinn.section_4_1.zero_d_adapter_identity.v1",
        "comparison": (
            "standalone GemsWorker versus GemsBackend.react using identical "
            "canonical element input and physical rescaling"
        ),
        "rows": identity_rows,
    }

    source_hashes = _source_hashes()
    source_hashes[Path(__file__).relative_to(REPO).as_posix()] = _sha256(
        Path(__file__)
    )
    anchor_provenance = _read(anchor_run / "provenance.json")
    config_path = Path(anchor_provenance["config_path"])
    provenance = {
        "schema": "tinn.section_4_1.zero_d_as_built_provenance.v1",
        "anchor_run": str(anchor_run),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_manifest_sha256": _sha256(checkpoint / "manifest.json"),
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path) if config_path.is_file() else None,
        "config_hash": cfg.config_hash(),
        "source_sha256": source_hashes,
        "git": _git_state(),
        "bundle_sha256_before": bundle_before,
        "bundle_sha256_after": bundle_after,
        "bundle_sha256_unchanged": bundle_before == bundle_after,
    }

    out.mkdir(parents=True)
    (out / "reference_0d_as_built.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (out / "adapter_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    (out / "qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8"
    )
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "out": str(out),
        "ages_h": [row["time_h"] for row in reference_rows],
        "adapter_identity_passed": qualification_gates[
            "all_adapter_identity_rows_pass"
        ],
        "bundle_sha256_unchanged": bundle_before == bundle_after,
        "qualification_passed": qualification["passed"],
    }, indent=2))
    return 0 if qualification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
