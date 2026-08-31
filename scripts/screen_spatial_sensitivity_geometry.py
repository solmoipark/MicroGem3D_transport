"""Geometry-only screen for the Section 4.1 spatial-sensitivity matrix."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig  # noqa: E402
from tinn.geometry import initialize_rve  # noqa: E402
from tinn.registry import registry_for  # noqa: E402

REFERENCE = (
    REPO
    / "examples"
    / "qualification"
    / "spatial_sensitivity"
    / "domain32_v1_seed20260731.json"
)
CONFIG_ROOT = (
    REPO / "examples" / "qualification" / "spatial_sensitivity"
)


def _interface_area_um2(solid: np.ndarray, voxel_size_um: float) -> float:
    """Periodic six-face interface counted once over the positive axes."""
    area_faces = 0.0
    for axis in range(3):
        area_faces += float(
            np.abs(solid - np.roll(solid, -1, axis=axis)).sum()
        )
    return area_faces * voxel_size_um**2


def _case(config_path: Path) -> dict:
    cfg = TinnConfig.from_json_file(str(config_path))
    init = initialize_rve(cfg, registry_for(cfg))
    n_vox = cfg.rve.grid_size**3
    solid = np.clip(init.anhydrous_fraction.sum(axis=0), 0.0, 1.0)
    phase_volume_vox = {
        phase: float(init.anhydrous_fraction[i].sum())
        for i, phase in enumerate(init.phase_ids)
    }
    report = dict(init.report)
    checks = {
        "solid_fraction_rel_error": (
            float(report["solid_fraction_rel_error"]) <= 5.0e-3
        ),
        "w_c_rel_error": float(report["w_c_rel_error"]) <= 5.0e-3,
        "unplaced_volume_vox": (
            abs(float(report["unplaced_volume_vox"])) <= 1.0
        ),
        "voxel_identity": bool(
            np.all(solid >= -1.0e-12) and np.all(solid <= 1.0 + 1.0e-12)
        ),
    }
    result = {
        "config_path": str(config_path.resolve()),
        "config_hash": cfg.config_hash(),
        "grid_size": cfg.rve.grid_size,
        "voxel_size_um": cfg.rve.voxel_size_um,
        "domain_size_um": cfg.rve.grid_size * cfg.rve.voxel_size_um,
        "seed": cfg.rve.seed,
        "n_vox": n_vox,
        "dense_hash": init.dense_hash(),
        "geometry_report": report,
        "phase_volume_vox": phase_volume_vox,
        "phase_volume_fraction": {
            phase: volume / n_vox
            for phase, volume in phase_volume_vox.items()
        },
        "initial_anhydrous_interface_area_um2": _interface_area_um2(
            solid, cfg.rve.voxel_size_um
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    del init, solid
    gc.collect()
    return result


def main() -> int:
    configs = sorted(CONFIG_ROOT.glob("*.json"))
    cases = [_case(path) for path in configs]
    payload = {
        "schema": "tinn.section_4_1.spatial_geometry_screen.v1",
        "evidence_level": "engineering_regression_only",
        "all_passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    out_dir = REPO / "runs" / "section_4_1" / "spatial_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "geometry_screen.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Section 4.1 spatial geometry screen",
        "",
        "Evidence level: `engineering_regression_only`.",
        "",
        f"Overall gate: `{'PASS' if payload['all_passed'] else 'FAIL'}`",
        "",
        "| Config | Domain (µm) | Grid | Voxel (µm) | Seed | "
        "Solid fraction | w/c | Interface area (µm²) | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        report = case["geometry_report"]
        lines.append(
            f"| {Path(case['config_path']).stem} | "
            f"{case['domain_size_um']:g} | {case['grid_size']}³ | "
            f"{case['voxel_size_um']:g} | {case['seed']} | "
            f"{report['solid_fraction_achieved']:.6f} | "
            f"{report['w_c_achieved']:.6f} | "
            f"{case['initial_anhydrous_interface_area_um2']:.3f} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "This screen checks only initial realization and does not invoke "
            "GEMS. Chemistry runs retain their own conservation, checkpoint, "
            "and immutable-bundle gates.",
        ]
    )
    out_md = out_dir / "geometry_screen.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(out_json),
                "markdown": str(out_md),
                "all_passed": payload["all_passed"],
                "case_count": len(cases),
            }
        )
    )
    return 0 if payload["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
