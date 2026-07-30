"""Generate the homogeneous PC/xGEMS engineering reference for Section 4.1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from run_section_4_1 import _git_state, _sha256, _source_hashes  # noqa: E402
from tinn.config import TinnConfig  # noqa: E402
from tinn.gems import GemsWorker, audit_bundle, run_0d_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite 0D evidence: {out}")
    out.mkdir(parents=True)

    cfg = TinnConfig.from_json_file(str(config_path))
    bundle = (REPO / str(cfg.chemistry.gems_bundle_lst)).resolve()
    before = audit_bundle(str(bundle))
    worker = GemsWorker(
        cfg.chemistry.gems_bundle_lst,
        python_executable=cfg.chemistry.gems_worker_python)
    rows = run_0d_probe(cfg, worker)
    after = audit_bundle(str(bundle))
    source_hashes = _source_hashes()
    source_hashes[Path(__file__).relative_to(REPO).as_posix()] = _sha256(
        Path(__file__))

    result = {
        "schema": "tinn.section_4_1.zero_d_reference.v2",
        "evidence_level": "engineering_regression_only",
        "claim_scope": (
            "nominal-composition, spatially homogeneous PC/xGEMS engineering "
            "reference with full solubility-controlled salt inventory; not "
            "experimental validation"
        ),
        "config_hash": cfg.config_hash(),
        "config_file_sha256": _sha256(config_path),
        "salt_carrier_policy": (
            "complete initial gypsum/hemihydrate/anhydrite/arcanite/thenardite "
            "inventory offered to equilibrium at every independent time point"
        ),
        "rows": rows,
    }
    provenance = {
        "schema": "tinn.section_4_1.zero_d_provenance.v1",
        "config_path": str(config_path),
        "source_sha256": source_hashes,
        "git": _git_state(),
        "bundle_sha256_before": before,
        "bundle_sha256_after": after,
        "bundle_sha256_unchanged": before == after,
    }
    (out / "reference_0d.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "bundle_sha256_unchanged": before == after,
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
