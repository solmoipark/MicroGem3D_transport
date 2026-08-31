"""Run a traceable Section 4.1 numerical-qualification case.

This is orchestration, not a second simulation implementation.  All numerical
work is delegated to :class:`tinn.engine.Engine`; this script only captures the
per-trial audit stream, immutable-input hashes, and checkpoint readback proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.analysis import qualification_summary  # noqa: E402
from tinn.config import TinnConfig  # noqa: E402
from tinn.engine import Engine  # noqa: E402
from tinn.gems import GemsWorker, audit_bundle, run_0d_probe  # noqa: E402
from tinn.registry import ELEMENT_IDS, registry_for  # noqa: E402
from tinn.state import code_version  # noqa: E402
from tinn.storage import load_checkpoint  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_hashes() -> Dict[str, str]:
    files = sorted((REPO / "src" / "tinn").glob("*.py")) + [Path(__file__)]
    return {p.relative_to(REPO).as_posix(): _sha256(p) for p in files}


def _git_state() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True,
            text=True)
        return result.stdout.strip()

    try:
        status = git("status", "--porcelain")
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=REPO, check=True,
            capture_output=True).stdout
        return {
            "commit": git("rev-parse", "HEAD"),
            "code_version": code_version(),
            "working_tree_dirty": bool(status),
            "status_porcelain": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "source_snapshot_without_git": False,
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Portable job bundles intentionally omit .git. Per-file source hashes
        # remain the cross-machine identity authority in provenance.json.
        return {
            "commit": "unavailable_portable_snapshot",
            "code_version": code_version(),
            "working_tree_dirty": None,
            "status_porcelain": [],
            "tracked_diff_sha256": None,
            "source_snapshot_without_git": True,
        }


class JsonlAudit:
    """Append-and-flush observer so an interrupted long run keeps its evidence."""

    def __init__(self, path: Path, *, write_attempts: int = 8,
                 retry_delay_s: float = 0.25,
                 sleep: Callable[[float], None] = time.sleep):
        if write_attempts < 1:
            raise ValueError("write_attempts must be >= 1")
        if retry_delay_s < 0.0:
            raise ValueError("retry_delay_s must be >= 0")
        self.path = path
        self.events: List[dict] = []
        self.write_attempts = int(write_attempts)
        self.retry_delay_s = float(retry_delay_s)
        self.sleep = sleep
        self.write_retries = 0

    def _append_line(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    def __call__(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True) + "\n"
        for attempt in range(self.write_attempts):
            try:
                self._append_line(line)
                self.events.append(event)
                return
            except OSError:
                if attempt + 1 >= self.write_attempts:
                    raise
                self.write_retries += 1
                # Transient Windows file locks are normally short. Cap the
                # exponential backoff so a long chemistry run waits rather
                # than dying, without hiding a persistent permission error.
                delay = min(self.retry_delay_s * (2 ** attempt), 2.0)
                self.sleep(delay)


def run_case(config_path: Path, out_dir: Path, label: str,
             restart_path: Path | None = None,
             probe_0d: bool = False,
             recovery_checkpoint_every_h: float | None = None,
             audit_write_attempts: int = 8,
             audit_retry_delay_s: float = 0.25) -> dict:
    config_path = config_path.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"qualification output already exists (refusing overwrite): {out_dir}")
    out_dir.mkdir(parents=True)

    cfg = TinnConfig.from_json_file(str(config_path))
    if cfg.chemistry.backend != "gems3k":
        raise ValueError("Section 4.1 qualification requires the real gems3k backend")
    if cfg.rve.grid_size > 128:
        raise ValueError(
            "Section 4.1 coupled chemistry is limited to grid_size <= 128; "
            "the 320^3 @ 0.1 um case is geometry/topology-only because its "
            "dense hydrate channels and transactional copies exceed the "
            "supported memory envelope"
        )
    bundle = (REPO / str(cfg.chemistry.gems_bundle_lst)).resolve()
    bundle_before = audit_bundle(str(bundle))
    start_state = None
    restart_identity = None
    if restart_path is not None:
        restart_path = restart_path.resolve()
        start_state = load_checkpoint(str(restart_path), registry_for(cfg))
        if start_state.config_hash != cfg.config_hash():
            raise ValueError(
                "restart checkpoint config does not match qualification config")
        restart_identity = {
            "path": str(restart_path),
            "time_h": float(start_state.time_h),
            "dense_hash": start_state.dense_hash(),
            "full_hash": start_state.full_hash(),
        }

    provenance = {
        "schema": "tinn.section_4_1.provenance.v1",
        "case_label": label,
        "evidence_level": "engineering_regression_only",
        "claim_scope": (
            "numerical qualification and conservation of the frozen "
            "Parrot-Killoh + PC/xGEMS engineering baseline; not independent "
            "scientific validation"
        ),
        "config_path": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "config_hash": cfg.config_hash(),
        "normalized_config": cfg.model_dump(mode="json"),
        "fixed_condition_contract": {
            "curing": "sealed",
            "initial_saturation": "saturated",
            "mixing_water": "pure_water",
            "initial_dissolved_ions_mol": {
                "Ca": 0.0, "Si": 0.0, "Al": 0.0, "Fe": 0.0,
                "S": 0.0, "Na": 0.0, "K": 0.0,
            },
            "water_cement_ratio": cfg.w_c,
            "temperature_K": cfg.temperature_K,
            "pressure_Pa": 100000.0,
            "boundary_element_exchange": "zero",
            "solver_redox_seed": (
                "The xGEMS adapter may add its declared O2 convergence seed; "
                "it is not an initial pore-solution ion and is recorded in "
                "injected_elements."
            ),
        },
        "bundle_list": str(bundle),
        "bundle_sha256_before": bundle_before,
        "source_sha256": _source_hashes(),
        "git": _git_state(),
        "runtime": {
            "orchestrator_python": sys.executable,
            "orchestrator_python_version": platform.python_version(),
            "gems_worker_python": cfg.chemistry.gems_worker_python,
            "audit_write_attempts": int(audit_write_attempts),
            "audit_retry_delay_s": float(audit_retry_delay_s),
            "recovery_checkpoint_every_h": recovery_checkpoint_every_h,
        },
        "restart_input": restart_identity,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    audit = JsonlAudit(
        out_dir / "step_audit.jsonl",
        write_attempts=audit_write_attempts,
        retry_delay_s=audit_retry_delay_s)
    engine = Engine(cfg)
    state, summary = engine.run(
        state=start_state, out_dir=str(out_dir), audit_hook=audit,
        recovery_checkpoint_every_h=recovery_checkpoint_every_h)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    readback = []
    written = {
        float(e["time_h"]): e for e in audit.events
        if e.get("event") == "checkpoint_written"
    }
    reg = registry_for(cfg)
    for path in sorted(out_dir.glob("ckpt_*")):
        loaded = load_checkpoint(str(path), reg)
        expected = written[float(loaded.time_h)]
        readback.append({
            "path": str(path),
            "time_h": float(loaded.time_h),
            "dense_hash": loaded.dense_hash(),
            "full_hash": loaded.full_hash(),
            "matches_written_dense_hash": (
                loaded.dense_hash() == expected["dense_hash"]),
            "matches_written_full_hash": (
                loaded.full_hash() == expected["full_hash"]),
        })

    recovery_readback = []
    recovery_written = {
        Path(e["path"]).resolve(): e for e in audit.events
        if e.get("event") == "recovery_checkpoint_written"
    }
    for path in sorted(out_dir.glob("recovery_*")):
        loaded = load_checkpoint(str(path), reg)
        expected = recovery_written[path.resolve()]
        recovery_readback.append({
            "path": str(path),
            "time_h": float(loaded.time_h),
            "dense_hash": loaded.dense_hash(),
            "full_hash": loaded.full_hash(),
            "matches_written_dense_hash": (
                loaded.dense_hash() == expected["dense_hash"]),
            "matches_written_full_hash": (
                loaded.full_hash() == expected["full_hash"]),
        })

    bundle_after = audit_bundle(str(bundle))
    qualification = qualification_summary(audit.events)
    qualification.update({
        "schema": "tinn.section_4_1.qualification.v1",
        "case_label": label,
        "final_time_h": float(state.time_h),
        "final_dense_hash": state.dense_hash(),
        "final_full_hash": state.full_hash(),
        "checkpoint_readback": readback,
        "all_checkpoint_readbacks_match": all(
            r["matches_written_dense_hash"] and r["matches_written_full_hash"]
            for r in readback),
        "recovery_checkpoint_readback": recovery_readback,
        "all_recovery_checkpoint_readbacks_match": all(
            r["matches_written_dense_hash"] and r["matches_written_full_hash"]
            for r in recovery_readback),
        "audit_write_retries": int(audit.write_retries),
        "bundle_sha256_unchanged": bundle_before == bundle_after,
        "bundle_sha256_after": bundle_after,
        "boundary_exchange_mol_by_element": {
            el: float(state.boundary_exchanged_elements[i])
            for i, el in enumerate(ELEMENT_IDS)
        },
    })
    (out_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8")
    if probe_0d:
        worker = GemsWorker(
            cfg.chemistry.gems_bundle_lst,
            python_executable=cfg.chemistry.gems_worker_python)
        rows_0d = run_0d_probe(cfg, worker)
        reference_0d = {
            "schema": "tinn.section_4_1.zero_d_reference.v1",
            "evidence_level": "engineering_regression_only",
            "claim_scope": (
                "nominal-composition, spatially homogeneous PC/xGEMS "
                "engineering reference; not experimental validation"
            ),
            "rows": rows_0d,
        }
        (out_dir / "reference_0d.json").write_text(
            json.dumps(reference_0d, indent=2), encoding="utf-8")
    provenance["bundle_sha256_after"] = bundle_after
    provenance["bundle_sha256_unchanged"] = bundle_before == bundle_after
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    return qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--restart", default=None,
        help="checkpoint directory to continue (same config required)")
    parser.add_argument(
        "--probe-0d", action="store_true",
        help="also write the nominal homogeneous PC/xGEMS reference")
    parser.add_argument(
        "--recovery-checkpoint-every-h", type=float, default=None,
        help=("write non-output recovery checkpoints after accepted steps at "
              "this simulated-hour interval; does not change the timestep or "
              "the config hash"))
    parser.add_argument(
        "--audit-write-attempts", type=int, default=8,
        help="attempts for transient audit JSONL write failures")
    parser.add_argument(
        "--audit-retry-delay-s", type=float, default=0.25,
        help="initial exponential-backoff delay for audit writes")
    args = parser.parse_args()
    result = run_case(
        Path(args.config), Path(args.out), args.label,
        restart_path=(Path(args.restart) if args.restart else None),
        probe_0d=args.probe_0d,
        recovery_checkpoint_every_h=args.recovery_checkpoint_every_h,
        audit_write_attempts=args.audit_write_attempts,
        audit_retry_delay_s=args.audit_retry_delay_s)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
