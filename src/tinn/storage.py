"""Dependency-free Zarr-v2 directory checkpoints: uncompressed raw chunks + JSON
tables + manifest checksums; written to a temp dir, verified, atomically renamed.

Restart contract (PRD §6.1): dense arrays, ledger scalars/vectors, and RNG state
round-trip bit-identically (floats survive JSON via repr round-tripping).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .config import TinnConfig
from .registry import Registry
from .state import SimulationState, _DENSE_FIELDS, code_version

FORMAT_VERSION = 1

_LEDGER_VECTORS = ("phase_mol", "initial_phase_mol", "unmet_mol", "hydrate_mol",
                   "initial_elements")
_LEDGER_SCALARS = ("time_h", "dt_h", "water_free_mol", "water_gel_mol",
                   "water_bound_mol", "initial_water_mol", "inert_volume_vox",
                   "accept_count", "config_hash", "backend_id")
_TABLE_SCHEMAS: Dict[str, Dict[str, str]] = {
    "particles": {"id": "<i8", "tier": "|i1", "diameter_um": "<f8",
                  "volume_vox": "<f8", "center_zyx": "<f8", "placed": "|b1"},
    "subgrid_bins": {"d_lo_um": "<f8", "d_hi_um": "<f8", "volume_vox": "<f8",
                     "number_est": "<f8"},
}


class StorageError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_zarr_array(dir_path: Path, arr: np.ndarray) -> None:
    dir_path.mkdir(parents=True)
    meta = {"zarr_format": 2, "shape": list(arr.shape), "chunks": list(arr.shape),
            "dtype": arr.dtype.str, "compressor": None, "fill_value": None,
            "order": "C", "filters": None}
    (dir_path / ".zarray").write_text(json.dumps(meta), encoding="utf-8")
    chunk_name = ".".join(["0"] * arr.ndim)
    (dir_path / chunk_name).write_bytes(np.ascontiguousarray(arr).tobytes())


def _read_zarr_array(dir_path: Path) -> np.ndarray:
    meta = json.loads((dir_path / ".zarray").read_text(encoding="utf-8"))
    chunk_name = ".".join(["0"] * len(meta["shape"]))
    raw = (dir_path / chunk_name).read_bytes()
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    return arr.copy()


def save_checkpoint(state: SimulationState, out_dir: str, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / name
    if final.exists():
        raise StorageError(f"checkpoint already exists: {final}")
    tmp = out / f".{name}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        (tmp / "arrays").mkdir(parents=True)
        (tmp / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
        for field in _DENSE_FIELDS:
            _write_zarr_array(tmp / "arrays" / field, getattr(state, field))
        _write_zarr_array(tmp / "arrays" / "cluster_inventory", state.cluster_inventory)

        header = {
            "format_version": FORMAT_VERSION,
            "code_version": code_version(),
            "config": state.config.model_dump(mode="json"),
            "rng_state": state.rng_state,
            "reject_counts": state.reject_counts,
        }
        for k in _LEDGER_SCALARS:
            header[k] = getattr(state, k)
        for k in _LEDGER_VECTORS:
            header[k] = np.asarray(getattr(state, k)).tolist()
        (tmp / "header.json").write_text(json.dumps(header), encoding="utf-8")

        tables = {
            "particles": {k: np.asarray(v).tolist() for k, v in state.particles.items()},
            "subgrid_bins": {k: np.asarray(v).tolist()
                             for k, v in state.subgrid_bins.items()},
            "parcels": state.parcels,
            "remap_events": state.remap_events,
        }
        (tmp / "tables.json").write_text(json.dumps(tables), encoding="utf-8")

        manifest = {}
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                manifest[p.relative_to(tmp).as_posix()] = _sha256_file(p)
        (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        _verify_dir(tmp)  # re-read and compare before publishing
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def _verify_dir(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for rel, digest in manifest.items():
        f = path / rel
        if not f.is_file():
            raise StorageError(f"checkpoint missing file {rel}")
        if _sha256_file(f) != digest:
            raise StorageError(f"checkpoint checksum mismatch: {rel}")


def load_checkpoint(path: str, registry: Registry) -> SimulationState:
    root = Path(path)
    if not (root / "manifest.json").is_file():
        raise StorageError(f"not a checkpoint directory: {root}")
    _verify_dir(root)

    header = json.loads((root / "header.json").read_text(encoding="utf-8"))
    if header["format_version"] != FORMAT_VERSION:
        raise StorageError(f"unsupported checkpoint format {header['format_version']}")
    config = TinnConfig.model_validate(header["config"])

    arrays = {f: _read_zarr_array(root / "arrays" / f) for f in _DENSE_FIELDS}
    cluster_inventory = _read_zarr_array(root / "arrays" / "cluster_inventory")

    tables = json.loads((root / "tables.json").read_text(encoding="utf-8"))

    def as_table(name: str) -> Dict[str, np.ndarray]:
        schema = _TABLE_SCHEMAS[name]
        return {k: np.asarray(v, dtype=np.dtype(schema.get(k, "<f8")))
                for k, v in tables[name].items()}

    return SimulationState(
        config=config,
        **arrays,
        particles=as_table("particles"),
        subgrid_bins=as_table("subgrid_bins"),
        parcels=tables["parcels"],
        remap_events=tables["remap_events"],
        cluster_inventory=cluster_inventory,
        time_h=header["time_h"],
        dt_h=header["dt_h"],
        phase_mol=np.asarray(header["phase_mol"]),
        initial_phase_mol=np.asarray(header["initial_phase_mol"]),
        unmet_mol=np.asarray(header["unmet_mol"]),
        hydrate_mol=np.asarray(header["hydrate_mol"]),
        water_free_mol=header["water_free_mol"],
        water_gel_mol=header["water_gel_mol"],
        water_bound_mol=header["water_bound_mol"],
        initial_water_mol=header["initial_water_mol"],
        inert_volume_vox=header["inert_volume_vox"],
        initial_elements=np.asarray(header["initial_elements"]),
        accept_count=header["accept_count"],
        reject_counts=dict(header["reject_counts"]),
        rng_state=header["rng_state"],
        config_hash=header["config_hash"],
        backend_id=header["backend_id"],
    )
