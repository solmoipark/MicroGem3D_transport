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
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, KINETIC_PHASE_IDS,
                       Registry, SOLID_PHASE_IDS)
from .state import SimulationState, _DENSE_FIELDS, code_version

# v3 (E1, PRD 2.3 rev.3): endmember ledger (endmember_ids/mol/elements) plus
# the reserved RT-W2 boundary_exchanged_elements vector — one format break for
# both, per the endmember plan's co-ride decision. v2 checkpoints are
# explicitly incompatible (no migration, PRD rule). E2's per-cluster pool
# array (cluster_endmember_mol) rides the SAME version: no external v3
# checkpoints existed when it landed, so no second break.
FORMAT_VERSION = 6  # RT-Cl: the element ledger gains the Cl column (E=12,
                    # appended last); every (.., E) array in a v5 checkpoint
                    # is one column short - refused, no migration.
                    # v5 was Tier 0 / RT-P0b (PRD 2.3/4.6.4): frozen aqueous
                    # speciation per domain (domain_species_mol +
                    # aq_species_ids) AND the reserved Tier-1 sorbed
                    # inventory (domain_sorbed_mol, zero rows until RT-S1a
                    # - the v4 boundary_water_mol precedent: one break, one
                    # anchor re-pin). v4 checkpoints predate species
                    # transport - rerun from the config (no migration).
                    # v4 was: domain-keyed rows + boundary_water_mol +
                    # economy snapshots.

_LEDGER_VECTORS = ("phase_mol", "initial_phase_mol", "unmet_mol", "hydrate_mol",
                   "hydrate_env_vol_vox", "injected_elements",
                   "initial_elements", "endmember_mol",
                   "boundary_exchanged_elements")
_LEDGER_SCALARS = ("time_h", "dt_h", "water_free_mol", "water_gel_mol",
                   "water_bound_mol", "initial_water_mol",
                   "boundary_water_mol", "inert_volume_vox",
                   "accept_count", "config_hash", "backend_id")
_TABLE_SCHEMAS: Dict[str, Dict[str, str]] = {
    "particles": {"id": "<i8", "tier": "|i1", "material": "|i1",
                  "diameter_um": "<f8", "volume_vox": "<f8",
                  "center_zyx": "<f8", "placed": "|b1"},
    "subgrid_bins": {"material": "|i1", "d_lo_um": "<f8", "d_hi_um": "<f8",
                     "volume_vox": "<f8", "number_est": "<f8"},
}


class StorageError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chemistry_identity(config: TinnConfig) -> Dict:
    """RT (external review 2026-09-07): content identity of the external
    thermodynamic inputs a restart must reproduce - the GEMS bundle files and
    the PHREEQC database by sha256, plus the chemistry-engine versions.
    Matching paths, species names, or endmember stoichiometry does not detect
    a changed equilibrium constant under the same name; a content hash does.
    Reuses gems.audit_bundle (the same sha map the worker guards with)."""
    ident: Dict = {}
    lst = getattr(config.chemistry, "gems_bundle_lst", None)
    if lst is not None:
        from .gems import audit_bundle
        ident["gems_bundle_sha256"] = audit_bundle(str(Path(lst).resolve()))
    if config.sorption is not None:
        dat = Path(config.sorption.phreeqc_dat).resolve()
        ident["phreeqc_dat_sha256"] = (_sha256_file(dat) if dat.is_file()
                                       else None)
    versions: Dict = {}
    for mod in ("xgems", "phreeqpython"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:
            versions[mod] = None      # not importable in this interpreter
    ident["engine_versions"] = versions
    return ident


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
        _write_zarr_array(tmp / "arrays" / "cluster_endmember_mol",
                          state.cluster_endmember_mol)
        _write_zarr_array(tmp / "arrays" / "domain_eq_inventory",
                          state.domain_eq_inventory)
        _write_zarr_array(tmp / "arrays" / "domain_eq_water",
                          state.domain_eq_water)
        _write_zarr_array(tmp / "arrays" / "domain_eq_age",
                          state.domain_eq_age)
        _write_zarr_array(tmp / "arrays" / "domain_species_mol",
                          state.domain_species_mol)
        _write_zarr_array(tmp / "arrays" / "domain_sorbed_mol",
                          state.domain_sorbed_mol)

        header = {
            "format_version": FORMAT_VERSION,
            "code_version": code_version(),
            "config": state.config.model_dump(mode="json"),
            "rng_state": state.rng_state,
            "reject_counts": state.reject_counts,
            # channel orderings the ledger vectors are positional over (PRD §2.3):
            # a restart under a build with different orderings must hard-fail.
            # hydrate channels are RUN-scoped (bundle-derived for gems3k).
            "kinetic_phase_ids": list(KINETIC_PHASE_IDS),
            "solid_phase_ids": list(SOLID_PHASE_IDS),
            "hydrate_phase_ids": list(state.hydrate_ids),
            "element_ids": list(ELEMENT_IDS),
            # E1: run-scoped endmember universe, positional over endmember_mol
            "endmember_ids": [list(pair) for pair in state.endmember_ids],
            # RT-P0b: run-scoped aqueous species order, positional over
            # domain_species_mol columns (empty when species transport off)
            "aq_species_ids": list(state.aq_species_ids),
            # RT reproducibility: content hashes of the external chemistry
            # inputs (checked on restart; a hard error on mismatch unless
            # TINN_ALLOW_CHEMISTRY_MISMATCH is set). Header-key addition only,
            # no format break; a pre-review checkpoint lacks this key -> warn.
            "chemistry_identity": chemistry_identity(state.config),
        }
        for k in _LEDGER_SCALARS:
            header[k] = getattr(state, k)
        for k in _LEDGER_VECTORS:
            header[k] = np.asarray(getattr(state, k)).tolist()
        header["hydrate_elements_ch"] = np.asarray(state.hydrate_elements_ch).tolist()
        header["endmember_elements"] = np.asarray(state.endmember_elements).tolist()
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
        raise StorageError(
            f"checkpoint format {header['format_version']} is not supported by "
            f"this build (current {FORMAT_VERSION}); v5 checkpoints predate the "
            f"chloride element column (RT-Cl, E=12), v4 checkpoints predate "
            f"species transport and the reserved sorbed-inventory array "
            f"(Tier 0/RT-P0b), v3 the equilibration domains and boundary "
            f"water ledger (v4.0/RT), v2 the endmember ledger (E1) "
            f"- rerun from the config (no migration code, PRD 0.3)")
    for key, current in (("kinetic_phase_ids", KINETIC_PHASE_IDS),
                         ("solid_phase_ids", SOLID_PHASE_IDS),
                         ("element_ids", ELEMENT_IDS)):
        if tuple(header[key]) != current:
            extra = sorted(set(current) - set(header[key]))
            hint = ""
            if extra:
                # E3 added the soluble salt carriers to the kinetic/solid
                # channel lists; name them so the failure is self-diagnosing
                # rather than an opaque list diff (PRD 5: no silent fallback,
                # and no migration code either - rerun from the config)
                hint = (f" (this build adds {extra}; a pre-E3 checkpoint has "
                        f"no channel for them - rerun from the config)")
            raise StorageError(
                f"checkpoint {key} {header[key]} does not match this build "
                f"{list(current)} - ledger vectors would be misinterpreted"
                f"{hint}")
    hydrate_ids = tuple(header["hydrate_phase_ids"])  # run-scoped, header-owned
    endmember_ids = tuple((str(h), str(dc)) for h, dc in header["endmember_ids"])
    config = TinnConfig.model_validate(header["config"])
    if config.config_hash() != header["config_hash"]:
        raise StorageError(
            "checkpoint config_hash does not match the hash of its own embedded "
            "config under this build - the config schema changed between versions; "
            "restarting would silently poison run provenance (no silent fallback)")
    # RT reproducibility: the external chemistry inputs must be byte-identical.
    # A checkpoint predating this check has no key -> warn and proceed (there is
    # nothing to compare against). Present -> recompute from the embedded
    # config's paths and hard-fail on any content-hash mismatch unless the
    # explicit TINN_ALLOW_CHEMISTRY_MISMATCH override is set (versions are
    # recorded for provenance but not enforced).
    stored_ident = header.get("chemistry_identity")
    if stored_ident is None:
        import warnings
        warnings.warn(
            "checkpoint predates the chemistry-identity record; the GEMS "
            "bundle / PHREEQC database content cannot be verified on this "
            "restart (rerun from the config for a guaranteed-consistent run)",
            stacklevel=2)
    else:
        current_ident = chemistry_identity(config)
        drift = [k for k in ("gems_bundle_sha256", "phreeqc_dat_sha256")
                 if stored_ident.get(k) != current_ident.get(k)]
        if drift and not os.environ.get("TINN_ALLOW_CHEMISTRY_MISMATCH"):
            raise StorageError(
                f"checkpoint chemistry inputs changed since it was written "
                f"{drift}: the GEMS bundle or PHREEQC database content no "
                f"longer matches (a changed equilibrium constant under the "
                f"same name is exactly what this guards). Restarting would "
                f"mix chemistries - refused. Set TINN_ALLOW_CHEMISTRY_MISMATCH "
                f"to override deliberately, or rerun from the config.\n"
                f"  stored:  {stored_ident.get('engine_versions')}\n"
                f"  current: {current_ident.get('engine_versions')}")

    arrays = {f: _read_zarr_array(root / "arrays" / f) for f in _DENSE_FIELDS}
    cluster_inventory = _read_zarr_array(root / "arrays" / "cluster_inventory")
    cluster_endmember_mol = _read_zarr_array(
        root / "arrays" / "cluster_endmember_mol")
    domain_eq_inventory = _read_zarr_array(
        root / "arrays" / "domain_eq_inventory")
    domain_eq_water = _read_zarr_array(root / "arrays" / "domain_eq_water")
    domain_eq_age = _read_zarr_array(
        root / "arrays" / "domain_eq_age").astype(np.int64)
    domain_species_mol = _read_zarr_array(
        root / "arrays" / "domain_species_mol")
    domain_sorbed_mol = _read_zarr_array(
        root / "arrays" / "domain_sorbed_mol")

    tables = json.loads((root / "tables.json").read_text(encoding="utf-8"))

    def as_table(name: str) -> Dict[str, np.ndarray]:
        schema = _TABLE_SCHEMAS[name]
        out = {}
        for k, v in tables[name].items():
            arr = np.asarray(v, dtype=np.dtype(schema.get(k, "<f8")))
            if k == "center_zyx":
                arr = arr.reshape(-1, 3)  # empty lists must keep the (0, 3) shape
            out[k] = arr
        return out

    return SimulationState(
        config=config,
        hydrate_ids=hydrate_ids,
        **arrays,
        particles=as_table("particles"),
        subgrid_bins=as_table("subgrid_bins"),
        parcels=tables["parcels"],
        remap_events=tables["remap_events"],
        cluster_inventory=cluster_inventory,
        domain_eq_inventory=domain_eq_inventory,
        domain_eq_water=domain_eq_water,
        domain_eq_age=domain_eq_age,
        cluster_endmember_mol=cluster_endmember_mol,
        aq_species_ids=tuple(header.get("aq_species_ids", [])),
        domain_species_mol=domain_species_mol,
        domain_sorbed_mol=domain_sorbed_mol,
        time_h=header["time_h"],
        dt_h=header["dt_h"],
        phase_mol=np.asarray(header["phase_mol"]),
        initial_phase_mol=np.asarray(header["initial_phase_mol"]),
        unmet_mol=np.asarray(header["unmet_mol"]),
        hydrate_mol=np.asarray(header["hydrate_mol"]),
        hydrate_env_vol_vox=np.asarray(header["hydrate_env_vol_vox"]),
        hydrate_elements_ch=np.asarray(header["hydrate_elements_ch"]).reshape(
            len(hydrate_ids), len(ELEMENT_IDS)),
        endmember_ids=endmember_ids,
        endmember_mol=np.asarray(header["endmember_mol"]),
        endmember_elements=np.asarray(header["endmember_elements"]).reshape(
            len(endmember_ids), len(ELEMENT_IDS)),
        boundary_exchanged_elements=np.asarray(
            header["boundary_exchanged_elements"]),
        injected_elements=np.asarray(header["injected_elements"]),
        water_free_mol=header["water_free_mol"],
        water_gel_mol=header["water_gel_mol"],
        water_bound_mol=header["water_bound_mol"],
        initial_water_mol=header["initial_water_mol"],
        boundary_water_mol=header["boundary_water_mol"],
        inert_volume_vox=header["inert_volume_vox"],
        initial_elements=np.asarray(header["initial_elements"]),
        accept_count=header["accept_count"],
        reject_counts=dict(header["reject_counts"]),
        rng_state=header["rng_state"],
        config_hash=header["config_hash"],
        backend_id=header["backend_id"],
    )
