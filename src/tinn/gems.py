"""xGEMS ChemicalEngine adapter: isolated worker process + PC bundle hash audit +
0D cumulative chemistry probe (PRD §3 M3).

The worker runs in a separate process (`python -m tinn.gems --worker req resp`)
with its own working directory, so ipmlog.txt/xGEMS.log never pollute the source
bundle, and xgems is imported ONLY inside the worker — every other tinn feature
works without xgems installed (PRD §5). The source bundle is hash-audited before
and after every call; any mutation is a hard error.

The probe equilibrates ONLY what kinetics released (plus total mix water and an
O2 seed) with clinker phases suppressed — unreacted clinker is never an input
(PRD §2.2).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

SUPPRESSED_CLINKER_PHASES = ("Alite", "Belite", "Aluminate", "Ferrite")
# Worker wire-protocol version. 2 (RT-P0a): responses carry
# protocol_version, info() carries species_charge, equilibrate responses
# carry aqueous_species_mol. Older workers simply omit the keys — the
# parent tolerates that (response.get defaults) unless species transport
# is configured, in which case GemsWorker.require_speciation() refuses.
PROTOCOL_VERSION = 2
SUCCESS_STATUSES = (
    "No GEM re-calculation needed",
    "OK after GEM calculation with LPP AIA",
    "OK after GEM calculation with SIA",
)
CHARGE_ELEMENT_ID = "Zz"
BULK_VERIFY_ATOL_MOL = 1e-12
BULK_VERIFY_RTOL = 1e-10
# tiny O2 seed so the aqueous redox state is well-posed; 1e-9 at the canonical
# input scale keeps the injected mass ~1e-7 relative (below the ledger's 1e-6
# injected-excess bound) while still converging (1e-10 does not)
O2_SEED_MOL_O = 1e-9
# RT-S3: the reduced-sulfur species of cemdata18 (aqueous + native sulfur)
# and the sulfide phases of the PC bundle - the S(VI)-only exclusion set a
# config declares via chemistry.suppressed_species / suppressed_phases
REDUCED_SULFUR_SPECIES = ("HS-", "H2S@", "S-2", "S2O3-2", "HSO3-", "SO3-2",
                          "H2S", "Sulfur")
SULFIDE_PHASES = ("Pyrite", "Troilite")


class GemsError(RuntimeError):
    """kind: 'config' | 'nonconvergence' | 'timeout' | 'crash' | 'protocol' |
    'internal' — a stable discriminator so M4 can reject trials on
    nonconvergence but hard-fail on configuration errors."""

    def __init__(self, message: str, kind: str = "internal"):
        super().__init__(message)
        self.kind = kind


class BundleAuditError(GemsError):
    def __init__(self, message: str):
        super().__init__(message, kind="config")


# sha256 memo keyed by (path, size, mtime_ns): the audit runs before AND after
# every call (hundreds of times per step at late age), and rehashing an
# unchanged 35k-line bundle each time cost ~5 % of a step. A content change
# always changes the stat signature in practice; the audit guarantee is kept.
_AUDIT_SHA_CACHE: Dict[tuple, str] = {}


def audit_bundle(dat_lst_path: str) -> Dict[str, str]:
    """sha256 of every file under the bundle directory, recursively (catches log
    files written into new subdirectories too). Bundles whose .lst references
    files OUTSIDE its own directory are not fully covered — keep bundles
    self-contained like the canonical 5-file PC bundle."""
    root = Path(dat_lst_path).resolve()
    if not root.is_file():
        raise GemsError(f"bundle .lst not found: {root}", kind="config")
    digests: Dict[str, str] = {}
    for f in sorted(root.parent.rglob("*")):
        if f.is_file():
            st = f.stat()
            key = (str(f), st.st_size, st.st_mtime_ns)
            sha = _AUDIT_SHA_CACHE.get(key)
            if sha is None:
                h = hashlib.sha256()
                h.update(f.read_bytes())
                sha = h.hexdigest()
                _AUDIT_SHA_CACHE[key] = sha
            digests[f.relative_to(root.parent).as_posix()] = sha
    return digests


@dataclass(frozen=True)
class SpeciesDwTable:
    """Vendored per-species diffusion data (RT-P0a, PRD 4.6.4).

    dw: 25 C self-diffusion coefficient (m^2/s) by phreeqc species name;
    z: ionic charge by phreeqc species name; aliases: GEMS DC name ->
    phreeqc species name (hand-curated identities only)."""
    dw: Dict[str, float]
    z: Dict[str, float]
    aliases: Dict[str, str]
    sha256: str          # of the json file itself (audit trail)
    source: str

    def lookup(self, gems_dc: str):
        """(dw_m2_s, z) for a GEMS DC name, or None when unmapped."""
        name = gems_dc if gems_dc in self.dw else self.aliases.get(gems_dc)
        if name is None or name not in self.dw:
            return None
        return self.dw[name], self.z[name]


def load_species_dw(path: str) -> SpeciesDwTable:
    p = Path(path)
    if not p.is_file():
        raise GemsError(f"species dw table not found: {p}", kind="config")
    raw = p.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    rows = payload.get("species")
    if not rows:
        raise GemsError(f"species dw table {p} has no species rows",
                        kind="config")
    dw: Dict[str, float] = {}
    z: Dict[str, float] = {}
    for row in rows:
        name = str(row["species"])
        val = float(row["dw_25C_m2_s"])
        if name in dw:
            raise GemsError(f"duplicate species {name!r} in {p}", kind="config")
        if not math.isfinite(val) or not (0.0 < val < 1e-7):
            raise GemsError(f"implausible dw {val!r} for {name!r} in {p}",
                            kind="config")
        dw[name] = val
        z[name] = float(row["z"])
    aliases = {str(k): str(v) for k, v in (payload.get("aliases") or {}).items()}
    dangling = sorted(v for v in set(aliases.values()) if v not in dw)
    if dangling:
        raise GemsError(
            f"dw table aliases point at unknown species: {dangling}",
            kind="config")
    return SpeciesDwTable(dw=dw, z=z, aliases=aliases,
                          sha256=hashlib.sha256(raw).hexdigest(),
                          source=str(payload.get("source", "unknown")))


@dataclass
class GemsResult:
    status: str
    ph: float                  # NaN when the solver cannot provide it
    ph_status: str             # 'ok' | 'not_available' (PRD §2.2: NaN, never 0)
    ionic_strength: float
    ionic_strength_status: str
    phase_amounts_mol: Dict[str, float]
    phase_masses_kg: Dict[str, float]
    phase_volumes_m3: Dict[str, float]
    phase_elements_mol: Dict[str, Dict[str, float]]
    # per-solid-phase endmember (DC) mols — E1, PRD 2.3 rev.3
    phase_species_mol: Dict[str, Dict[str, float]] = field(default_factory=dict)
    aqueous_h2o_mol: Optional[float] = None  # solvent split of the aqueous phase
    element_input: Dict[str, object] = field(default_factory=dict)
    xgems_version: str = "not_available"
    # RT-P0a: full aqueous per-species mols (H2O@ included); empty when the
    # worker predates PROTOCOL_VERSION 2 (tolerated unless species transport
    # is configured — see GemsWorker.require_speciation)
    aqueous_species_mol: Dict[str, float] = field(default_factory=dict)

    def floor_adjust_max_rel(self) -> float:
        """Largest cleared-state floor clamp relative to the requested inventory —
        how far the equilibrated composition deviates from the physical ledger."""
        adjust = self.element_input.get("floor_adjustments_mol") or {}
        requested = self.element_input.get("requested_element_mol") or {}
        scale = max((abs(v) for v in requested.values()), default=0.0)
        if scale == 0.0:
            return 0.0
        return max((v for v in adjust.values()), default=0.0) / scale

    def element_closure_max_rel(self) -> float:
        """Max relative element imbalance: sum over phases vs effective input."""
        effective = self.element_input.get("effective_element_mol")
        if not effective:
            raise GemsError("element closure needs an elements-mode result")
        worst = 0.0
        for el, target in effective.items():
            if el == CHARGE_ELEMENT_ID:
                continue
            total = sum(pe.get(el, 0.0) for pe in self.phase_elements_mol.values())
            worst = max(worst, abs(total - target) / (1e-30 + abs(target)))
        return worst


class GemsWorker:
    """Spawns one isolated xGEMS process per equilibration request."""

    def __init__(self, bundle_lst: str, python_executable: Optional[str] = None,
                 work_root: Optional[str] = None, timeout_s: float = 300.0,
                 persistent: Optional[bool] = None):
        self.bundle_lst = str(Path(bundle_lst).resolve())
        self.python_executable = str(python_executable or sys.executable)
        if not Path(self.python_executable).is_file():
            raise GemsError(
                f"worker interpreter not found: {self.python_executable}", kind="config")
        self.work_root = (Path(work_root) if work_root
                          else Path.cwd() / ".gems_runs").resolve()
        self.timeout_s = timeout_s
        # persistent server amortizes interpreter+xgems+bundle load (~0.3 s)
        # across calls; every request still builds a FRESH ChemicalEngine with
        # cold_start, so outputs are bit-identical to spawn-per-call.
        # Infrastructure toggle (like the interpreter path), not config.
        if persistent is None:
            persistent = os.environ.get("TINN_GEMS_PERSISTENT", "1") != "0"
        self.persistent = persistent
        self._proc = None
        self._serve_dir: Optional[Path] = None
        self._req_counter = 0
        self._info_cache: Optional[Dict] = None
        self.baseline_audit = audit_bundle(self.bundle_lst)

    # ------------------------------------------------------- persistent server
    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._serve_dir is not None:
                    (self._serve_dir / "stop").write_text("", encoding="utf-8")
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
            self._proc = None
        if self._serve_dir is not None:
            shutil.rmtree(self._serve_dir, ignore_errors=True)
            self._serve_dir = None

    def __del__(self):  # best effort
        self.close()

    def _ensure_server(self) -> Path:
        if self._proc is not None and self._proc.poll() is None:
            return self._serve_dir
        self._serve_dir = self.work_root / f"serve-{uuid.uuid4().hex[:12]}"
        self._serve_dir.mkdir(parents=True, exist_ok=True)
        self._req_counter = 0
        src_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
        self._proc = subprocess.Popen(
            [self.python_executable, "-m", "tinn.gems", "--serve",
             str(self._serve_dir)],
            cwd=self._serve_dir, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return self._serve_dir

    def _persistent_call(self, request: Dict) -> Dict:
        import time
        serve_dir = self._ensure_server()
        k = self._req_counter
        self._req_counter += 1
        req = serve_dir / f"req_{k:06d}.json"
        resp = serve_dir / f"resp_{k:06d}.json"
        ready = serve_dir / f"resp_{k:06d}.ready"
        req.write_text(json.dumps(request), encoding="utf-8")
        (serve_dir / f"req_{k:06d}.ready").write_text("", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_s
        while not ready.is_file():
            if self._proc.poll() is not None:
                self._proc = None
                raise GemsError(
                    f"persistent xGEMS worker died (see {serve_dir})", kind="crash")
            if time.monotonic() > deadline:
                self.close()
                raise GemsError(
                    f"persistent xGEMS worker timed out after {self.timeout_s}s "
                    f"(artifacts in {serve_dir})", kind="timeout")
            time.sleep(0.005)
        try:
            response = json.loads(resp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            self.close()
            raise GemsError(
                f"persistent worker response is not valid JSON: {e} "
                f"(artifacts in {serve_dir})", kind="protocol") from e
        # bound on-disk growth: a long run makes tens of thousands of calls
        for f in (req, serve_dir / f"req_{k:06d}.ready", resp, ready):
            try:
                f.unlink()
            except OSError:
                pass
        return response

    # ---------------------------------------------------------------- public
    def info(self) -> Dict:
        """Bundle metadata: phase names, independent element names, species per
        phase — needed to fix the run's hydrate channel set before state creation."""
        run_dir = self.work_root / f"info-{uuid.uuid4().hex[:12]}"
        result = self._raw_call({"mode": "info", "dat_lst": self.bundle_lst}, run_dir)
        return result

    def _cached_info(self) -> Dict:
        """info(), fetched once — the bundle is audit-immutable, so caching is
        exact. Internal (info() itself stays a fresh call: cheap, and tests
        rely on it exercising the worker round-trip)."""
        if self._info_cache is None:
            self._info_cache = self.info()
        return self._info_cache

    def require_speciation(self) -> None:
        """Refuse to serve species transport with a pre-v2 worker (spec §2.3:
        old response + Tier 0 active = error; inactive = ignore)."""
        got = int(self._cached_info().get("protocol_version", 1))
        if got < 2:
            raise GemsError(
                f"species transport needs worker protocol >= 2 "
                f"(aqueous speciation in responses); this worker speaks "
                f"{got}. Rebuild/redeploy the worker environment.",
                kind="protocol")

    def equilibrate_stored(self) -> GemsResult:
        """Re-equilibrate the bundle's stored DBR node (PRD §6.2 anchor 1)."""
        return self._call({"mode": "stored", "dat_lst": self.bundle_lst})

    def equilibrate_elements(self, element_mol: Mapping[str, float],
                             temperature_k: float, pressure_pa: float = 1e5,
                             suppressed_phases: Sequence[str] = SUPPRESSED_CLINKER_PHASES,
                             suppressed_species: Sequence[str] = (),
                             ) -> GemsResult:
        """Cold-start equilibration of an authoritative element inventory with
        clinker phases suppressed (bound=0). suppressed_species (RT-S3) are
        dependent components excluded the same way (typo-guarded by the
        worker, leak-witnessed here)."""
        for el, v in element_mol.items():
            if el == CHARGE_ELEMENT_ID:
                if v != 0.0:
                    raise GemsError(
                        "charge (Zz) targets are not supported - released "
                        "inventories are charge-neutral by construction", kind="config")
                continue
            if not math.isfinite(v) or v < 0.0:
                raise GemsError(f"element amount {el}={v} must be finite and >= 0",
                                kind="config")
        # Filter the DEFAULT suppression list to phases the bundle declares —
        # the engine path (GemsBackend.__init__) and run_0d_probe already do,
        # and without it a hydrates-only bundle (CNASH Test) hard-errored on
        # every 0D call (measured 2026-09-01). An EXPLICIT caller-provided
        # list is passed through unfiltered: the worker's unknown-name error
        # stays a typo guard for deliberate suppression requests.
        if suppressed_phases is SUPPRESSED_CLINKER_PHASES:
            present = set(self._cached_info()["phase_names"])
            suppressed_phases = tuple(
                p for p in SUPPRESSED_CLINKER_PHASES if p in present)
        result = self._call({
            "mode": "elements",
            "dat_lst": self.bundle_lst,
            "element_mol": {str(k): float(v) for k, v in element_mol.items()},
            "temperature_k": float(temperature_k),
            "pressure_pa": float(pressure_pa),
            "suppressed_phases": list(suppressed_phases),
            "suppressed_species": [str(s) for s in suppressed_species],
        })
        # suppression witness, mirroring the engine path (gems.py react):
        # bound=0 leaves numerical dust, anything material means the
        # suppression LEAKED. Measured 2026-09-02: the CO3_SO4_AFt solid
        # solution ignored suppress_multiple_phases on this 0D path and
        # precipitated up to 45% of the input sulfur while its sibling
        # SO4_CO3_AFt and every single-DC phase suppressed fine - a
        # silent leak would contaminate every crosscheck built on this
        # API (no silent fallback, PRD 0.3).
        total_in = sum(v for k, v in element_mol.items()
                       if k != CHARGE_ELEMENT_ID)
        for phase in suppressed_phases:
            amt = (result.phase_amounts_mol or {}).get(phase, 0.0)
            if amt > 1e-9 * total_in:
                raise GemsError(
                    f"suppressed phase {phase} precipitated {amt!r} mol "
                    f"(input total {total_in!r}) - suppression leaked "
                    f"(known xgems issue for some solid solutions, e.g. "
                    f"CO3_SO4_AFt); drop it from the request or treat the "
                    f"result as unsuppressed", kind="internal")
        for sp in suppressed_species:
            # aqueous species travel in aqueous_species_mol (protocol v2),
            # the others per phase - a witness reading one map only would
            # be blind to the very HS- this exclusion exists for
            amt = float((result.aqueous_species_mol or {}).get(sp, 0.0))
            for d in (result.phase_species_mol or {}).values():
                if isinstance(d, dict):
                    amt += float(d.get(sp, 0.0))
            if amt > 1e-9 * total_in:
                raise GemsError(
                    f"suppressed species {sp} formed {amt!r} mol (input total "
                    f"{total_in!r}) - species suppression leaked",
                    kind="internal")
        return result

    # --------------------------------------------------------------- private
    def _call(self, request: Dict) -> GemsResult:
        run_dir = self.work_root / f"run-{uuid.uuid4().hex[:12]}"
        response = self._raw_call(request, run_dir)
        ph = response["pH"]
        ionic = response["ionic_strength"]
        return GemsResult(
            status=response["status"],
            ph=math.nan if ph is None else float(ph),
            ph_status="not_available" if ph is None else "ok",
            ionic_strength=math.nan if ionic is None else float(ionic),
            ionic_strength_status="not_available" if ionic is None else "ok",
            phase_amounts_mol=response["phase_amounts_mol"],
            phase_masses_kg=response["phase_masses_kg"],
            phase_volumes_m3=response["phase_volumes_m3"],
            phase_elements_mol=response["phase_elements_mol"],
            phase_species_mol=response.get("phase_species_mol", {}),
            aqueous_h2o_mol=response.get("aqueous_h2o_mol"),
            element_input=response.get("element_input", {}),
            xgems_version=response.get("xgems_version", "not_available"),
            aqueous_species_mol=response.get("aqueous_species_mol", {}),
        )

    def _raw_call(self, request: Dict, run_dir: Path) -> Dict:
        before = audit_bundle(self.bundle_lst)
        if before != self.baseline_audit:
            changed = sorted(k for k in set(before) | set(self.baseline_audit)
                             if before.get(k) != self.baseline_audit.get(k))
            raise BundleAuditError(
                f"bundle changed since worker construction: {changed}")
        if self.persistent:
            response = self._persistent_call(request)
            after = audit_bundle(self.bundle_lst)
            if after != before:
                changed = sorted(k for k in set(before) | set(after)
                                 if before.get(k) != after.get(k))
                raise BundleAuditError(
                    f"xGEMS call mutated the source bundle: {changed}")
            if not response.get("ok"):
                kind = str(response.get("error_kind", "internal"))
                raise GemsError(
                    f"xGEMS worker failed [{kind}]: {response.get('error')}\n"
                    f"{response.get('traceback', '')}", kind=kind)
            return response
        run_dir.mkdir(parents=True, exist_ok=False)
        req_path = run_dir / "request.json"
        resp_path = run_dir / "response.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        src_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
        timed_out = False
        proc = None
        try:
            proc = subprocess.run(
                [self.python_executable, "-m", "tinn.gems", "--worker",
                 str(req_path), str(resp_path)],
                cwd=run_dir, env=env, capture_output=True, text=True,
                timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        # the bundle audit runs on EVERY outcome, including timeouts
        after = audit_bundle(self.bundle_lst)
        if after != before:
            changed = sorted(k for k in set(before) | set(after)
                             if before.get(k) != after.get(k))
            raise BundleAuditError(
                f"xGEMS call mutated the source bundle: {changed}"
                + (" (worker timed out)" if timed_out else ""))
        if timed_out:
            raise GemsError(
                f"xGEMS worker timed out after {self.timeout_s}s "
                f"(artifacts kept in {run_dir})", kind="timeout")
        if not resp_path.is_file():
            raise GemsError(
                f"worker produced no response (exit {proc.returncode}); "
                f"artifacts kept in {run_dir}\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}",
                kind="crash")
        try:
            response = json.loads(resp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise GemsError(
                f"worker response is not valid JSON (exit {proc.returncode}): {e}; "
                f"artifacts kept in {run_dir}\nstderr: {proc.stderr[-2000:]}",
                kind="protocol") from e
        if not response.get("ok"):
            kind = str(response.get("error_kind", "internal"))
            if kind == "nonconvergence":
                # recurring, expected failure mode (e.g. dry pockets probed every
                # step) — retaining thousands of artifact dirs helps nobody
                shutil.rmtree(run_dir, ignore_errors=True)
                raise GemsError(
                    f"xGEMS worker failed [nonconvergence]: {response.get('error')}",
                    kind=kind)
            raise GemsError(
                f"xGEMS worker failed [{kind}]: {response.get('error')}\n"
                f"{response.get('traceback', '')}\n"
                f"artifacts kept in {run_dir}", kind=kind)
        # solver artifacts are only discarded on success
        shutil.rmtree(run_dir, ignore_errors=True)
        return response


# --------------------------------------------------------------- ReactionBackend

AQUEOUS_PHASE = "aq_gen"
GAS_PHASE = "gas_gen"
# equilibrium is intensive: inputs are scaled to a canonical magnitude where the
# solver is accurate (RVE clusters hold ~1e-10 mol; floors are ~1e-15 mol) and
# every extensive output is scaled back exactly
CANONICAL_MAX_ELEMENT_MOL = 1e-2


class GemsBackend:
    """ReactionBackend over an isolated xGEMS worker (PRD §2.2).

    Per cluster: (existing solution inventory + newly released elements + free
    water) -> equilibrate with clinker suppressed -> (new solid parcels with
    their own element vectors and skeleton volumes, residual solution inventory).
    Unreacted clinker is never an input; re-dissolution of existing hydrates is
    inactive (they are simply never fed back). Nonconvergence/timeouts raise
    BackendTransientError (trial reject); config errors propagate.
    """

    backend_id = "gems3k"
    # full re-equilibration (PRD v2.2): parcels are the cluster's ENTIRE new
    # assemblage and water_consumed_mol may be negative (re-dissolution
    # returning bound water to solution)
    mode = "snapshot"

    def __init__(self, worker: GemsWorker, temperature_k: float,
                 registry=None, suppressed_species: Sequence[str] = (),
                 suppressed_phases: Sequence[str] = ()):
        from collections import OrderedDict
        from .registry import (ELEMENT_IDS, KINETIC_PHASE_IDS, default_registry,
                               element_vector)
        # exact input-hash memoization: a quiescent cluster whose scaled input
        # dict is bitwise-identical to a previous solve reuses that response
        # (deterministic, no physics change); bounded LRU
        self._memo: "OrderedDict[str, Dict]" = OrderedDict()
        self.memo_cap = 256   # raised by the engine for mode-C runs
        self._element_ids = ELEMENT_IDS
        self._worker = worker
        self.temperature_k = temperature_k
        info = worker.info()
        # RT-Cl: the ledger may be a SUPERSET of the bundle's independent
        # components (PC/CNASH carry no Cl, PC-Cl does). An absent element
        # is a zero column in every matrix below; feeding it mass is a
        # hard error in react() - never a silent drop.
        self._absent_elements = tuple(
            sorted(set(ELEMENT_IDS) - set(info["element_names"])))
        excluded = set(SUPPRESSED_CLINKER_PHASES) | {AQUEOUS_PHASE, GAS_PHASE}
        self.hydrate_ids = tuple(p for p in info["phase_names"] if p not in excluded)
        # suppress only the clinker phases this bundle actually declares — a
        # hydrates-only bundle (e.g. CNASH Test) has none, and the worker
        # hard-errors on unknown suppression names by design (typo guard)
        self._suppressed = tuple(p for p in SUPPRESSED_CLINKER_PHASES
                                 if p in set(info["phase_names"]))
        # RT-S3: config-declared exclusions ride the same request; unknown
        # names are the worker's typo guard, leaks are witnessed per call
        self._suppressed = self._suppressed + tuple(
            p for p in suppressed_phases if p not in self._suppressed)
        self._suppressed_species = tuple(suppressed_species)
        # E1 (PRD 2.3 rev.3): endmember universe per hydrate channel and each
        # endmember's element row — both straight from the bundle (info's
        # phase_species + DCH stoichiometry matrix), never hardcoded
        self.hydrate_endmembers = {
            h: tuple(info["phase_species"][h]) for h in self.hydrate_ids}
        species_elements = info.get("species_elements")
        if species_elements is None:
            raise GemsError(
                "worker info lacks species_elements — bundle DCH stoichiometry "
                "is required for the endmember ledger (E1)", kind="config")
        import numpy as _np
        el_idx = {el: i for i, el in enumerate(ELEMENT_IDS)}
        self.endmember_elements = {}
        for h in self.hydrate_ids:
            for dc in self.hydrate_endmembers[h]:
                row = species_elements.get(dc)
                if row is None:
                    raise GemsError(
                        f"DCH has no stoichiometry row for endmember {dc!r} "
                        f"of phase {h!r}", kind="config")
                vec = _np.zeros(len(ELEMENT_IDS))
                for el, coeff in row.items():
                    if el in el_idx:  # charge (Zz) already dropped worker-side
                        vec[el_idx[el]] = float(coeff)
                self.endmember_elements[dc] = vec
        # RT-P0a: run-scoped aqueous species universe for the NP weighting —
        # names (solvent excluded: transported solutes only), charges (DCH Zz
        # column, worker protocol >= 2; empty dict on older workers, refused
        # later by require_speciation when species transport is configured),
        # and element rows over ELEMENT_IDS. Never hardcoded.
        aq_all = tuple(info["phase_species"].get(AQUEOUS_PHASE, ()))
        self.aq_species_ids = tuple(dc for dc in aq_all if dc != "H2O@")
        charge_map = info.get("species_charge") or {}
        self.aq_species_charge = _np.array(
            [float(charge_map.get(dc, 0.0)) for dc in self.aq_species_ids])
        self.aq_species_elements = _np.zeros(
            (len(self.aq_species_ids), len(ELEMENT_IDS)))
        for k, dc in enumerate(self.aq_species_ids):
            row = species_elements.get(dc)
            if row is None:
                raise GemsError(
                    f"DCH has no stoichiometry row for aqueous species {dc!r}",
                    kind="config")
            for el, coeff in row.items():
                if el in el_idx:
                    self.aq_species_elements[k, el_idx[el]] = float(coeff)
        # the RUN's registry: a config-declared SCM composition (PRD 1.2
        # v3.0) must reach the elements this backend is actually fed
        reg = registry or default_registry()
        self._formula_vec = {p: element_vector(reg.get(p).formula, 1.0)
                             for p in KINETIC_PHASE_IDS}
        self._h2o_vec = element_vector(reg.get("H2O").formula, 1.0)
        self._h2o_index = {el: i for i, el in enumerate(ELEMENT_IDS)}

    def react(self, released_mol, water_available_mol: float, inventory,
              solid_elements=None):
        import numpy as np
        from .backend import (BackendTransientError, Parcel, ReactionResult,
                              STATUS_OK)

        e_ids = self._element_ids
        elements = np.asarray(inventory, dtype=np.float64).copy()
        for phase_id, mol in released_mol.items():
            if mol > 0.0:
                elements = elements + self._formula_vec[phase_id] * mol
        if solid_elements is not None:
            # full re-equilibration: the cluster's owned hydrate elements are
            # part of the system and may re-dissolve (PRD v2.2)
            elements = elements + np.asarray(solid_elements, dtype=np.float64)
        elements = elements + self._h2o_vec * water_available_mol
        # tiny negatives are float dust from previous residual splits
        elements = np.where(np.abs(elements) < 1e-30, 0.0, elements)
        if np.any(elements < 0.0):
            raise GemsError(f"negative element input: {dict(zip(e_ids, elements))}",
                            kind="config")
        peak = float(elements.max())
        if peak <= 0.0:
            raise GemsError("empty element input to GEMS backend", kind="config")
        s = CANONICAL_MAX_ELEMENT_MOL / peak

        for el in self._absent_elements:
            if elements[e_ids.index(el)] > 0.0:
                raise GemsError(
                    f"the reactor carries {elements[e_ids.index(el)]!r} mol "
                    f"of {el} but the bundle declares no {el} component - "
                    f"use a bundle with {el} (e.g. gems_bundles/PC-Cl)",
                    kind="config")
        scaled = {el: float(elements[i] * s) for i, el in enumerate(e_ids)
                  if elements[i] > 0.0}
        injected = np.zeros(len(e_ids))
        if not np.any(np.asarray(inventory) != 0.0):
            # first equilibration of this cluster: O2 seed fixes the redox state;
            # it returns via the residual inventory on later calls
            scaled["O"] = scaled.get("O", 0.0) + O2_SEED_MOL_O
            injected[e_ids.index("O")] += O2_SEED_MOL_O / s

        memo_key = json.dumps(
            {k: scaled[k].hex() if hasattr(scaled[k], "hex") else scaled[k]
             for k in sorted(scaled)}, sort_keys=True)
        if memo_key in self._memo:
            self._memo.move_to_end(memo_key)
            r = self._memo[memo_key]
        else:
            try:
                r = self._worker.equilibrate_elements(
                    scaled, self.temperature_k, suppressed_phases=self._suppressed,
                    suppressed_species=self._suppressed_species)
            except GemsError as e:
                if e.kind in ("nonconvergence", "timeout"):
                    raise BackendTransientError(str(e)) from e
                raise
            self._memo[memo_key] = r
            if len(self._memo) > self.memo_cap:
                self._memo.popitem(last=False)

        for el, adj in r.element_input["floor_adjustments_mol"].items():
            if el in self._h2o_index:
                injected[self._h2o_index[el]] += adj / s
        # the engine may hold slightly more/less than asked (verification slack);
        # book that signed slack too so the ledger tracks the solver exactly
        for el, slack in r.element_input.get("verification_slack_mol", {}).items():
            if el in self._h2o_index and slack != 0.0:
                injected[self._h2o_index[el]] += slack / s

        total_scaled = float(sum(scaled.values()))
        parcels = []
        residual_phases = [AQUEOUS_PHASE, GAS_PHASE]
        for phase, mol_scaled in r.phase_amounts_mol.items():
            if phase in (AQUEOUS_PHASE, GAS_PHASE) or mol_scaled <= 0.0:
                continue
            if phase in self._suppressed:
                # suppression (bound=0) can leave numerical dust; dust elements
                # stay in solution, anything material is precipitating clinker.
                # The threshold is relative to the call's own input magnitude so
                # big and small clusters get the same relative strictness.
                if mol_scaled > 1e-9 * total_scaled:
                    raise GemsError(
                        f"suppressed clinker phase {phase} precipitated "
                        f"{mol_scaled!r} mol (scaled, input {total_scaled!r}) - "
                        f"suppression failed", kind="internal")
                residual_phases.append(phase)
                continue
            pe = r.phase_elements_mol.get(phase, {})
            vec = np.zeros(len(e_ids))
            for el, v in pe.items():
                if el in self._h2o_index:
                    vec[self._h2o_index[el]] = v / s
            # E1: endmember mols ride the parcel, rescaled by the same 1/s as
            # mol/elements/volume so the endmember-sum identity survives the
            # canonical scaling exactly (worker verified it at scale s). A
            # solid with amount but no species split is a protocol violation
            # (None is reserved for single-endmember synthetic backends whose
            # channel IS the endmember — bundle DC names differ from phase
            # names, so None is never bookable for GEMS parcels).
            em_scaled = r.phase_species_mol.get(phase)
            if not em_scaled:
                raise GemsError(
                    f"worker response lacks the endmember split for solid "
                    f"phase {phase!r}", kind="protocol")
            em = {dc: m / s for dc, m in em_scaled.items()}
            parcels.append(Parcel(
                phase_id=phase, mol=mol_scaled / s, elements=vec,
                skel_vol_cm3=r.phase_volumes_m3.get(phase, 0.0) * 1e6 / s,
                endmember_mol=em))

        if r.aqueous_h2o_mol is None:
            raise GemsError("worker did not split the aqueous solvent", kind="protocol")
        h2o_out = r.aqueous_h2o_mol / s
        aqueous_elements = np.zeros(len(e_ids))
        for el, v in r.phase_elements_mol.get(AQUEOUS_PHASE, {}).items():
            if el in self._h2o_index:
                aqueous_elements[self._h2o_index[el]] = v / s
        residual = np.zeros(len(e_ids))
        for phase in residual_phases:
            for el, v in r.phase_elements_mol.get(phase, {}).items():
                if el in self._h2o_index:
                    residual[self._h2o_index[el]] += v / s
        residual = residual - self._h2o_vec * h2o_out

        # per-call solver closure gate: outputs must equal what the engine held
        # (input + injected) — a biased solver residual must never leak into the
        # cumulative blocking ledger silently
        e_out = residual + self._h2o_vec * h2o_out
        for pc in parcels:
            e_out = e_out + pc.elements
        e_in = elements + injected
        scale_mol = float(np.abs(e_in).max())
        resid_vec = e_out - e_in
        closure = float(np.abs(resid_vec).max()) / max(scale_mol, 1e-300)
        if closure > 1e-9:
            raise BackendTransientError(
                f"xGEMS per-call element closure {closure:.2e} exceeds 1e-9")
        # book the signed sub-gate residual as injected/lost solver mass: under
        # full re-equilibration the WHOLE inventory passes through the solver
        # every step, so per-call dust would otherwise accumulate against the
        # blocking element bound (the 1e-6 injected-excess bound guards drift)
        injected = injected + resid_vec

        return ReactionResult(
            status=STATUS_OK,
            parcels=parcels,
            water_consumed_mol=water_available_mol - h2o_out,
            residual_inventory=residual,
            injected_elements=injected,
            ph=r.ph, ph_status=r.ph_status,
            ionic_strength=r.ionic_strength,
            ionic_strength_status=r.ionic_strength_status,
            aqueous_h2o_mol=h2o_out,
            aqueous_elements=aqueous_elements,
            # RT-P0a: unscale by the same 1/s as every other extensive output;
            # empty dict from a pre-v2 worker propagates as None
            aqueous_species_mol=({dc: m / s
                                  for dc, m in r.aqueous_species_mol.items()}
                                 if r.aqueous_species_mol else None),
        )


# ------------------------------------------------------------------ 0D probe

def run_0d_probe(config, worker: GemsWorker,
                 times_h: Optional[Sequence[float]] = None) -> List[Dict]:
    """Cumulative 0D chemistry probe: at each time, equilibrate exactly what the
    kinetics released so far (per 1 g binder) plus total mix water and O2 seed.
    Returns one row per time with pH, ionic strength, phase masses, and the
    element-closure error. Unreacted clinker is never an input."""
    from .kinetics import make_kinetics
    from .registry import KINETIC_PHASE_IDS, SALT_PHASE_IDS, default_registry

    reg = default_registry()
    kin = make_kinetics(config)
    present = set(worker.info()["phase_names"])
    suppressed = tuple(p for p in SUPPRESSED_CLINKER_PHASES if p in present)
    suppressed = suppressed + tuple(
        p for p in (config.chemistry.suppressed_phases or ())
        if p not in suppressed)
    suppressed_species = tuple(config.chemistry.suppressed_species or ())
    times = list(times_h if times_h is not None else config.schedule.output_times_h)
    n0 = {p: config.binder.mass_fractions.get(p, 0.0) / reg.get(p).molar_mass_g_mol
          for p in KINETIC_PHASE_IDS}  # mol per g binder
    water_mol = config.w_c / reg.get("H2O").molar_mass_g_mol

    rows: List[Dict] = []
    for t in times:
        alpha = kin.alpha_at(t)
        elements: Dict[str, float] = {"O": O2_SEED_MOL_O}
        for k, p in enumerate(KINETIC_PHASE_IDS):
            # E3 parity with the spatial engine: salt carriers have no kinetic
            # alpha.  Their complete initial inventory is offered to GEMS and
            # equilibrium decides what dissolves or remains as a solid.  Using
            # alpha[k] here silently omitted sulfate/alkali from the old 0D
            # reference because P&K correctly returns zero for these channels.
            released = (n0[p] if p in SALT_PHASE_IDS
                        else n0[p] * float(alpha[k]))
            if released <= 0.0:
                continue
            for el, count in reg.get(p).formula.items():
                elements[el] = elements.get(el, 0.0) + released * count
        for el, count in reg.get("H2O").formula.items():
            elements[el] = elements.get(el, 0.0) + water_mol * count

        result = worker.equilibrate_elements(elements, config.temperature_K,
                                             suppressed_phases=suppressed,
                                             suppressed_species=suppressed_species)
        rows.append({
            "time_h": t,
            "alpha": {p: float(alpha[k]) for k, p in enumerate(KINETIC_PHASE_IDS)},
            "ph": result.ph,
            "ph_status": result.ph_status,
            "ionic_strength": result.ionic_strength,
            "phase_amounts_mol": result.phase_amounts_mol,
            "phase_masses_kg": result.phase_masses_kg,
            "phase_elements_mol": result.phase_elements_mol,
            "phase_species_mol": result.phase_species_mol,
            # RT-P0a: aqueous speciation for cross-engine comparison (spike);
            # getattr — probe tests fake the result object without the field
            "aqueous_species_mol": getattr(result, "aqueous_species_mol", {}),
            # solver self-consistency vs the effective (floor-clamped) input
            "element_closure_max_rel": result.element_closure_max_rel(),
            # divergence of the effective input from the physical ledger
            "floor_adjust_max_rel": result.floor_adjust_max_rel(),
            "status": result.status,
        })
    return rows


# ------------------------------------------------------------ worker process

def _dch_species_elements(dat_lst: Path):
    """Per-DC element coefficients AND charges from the bundle's DCH file (the
    -f "..." list in the .lst names it first). Returns
    ({dc_name: {element: coeff}}, {dc_name: charge}) for ALL DCs; the element
    rows drop the charge column (Zz) — the element ledger carries no charge —
    while the charge map keeps it for the Tier 0 NP weighting (RT-P0a).
    Raises on malformed bundles (no guessing)."""
    lst_text = dat_lst.read_text(encoding="utf-8", errors="replace")
    import re
    names = re.findall(r'"([^"]+)"', lst_text)
    dch_name = next((n for n in names if "dch" in n.lower()), None)
    if dch_name is None:
        raise ValueError(f"bundle .lst names no DCH file: {dat_lst}")
    dch_path = dat_lst.parent / dch_name
    try:
        payload = json.loads(dch_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # pre-E1 any xgems-loadable bundle worked; the endmember ledger needs
        # the DCH as JSON (-j export). Refuse key-value (-t) exports with a
        # diagnosable message instead of a bare parse crash.
        raise ValueError(
            f"bundle DCH {dch_path.name} is not JSON — the E1 endmember "
            f"ledger requires GEMS3K JSON (-j) exports, not key-value (-t) "
            f"bundles: {e}") from e
    dch = payload[0]["dch"] if isinstance(payload, list) else payload["dch"]
    ic_names = [str(x) for x in dch["ICNL"]]
    dc_names = [str(x) for x in dch["DCNL"]]
    a = dch["A"]
    n_ic = len(ic_names)
    if len(a) != len(dc_names) * n_ic:
        raise ValueError(
            f"DCH stoichiometry matrix shape mismatch in {dch_path.name}: "
            f"len(A)={len(a)} != nDC({len(dc_names)}) x nIC({n_ic})")
    out: Dict[str, Dict[str, float]] = {}
    charges: Dict[str, float] = {}
    for j, dc in enumerate(dc_names):
        row = a[j * n_ic:(j + 1) * n_ic]
        entry = {ic: float(v) for ic, v in zip(ic_names, row)
                 if ic != CHARGE_ELEMENT_ID and float(v) != 0.0}
        z = next((float(v) for ic, v in zip(ic_names, row)
                  if ic == CHARGE_ELEMENT_ID), 0.0)
        # Cemdata bundles legitimately repeat a DC name across phases (e.g.
        # C4AH13 as pure phase AND AFm endmember) with IDENTICAL rows; a
        # same-named DC with a DIFFERENT stoichiometry would silently
        # mis-map the ledger, so refuse it (no guessing)
        if dc in out and (out[dc] != entry or charges[dc] != z):
            raise ValueError(
                f"DCH declares DC {dc!r} twice with differing stoichiometry "
                f"- endmember rows would be ambiguous")
        out[dc] = entry
        charges[dc] = z
    return out, charges


def _worker_execute(request: Mapping,
                    engine_cache: Optional[Dict[str, object]] = None) -> Dict:
    """Runs inside the isolated worker process — the only place xgems is imported.

    engine_cache (persistent server only): reusing a loaded ChemicalEngine and
    re-running the exact clear/T/P/cold_start/suppress sequence per request is
    BITWISE identical to a fresh construction (measured: identical phase
    amounts and pH) while skipping the ~110 ms bundle re-parse that dominated
    late-age steps. Spawn mode passes None and stays construct-per-call."""
    import xgems  # deferred: absence must not affect any other tinn feature

    try:
        import importlib.metadata
        version = importlib.metadata.version("xgems")
    except Exception:
        version = "not_available"

    dat = Path(str(request["dat_lst"])).resolve()
    if not dat.is_file():
        raise FileNotFoundError(f"worker cannot find dat.lst: {dat}")
    mode = request["mode"]
    element_input: Dict[str, object] = {}

    if mode == "info":
        engine = xgems.ChemicalEngineDicts(str(dat))
        phase_species = {str(p): sorted(str(s) for s in engine.phase_species_amounts(p))
                         for p in engine.phase_names}
        # per-species element rows straight from the bundle's DCH stoichiometry
        # matrix (E1, PRD 2.3 rev.3): the SAME matrix xGEMS solves with, read
        # from the JSON rather than guessed from formulas — endmember ledgers
        # close against it by construction
        species_elements, species_charge = _dch_species_elements(dat)
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "phase_names": [str(p) for p in engine.phase_names],
            "element_names": [str(e) for e in engine.bulk_composition],
            "phase_species": phase_species,
            "species_elements": species_elements,
            # Zz column of the same DCH matrix (RT-P0a) — the NP weighting's
            # charge source, never guessed from species names
            "species_charge": species_charge,
            "xgems_version": version,
        }

    if mode == "stored":
        engine = xgems.ChemicalEngineDicts(str(dat))
    elif mode == "elements":
        if engine_cache is not None and str(dat) in engine_cache:
            engine = engine_cache[str(dat)]
        else:
            engine = xgems.ChemicalEngineDicts(str(dat), reset_calc=True,
                                               cold_start=True)
            if engine_cache is not None:
                engine_cache[str(dat)] = engine
        engine.clear()
        engine.T = float(request["temperature_k"])
        engine.P = float(request["pressure_pa"])
        engine.cold_start()
        suppressed = [str(s) for s in request["suppressed_phases"]]
        missing = sorted(set(suppressed) - {str(p) for p in engine.phase_names})
        if missing:
            raise ValueError(f"bundle lacks phases requested for suppression: {missing}")
        # a cached engine KEEPS phase suppression across clear()/cold_start()
        # too (measured, RT-S3: a request that suppressed Pyrite left it
        # suppressed for the next plain request) - re-activate the previous
        # request's set so every request starts from one activation state
        phase_key = str(dat) + "#suppressed_phases"
        if engine_cache is not None and engine_cache.get(phase_key):
            engine.activate_multiple_phases(list(engine_cache[phase_key]))
        engine.suppress_multiple_phases(suppressed)
        if engine_cache is not None:
            engine_cache[phase_key] = list(suppressed)
        # RT-S3 species exclusion. A cached engine KEEPS species suppression
        # across clear()/cold_start() (measured), unlike the per-request
        # phase list, so the previous request's set is re-activated first -
        # every request then starts from the same activation state.
        species_req = [str(s) for s in request.get("suppressed_species", [])]
        meta_key = str(dat) + "#suppressed_species"
        if engine_cache is not None and engine_cache.get(meta_key):
            engine.activate_multiple_species(list(engine_cache[meta_key]))
        if species_req:
            names = engine.species_names
            names = {str(s) for s in (names() if callable(names) else names)}
            missing = sorted(set(species_req) - names)
            if missing:
                raise ValueError(
                    f"bundle lacks species requested for suppression: {missing}")
            engine.suppress_multiple_species(species_req)
        if engine_cache is not None:
            engine_cache[meta_key] = list(species_req)

        requested = {str(k): float(v) for k, v in request["element_mol"].items()}
        if requested.get(CHARGE_ELEMENT_ID, 0.0) != 0.0:
            raise ValueError("charge (Zz) targets are not supported")
        floors = {str(k): float(v) for k, v in engine.bulk_composition.items()}
        unknown = sorted(set(requested) - set(floors))
        if unknown:
            raise ValueError(f"bundle lacks independent elements: {unknown}")
        # a cleared engine keeps tiny numerical floors; physical elements are
        # additive-only, so targets below the floor clamp to it (reported)
        effective: Dict[str, float] = {}
        floor_adjust: Dict[str, float] = {}
        deltas: Dict[str, float] = {}
        for el, floor in floors.items():
            desired = requested.get(el, 0.0)
            if el == CHARGE_ELEMENT_ID:
                effective[el] = floor  # charge left at the cleared state
                continue
            eff = max(desired, floor)
            effective[el] = eff
            if eff > desired:
                floor_adjust[el] = eff - desired
            if eff - floor > 0.0:
                deltas[el] = eff - floor
        if deltas:
            engine.add_multiple_elements_amt(deltas, units="moles")
        verified = {str(k): float(v) for k, v in engine.bulk_composition.items()}
        for el, eff in effective.items():
            if abs(verified[el] - eff) > BULK_VERIFY_ATOL_MOL + BULK_VERIFY_RTOL * abs(eff):
                raise ValueError(
                    f"bulk composition verification failed for {el}: "
                    f"got {verified[el]!r}, wanted {eff!r}")
        element_input = {
            "requested_element_mol": requested,
            "effective_element_mol": effective,
            "floor_adjustments_mol": floor_adjust,
            # what the engine actually holds minus what we asked it to hold —
            # this slack is real injected/removed mass and must be booked
            "verification_slack_mol": {el: verified[el] - effective[el]
                                       for el in effective},
        }
    else:
        raise ValueError(f"unknown worker mode {mode!r}")

    status = str(engine.equilibrate())
    if status not in SUCCESS_STATUSES:
        raise RuntimeError(f"xGEMS equilibrium did not converge: {status}")

    def _mapping(name: str) -> Dict[str, float]:
        return {str(k): float(v) for k, v in getattr(engine, name).items()}

    def _nullable(value) -> Optional[float]:
        """Non-finite solver outputs travel as null -> NaN + not_available on the
        parent side (PRD §2.2) instead of killing the whole call."""
        v = float(value)
        return v if math.isfinite(v) else None

    phase_elements = {str(ph): {str(e): float(v) for e, v in row.items()}
                      for ph, row in engine.phases_elements_moles.items()}
    phase_amounts = _mapping("phase_amounts")

    # every multi-species phase amount must equal the sum of its species
    # (endmember-sum verification, PRD M4: CSHQ never reinterpreted as a fixed
    # formula) and the aqueous solvent is split out for the water ledger.
    # E1 (PRD 2.3 rev.3): the per-species mols of SOLID phases now ride the
    # response — the parent's endmember ledger is fed by the same numbers the
    # sum verification just checked.
    aqueous_h2o = None
    aqueous_species: Dict[str, float] = {}
    phase_species_mol: Dict[str, Dict[str, float]] = {}
    for phase_name, total in phase_amounts.items():
        species = {str(k): float(v)
                   for k, v in engine.phase_species_amounts(phase_name).items()}
        if species:
            ssum = sum(species.values())
            if abs(ssum - total) > 1e-12 + 1e-9 * abs(total):
                # protocol violation, not solver nonconvergence — must hard-fail
                raise ValueError(
                    f"phase {phase_name} amount {total!r} != endmember sum {ssum!r}")
        if phase_name == "aq_gen":
            aqueous_h2o = species.get("H2O@")
            if aqueous_h2o is None:
                raise ValueError("aqueous phase lacks the H2O@ solvent species")
            # RT-P0a: the full aqueous split rides the response (it was
            # computed here all along and discarded save for the solvent)
            aqueous_species = species
        elif phase_name != GAS_PHASE:
            phase_species_mol[phase_name] = species

    # some bundles report zero phase volume for phases holding a positive
    # amount (observed: single-DC solids in the CNASH Test bundle; InverseGems
    # works around the same defect as "reconstructed_from_phase_species").
    # Rebuild those volumes from species amounts x standard molar volumes —
    # both from the same engine, so no external constants enter the ledger.
    phase_volumes = _mapping("phase_volumes")
    species_mv = {str(k): float(v)
                  for k, v in engine.species_molar_volumes.items()}
    for phase_name, total in phase_amounts.items():
        if phase_name in (AQUEOUS_PHASE, GAS_PHASE):
            continue
        if total > 0.0 and phase_volumes.get(phase_name, 0.0) == 0.0:
            sp = engine.phase_species_amounts(phase_name)
            phase_volumes[phase_name] = float(
                sum(float(n) * species_mv[str(dc)] for dc, n in sp.items()))

    result = {
        "ok": True,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "pH": _nullable(engine.pH),
        "ionic_strength": _nullable(engine.IS),
        "phase_amounts_mol": phase_amounts,
        "phase_masses_kg": _mapping("phase_masses"),
        "phase_volumes_m3": phase_volumes,
        "phase_elements_mol": phase_elements,
        "phase_species_mol": phase_species_mol,
        "aqueous_h2o_mol": aqueous_h2o,
        "aqueous_species_mol": aqueous_species,
        "element_input": element_input,
        "xgems_version": version,
    }
    json.dumps(result, allow_nan=False)  # protocol carries null, never bare NaN
    return result


def _worker_main(request_path: str, response_path: str) -> int:
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        response = _worker_execute(request)
    except Exception as exc:
        import traceback
        if isinstance(exc, RuntimeError):
            kind = "nonconvergence"       # only raised by the equilibrate check
        elif isinstance(exc, (ValueError, KeyError, FileNotFoundError, TypeError)):
            kind = "config"
        else:
            kind = "internal"
        response = {"ok": False, "error_kind": kind,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
    Path(response_path).write_text(json.dumps(response), encoding="utf-8")
    return 0 if response.get("ok") else 1


def _serve_main(serve_dir: str) -> int:
    """Persistent request loop. The loaded ChemicalEngine is CACHED per bundle
    and fully reset per request (clear/T/P/cold_start/suppress) — measured
    bitwise-identical to a fresh construction and verified by the
    persistent-equals-spawn engine test, while skipping the dominant
    bundle-parse cost."""
    import time
    root = Path(serve_dir)
    engine_cache: Dict[str, object] = {}
    k = 0
    while True:
        if (root / "stop").is_file():
            return 0
        req_ready = root / f"req_{k:06d}.ready"
        if not req_ready.is_file():
            time.sleep(0.005)
            continue
        req = root / f"req_{k:06d}.json"
        resp = root / f"resp_{k:06d}.json"
        try:
            request = json.loads(req.read_text(encoding="utf-8"))
            response = _worker_execute(request, engine_cache=engine_cache)
        except Exception as exc:
            import traceback
            if isinstance(exc, RuntimeError):
                kind = "nonconvergence"
            elif isinstance(exc, (ValueError, KeyError, FileNotFoundError, TypeError)):
                kind = "config"
            else:
                kind = "internal"
            response = {"ok": False, "error_kind": kind,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}
        resp.write_text(json.dumps(response), encoding="utf-8")
        (root / f"resp_{k:06d}.ready").write_text("", encoding="utf-8")
        k += 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        sys.exit(_serve_main(sys.argv[2]))
    print("usage: python -m tinn.gems --worker <req.json> <resp.json> | "
          "--serve <dir>", file=sys.stderr)
    sys.exit(2)
