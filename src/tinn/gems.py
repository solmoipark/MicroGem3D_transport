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
SUCCESS_STATUSES = (
    "No GEM re-calculation needed",
    "OK after GEM calculation with LPP AIA",
    "OK after GEM calculation with SIA",
)
CHARGE_ELEMENT_ID = "Zz"
BULK_VERIFY_ATOL_MOL = 1e-12
BULK_VERIFY_RTOL = 1e-10
O2_SEED_MOL_O = 1e-7  # tiny O2 seed so the aqueous redox state is well-posed


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
            h = hashlib.sha256()
            h.update(f.read_bytes())
            digests[f.relative_to(root.parent).as_posix()] = h.hexdigest()
    return digests


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
    element_input: Dict[str, object] = field(default_factory=dict)
    xgems_version: str = "not_available"

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
                 work_root: Optional[str] = None, timeout_s: float = 300.0):
        self.bundle_lst = str(Path(bundle_lst).resolve())
        self.python_executable = str(python_executable or sys.executable)
        if not Path(self.python_executable).is_file():
            raise GemsError(
                f"worker interpreter not found: {self.python_executable}", kind="config")
        self.work_root = (Path(work_root) if work_root
                          else Path.cwd() / ".gems_runs").resolve()
        self.timeout_s = timeout_s
        self.baseline_audit = audit_bundle(self.bundle_lst)

    # ---------------------------------------------------------------- public
    def equilibrate_stored(self) -> GemsResult:
        """Re-equilibrate the bundle's stored DBR node (PRD §6.2 anchor 1)."""
        return self._call({"mode": "stored", "dat_lst": self.bundle_lst})

    def equilibrate_elements(self, element_mol: Mapping[str, float],
                             temperature_k: float, pressure_pa: float = 1e5,
                             suppressed_phases: Sequence[str] = SUPPRESSED_CLINKER_PHASES,
                             ) -> GemsResult:
        """Cold-start equilibration of an authoritative element inventory with
        clinker phases suppressed (bound=0)."""
        for el, v in element_mol.items():
            if el == CHARGE_ELEMENT_ID:
                if v != 0.0:
                    raise GemsError(
                        "charge (Zz) targets are not supported — released "
                        "inventories are charge-neutral by construction", kind="config")
                continue
            if not math.isfinite(v) or v < 0.0:
                raise GemsError(f"element amount {el}={v} must be finite and >= 0",
                                kind="config")
        return self._call({
            "mode": "elements",
            "dat_lst": self.bundle_lst,
            "element_mol": {str(k): float(v) for k, v in element_mol.items()},
            "temperature_k": float(temperature_k),
            "pressure_pa": float(pressure_pa),
            "suppressed_phases": list(suppressed_phases),
        })

    # --------------------------------------------------------------- private
    def _call(self, request: Dict) -> GemsResult:
        before = audit_bundle(self.bundle_lst)
        if before != self.baseline_audit:
            changed = sorted(k for k in set(before) | set(self.baseline_audit)
                             if before.get(k) != self.baseline_audit.get(k))
            raise BundleAuditError(
                f"bundle changed since worker construction: {changed}")
        run_dir = self.work_root / f"run-{uuid.uuid4().hex[:12]}"
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
            raise GemsError(
                f"xGEMS worker failed [{kind}]: {response.get('error')}\n"
                f"{response.get('traceback', '')}\n"
                f"artifacts kept in {run_dir}", kind=kind)
        # solver artifacts are only discarded on success
        shutil.rmtree(run_dir, ignore_errors=True)
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
            element_input=response.get("element_input", {}),
            xgems_version=response.get("xgems_version", "not_available"),
        )


# ------------------------------------------------------------------ 0D probe

def run_0d_probe(config, worker: GemsWorker,
                 times_h: Optional[Sequence[float]] = None) -> List[Dict]:
    """Cumulative 0D chemistry probe: at each time, equilibrate exactly what the
    kinetics released so far (per 1 g binder) plus total mix water and O2 seed.
    Returns one row per time with pH, ionic strength, phase masses, and the
    element-closure error. Unreacted clinker is never an input."""
    from .kinetics import make_kinetics
    from .registry import KINETIC_PHASE_IDS, default_registry

    reg = default_registry()
    kin = make_kinetics(config)
    times = list(times_h if times_h is not None else config.schedule.output_times_h)
    n0 = {p: config.binder.mass_fractions.get(p, 0.0) / reg.get(p).molar_mass_g_mol
          for p in KINETIC_PHASE_IDS}  # mol per g binder
    water_mol = config.w_c / reg.get("H2O").molar_mass_g_mol

    rows: List[Dict] = []
    for t in times:
        alpha = kin.alpha_at(t)
        elements: Dict[str, float] = {"O": O2_SEED_MOL_O}
        for k, p in enumerate(KINETIC_PHASE_IDS):
            released = n0[p] * float(alpha[k])
            if released <= 0.0:
                continue
            for el, count in reg.get(p).formula.items():
                elements[el] = elements.get(el, 0.0) + released * count
        for el, count in reg.get("H2O").formula.items():
            elements[el] = elements.get(el, 0.0) + water_mol * count

        result = worker.equilibrate_elements(elements, config.temperature_K)
        rows.append({
            "time_h": t,
            "alpha": {p: float(alpha[k]) for k, p in enumerate(KINETIC_PHASE_IDS)},
            "ph": result.ph,
            "ph_status": result.ph_status,
            "ionic_strength": result.ionic_strength,
            "phase_masses_kg": result.phase_masses_kg,
            # solver self-consistency vs the effective (floor-clamped) input
            "element_closure_max_rel": result.element_closure_max_rel(),
            # divergence of the effective input from the physical ledger
            "floor_adjust_max_rel": result.floor_adjust_max_rel(),
            "status": result.status,
        })
    return rows


# ------------------------------------------------------------ worker process

def _worker_execute(request: Mapping) -> Dict:
    """Runs inside the isolated worker process — the only place xgems is imported."""
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

    if mode == "stored":
        engine = xgems.ChemicalEngineDicts(str(dat))
    elif mode == "elements":
        engine = xgems.ChemicalEngineDicts(str(dat), reset_calc=True, cold_start=True)
        engine.clear()
        engine.T = float(request["temperature_k"])
        engine.P = float(request["pressure_pa"])
        engine.cold_start()
        suppressed = [str(s) for s in request["suppressed_phases"]]
        missing = sorted(set(suppressed) - {str(p) for p in engine.phase_names})
        if missing:
            raise ValueError(f"bundle lacks phases requested for suppression: {missing}")
        engine.suppress_multiple_phases(suppressed)

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
    result = {
        "ok": True,
        "status": status,
        "pH": _nullable(engine.pH),
        "ionic_strength": _nullable(engine.IS),
        "phase_amounts_mol": _mapping("phase_amounts"),
        "phase_masses_kg": _mapping("phase_masses"),
        "phase_volumes_m3": _mapping("phase_volumes"),
        "phase_elements_mol": phase_elements,
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


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2], sys.argv[3]))
    print("usage: python -m tinn.gems --worker <request.json> <response.json>",
          file=sys.stderr)
    sys.exit(2)
