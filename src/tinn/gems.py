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
    pass


class BundleAuditError(GemsError):
    pass


def audit_bundle(dat_lst_path: str) -> Dict[str, str]:
    """sha256 of every file in the bundle directory (the .lst and all sidecars)."""
    root = Path(dat_lst_path).resolve()
    if not root.is_file():
        raise GemsError(f"bundle .lst not found: {root}")
    digests: Dict[str, str] = {}
    for f in sorted(root.parent.iterdir()):
        if f.is_file():
            h = hashlib.sha256()
            h.update(f.read_bytes())
            digests[f.name] = h.hexdigest()
    return digests


@dataclass
class GemsResult:
    status: str
    ph: float
    ionic_strength: float
    phase_amounts_mol: Dict[str, float]
    phase_masses_kg: Dict[str, float]
    phase_volumes_m3: Dict[str, float]
    phase_elements_mol: Dict[str, Dict[str, float]]
    element_input: Dict[str, object] = field(default_factory=dict)
    xgems_version: str = "not_available"

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
        self.python_executable = python_executable or sys.executable
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
            if el != CHARGE_ELEMENT_ID and (not math.isfinite(v) or v < 0.0):
                raise GemsError(f"element amount {el}={v} must be finite and >= 0")
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
            raise BundleAuditError("bundle changed since worker construction")
        run_dir = self.work_root / f"run-{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        req_path = run_dir / "request.json"
        resp_path = run_dir / "response.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")
        src_root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                [self.python_executable, "-m", "tinn.gems", "--worker",
                 str(req_path), str(resp_path)],
                cwd=run_dir, env=env, capture_output=True, text=True,
                timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise GemsError(f"xGEMS worker timed out after {self.timeout_s}s") from e
        after = audit_bundle(self.bundle_lst)
        if after != before:
            changed = sorted(k for k in set(before) | set(after)
                             if before.get(k) != after.get(k))
            raise BundleAuditError(f"xGEMS call mutated the source bundle: {changed}")
        if not resp_path.is_file():
            raise GemsError(
                f"worker produced no response (exit {proc.returncode}):\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
        response = json.loads(resp_path.read_text(encoding="utf-8"))
        shutil.rmtree(run_dir, ignore_errors=True)
        if not response.get("ok"):
            raise GemsError(f"xGEMS worker failed: {response.get('error')}\n"
                            f"{response.get('traceback', '')}")
        return GemsResult(
            status=response["status"],
            ph=response["pH"],
            ionic_strength=response["ionic_strength"],
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
            "ionic_strength": result.ionic_strength,
            "phase_masses_kg": result.phase_masses_kg,
            "element_closure_max_rel": result.element_closure_max_rel(),
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

    phase_elements = {str(ph): {str(e): float(v) for e, v in row.items()}
                      for ph, row in engine.phases_elements_moles.items()}
    result = {
        "ok": True,
        "status": status,
        "pH": float(engine.pH),
        "ionic_strength": float(engine.IS),
        "phase_amounts_mol": _mapping("phase_amounts"),
        "phase_masses_kg": _mapping("phase_masses"),
        "phase_volumes_m3": _mapping("phase_volumes"),
        "phase_elements_mol": phase_elements,
        "element_input": element_input,
        "xgems_version": version,
    }
    json.dumps(result, allow_nan=False)  # refuse NaN leaking into the protocol
    return result


def _worker_main(request_path: str, response_path: str) -> int:
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        response = _worker_execute(request)
    except Exception as exc:
        import traceback
        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
    Path(response_path).write_text(json.dumps(response), encoding="utf-8")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2], sys.argv[3]))
    print("usage: python -m tinn.gems --worker <request.json> <response.json>",
          file=sys.stderr)
    sys.exit(2)
