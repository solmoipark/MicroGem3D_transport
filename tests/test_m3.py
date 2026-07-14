"""M3 tests: xGEMS isolated worker, bundle audit, 0D cumulative probe (PRD §3 M3).

Skipped entirely when the xgems worker interpreter or the PC bundle is absent —
every other tinn feature must work without them (PRD §5).
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinn.config import TinnConfig
from tinn.gems import (BundleAuditError, GemsError, GemsWorker, audit_bundle,
                       run_0d_probe)

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))

pytestmark = pytest.mark.skipif(
    not (BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    return GemsWorker(str(BUNDLE), python_executable=str(GEMS_PYTHON),
                      work_root=str(tmp_path_factory.mktemp("gems_work")))


def _anchor2_elements():
    n_c3s = 1.0 / 228.3145   # 1.0 g C3S
    n_w = 0.5 / 18.015       # 0.5 g H2O
    return {"Ca": 3 * n_c3s, "Si": n_c3s,
            "O": 5 * n_c3s + n_w + 1e-7, "H": 2 * n_w}


# ---------------- anchors (PRD §6.2, regression snapshots) ----------------

def test_anchor1_stored_dbr_reequilibration(worker):
    r = worker.equilibrate_stored()
    assert r.ph == pytest.approx(13.596951143910216, rel=1e-9)
    assert r.ionic_strength == pytest.approx(0.3662192687232276, rel=1e-9)
    assert r.status in ("OK after GEM calculation with LPP AIA",
                        "OK after GEM calculation with SIA",
                        "No GEM re-calculation needed")


def test_anchor2_c3s_water_o2_seed(worker):
    r = worker.equilibrate_elements(_anchor2_elements(), 293.15)
    assert r.ph == pytest.approx(12.6619326886856, rel=1e-6)
    solids = {k: v for k, v in r.phase_masses_kg.items()
              if k != "aq_gen" and v > 1e-7}
    assert set(solids) == {"CSHQ", "Portlandite"}
    assert r.element_closure_max_rel() <= 1e-12


def test_worker_determinism(worker):
    a = worker.equilibrate_elements(_anchor2_elements(), 293.15)
    b = worker.equilibrate_elements(_anchor2_elements(), 293.15)
    assert repr(a.ph) == repr(b.ph)
    assert a.phase_masses_kg == b.phase_masses_kg


def test_element_floors_are_reported_not_hidden(worker):
    # elements we did not request (Al, Fe, ...) clamp to the cleared-state
    # numerical floor; that adjustment must be reported, never hidden
    r = worker.equilibrate_elements(_anchor2_elements(), 293.15)
    floors = r.element_input["floor_adjustments_mol"]
    assert floors, "clamping to the cleared-state floor must be reported"
    eff = r.element_input["effective_element_mol"]
    req = r.element_input["requested_element_mol"]
    for el in floors:
        assert eff[el] >= req.get(el, 0.0)
    # requested elements were applied essentially as requested
    assert eff["Ca"] == pytest.approx(req["Ca"], rel=1e-9)


# ---------------- audit / isolation / errors ----------------

def test_bundle_audit_detects_mutation(tmp_path):
    bundle_copy = tmp_path / "PC"
    shutil.copytree(BUNDLE.parent, bundle_copy)
    w = GemsWorker(str(bundle_copy / "PC-dat.lst"),
                   python_executable=str(GEMS_PYTHON),
                   work_root=str(tmp_path / "work"))
    with open(bundle_copy / "PC-dch.json", "ab") as f:
        f.write(b" ")
    with pytest.raises(BundleAuditError):
        w.equilibrate_stored()


def test_audit_missing_bundle():
    with pytest.raises(GemsError):
        audit_bundle(str(BUNDLE.parent / "does-not-exist.lst"))


def test_source_bundle_never_polluted(worker):
    before = audit_bundle(str(BUNDLE))
    worker.equilibrate_elements(_anchor2_elements(), 293.15)
    assert audit_bundle(str(BUNDLE)) == before  # no ipmlog.txt/xGEMS.log etc.


def test_missing_suppression_phase_is_hard_error(worker):
    with pytest.raises(GemsError, match="suppression"):
        worker.equilibrate_elements(_anchor2_elements(), 293.15,
                                    suppressed_phases=("NotAPhase",))


def test_negative_element_rejected_client_side(worker):
    with pytest.raises(GemsError):
        worker.equilibrate_elements({"Ca": -1.0}, 293.15)


# ---------------- 0D cumulative probe (M3 DoD) ----------------

def test_0d_probe_c3s_closure_and_sanity(worker):
    cfg = TinnConfig.from_json_file(str(REPO / "examples" / "c3s_32.json"))
    rows = run_0d_probe(cfg, worker)
    assert [r["time_h"] for r in rows] == cfg.schedule.output_times_h
    ch_masses = []
    for r in rows:
        assert r["element_closure_max_rel"] <= 1e-12   # element + water closure
        assert 12.4 <= r["ph"] <= 13.9                 # PRD §6.3 cluster pH band
        ch_masses.append(r["phase_masses_kg"].get("Portlandite", 0.0))
    assert all(b > a for a, b in zip(ch_masses, ch_masses[1:]))  # CH monotone


def test_0d_probe_opc_four_phase(worker):
    cfg = TinnConfig.from_json_file(str(REPO / "examples" / "opc_srm114q_32.json"))
    rows = run_0d_probe(cfg, worker, times_h=[24.0])
    r = rows[0]
    assert r["element_closure_max_rel"] <= 1e-10
    assert r["alpha"]["C4AF"] > 0.0  # all four phases released something
    assert r["phase_masses_kg"].get("CSHQ", 0.0) > 0.0


# ---------------- config / scope ----------------

def test_config_gems3k_requires_bundle():
    raw = json.loads((REPO / "examples" / "c3s_32.json").read_text(encoding="utf-8"))
    raw["chemistry"] = {"backend": "gems3k"}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw)
    raw["chemistry"] = {"backend": "stoichiometric",
                        "gems_bundle_lst": str(BUNDLE)}
    with pytest.raises(ValidationError):
        TinnConfig.model_validate(raw)


def test_engine_gems3k_still_pending_m4():
    from tinn.engine import Engine
    raw = json.loads((REPO / "examples" / "c3s_32.json").read_text(encoding="utf-8"))
    raw["chemistry"] = {"backend": "gems3k", "gems_bundle_lst": str(BUNDLE)}
    cfg = TinnConfig.model_validate(raw)
    with pytest.raises(NotImplementedError):
        Engine(cfg)
