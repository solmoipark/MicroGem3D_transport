"""M5 tests: analysis + report (PRD §3 M5) — porosity, percolation, phase
fractions, slice PNGs, §6.3 judgment, cli report."""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from tinn import analysis, cli
from tinn.config import TinnConfig
from tinn.registry import HYDRATE_PHASE_IDS, default_registry
from tinn.storage import load_checkpoint

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
BUNDLE = REPO / "gems_bundles" / "PC" / "PC-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))
needs_gems = pytest.mark.skipif(
    not (BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or PC bundle not available")

REG = default_registry()


@pytest.fixture(scope="module")
def stoich_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("m5") / "run"
    cfg_path = tmp_path_factory.mktemp("m5cfg") / "cfg.json"
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 6.0], "alpha": {"C3S": [0.0, 0.08]}}}
    raw["schedule"] = {"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["run", str(cfg_path), "--out", str(out)]) == 0
    return out


# ---------------- policies / quantities ----------------

def test_gel_porosity_vector_policies():
    stoich = TinnConfig.from_json_file(str(EXAMPLES / "c3s_32.json"))
    eps = analysis.gel_porosity_vector(stoich, HYDRATE_PHASE_IDS, REG)
    assert eps[HYDRATE_PHASE_IDS.index("CSH")] == pytest.approx(0.28)
    assert eps[HYDRATE_PHASE_IDS.index("CH")] == 0.0
    gems = TinnConfig.from_json_file(str(EXAMPLES / "opc_gems_32.json"))
    eps2 = analysis.gel_porosity_vector(gems, ("CSHQ", "Portlandite"), REG)
    assert eps2.tolist() == [0.28, 0.0]


def test_state_row_matches_engine_summary(stoich_run):
    ck = sorted(stoich_run.glob("ckpt_*"))[-1]
    state = load_checkpoint(str(ck), REG)
    gel_eps = analysis.gel_porosity_vector(state.config, state.hydrate_ids, REG)
    row = analysis.state_row(state, REG, gel_eps)
    summary = json.loads((stoich_run / "summary.json").read_text(encoding="utf-8"))
    final = summary["final"]
    assert row["porosity_capillary"] == pytest.approx(final["porosity_capillary"])
    assert row["alpha"] == pytest.approx(final["alpha"])
    assert row["water_mol"] == pytest.approx(final["water_mol"])


# ---------------- percolation ----------------

def test_percolation_spanning_slab():
    liq = np.zeros((16, 16, 16))
    liq[:, 3, 3] = 1.0  # a 1-voxel column spanning z
    perc = analysis.liquid_percolation(liq)
    assert perc == {"z": True, "y": False, "x": False, "any": True}


def test_percolation_isolated_blob():
    liq = np.zeros((16, 16, 16))
    liq[5:8, 5:8, 5:8] = 1.0
    perc = analysis.liquid_percolation(liq)
    assert not perc["any"]


def test_percolation_full_liquid():
    perc = analysis.liquid_percolation(np.ones((8, 8, 8)))
    assert perc["z"] and perc["y"] and perc["x"] and perc["any"]


# ---------------- PNG ----------------

def test_write_png_valid_and_deterministic(tmp_path):
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    rgb[2, 3] = (255, 0, 0)
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    analysis.write_png(str(p1), rgb)
    analysis.write_png(str(p2), rgb)
    raw = p1.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", raw[16:24])
    assert (w, h) == (7, 5)
    assert raw == p2.read_bytes()


def test_central_slice_colors(stoich_run):
    state = load_checkpoint(str(sorted(stoich_run.glob("ckpt_*"))[0]), REG)
    img = analysis.central_slice_rgb(state)
    n = state.grid_size
    assert img.shape == (n, n, 3) and img.dtype == np.uint8
    # a fully liquid voxel renders as the pure liquid color
    z = n // 2
    liq_vox = np.argwhere(state.capillary_liquid[z] > 0.999)
    if len(liq_vox):
        y, x = liq_vox[0]
        assert img[y, x].tolist() == [40, 90, 220]


# ---------------- §6.3 judgment ----------------

def test_sanity_band_pass_and_warn():
    cfg = TinnConfig.from_json_file(str(EXAMPLES / "opc_srm114q_32.json"))
    row = {"time_h": 24.0, "alpha": {"C3S": 0.4, "C2S": 0.1, "C3A": 0.3, "C4AF": 0.2},
           "hydrate_mol": {"CH": 1.0}, "porosity_capillary": 0.5,
           "chem_shrinkage_ml_per_g_reacted": 0.05, "ledger_metrics": {}}
    band = analysis.sanity_band([row], cfg)
    by_name = {c["check"]: c["status"] for c in band["checks"]}
    assert by_name["total_clinker_alpha@24h"] == "pass"
    assert "cluster_ph_band_12.4_13.9" not in by_name  # no pH data -> no check
    bad = dict(row, alpha={"C3S": 0.05, "C2S": 0.2, "C3A": 0.0, "C4AF": 0.0})
    band2 = analysis.sanity_band([bad], cfg)
    by_name2 = {c["check"]: c["status"] for c in band2["checks"]}
    assert by_name2["total_clinker_alpha@24h"] == "warn"
    assert by_name2["alpha_order_C3S_ge_C2S"] == "warn"
    assert "not scientific validation" in band2["note"]


# ---------------- report (M5 DoD) ----------------

def test_report_from_run_dir(stoich_run, tmp_path):
    result = analysis.report(str(stoich_run), str(tmp_path))
    ckpts = sorted(stoich_run.glob("ckpt_*"))
    assert len(result["outputs"]) == len(ckpts)
    assert (tmp_path / "report.json").is_file()
    for row in result["outputs"]:
        assert row["ledger_violations"] == []          # §6.1 re-checked from disk
        assert "percolation" in row and "phase_volume_fractions" in row
        assert (tmp_path / row["slice_png"]).is_file()
    assert result["sanity_band"]["checks"]
    assert result["backend"] == "stoichiometric"


def test_report_is_read_only(stoich_run, tmp_path):
    ck = sorted(stoich_run.glob("ckpt_*"))[-1]
    before = load_checkpoint(str(ck), REG).full_hash()
    analysis.report(str(stoich_run), str(tmp_path))
    assert load_checkpoint(str(ck), REG).full_hash() == before


def test_cli_report(stoich_run, tmp_path, capsys):
    assert cli.main(["report", str(stoich_run), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "sanity band:" in out and "report written" in out
    assert cli.main(["report", str(tmp_path / "nope")]) == 2


@needs_gems
def test_report_on_gems_run_with_cluster_ph(tmp_path):
    # M5 DoD: report generated from M4-run artifacts, pH band included
    raw = json.loads((EXAMPLES / "opc_gems_32.json").read_text(encoding="utf-8"))
    raw["chemistry"]["gems_bundle_lst"] = str(BUNDLE)
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw["schedule"] = {"output_times_h": [2.0, 6.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.001}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "run"
    assert cli.main(["run", str(cfg_path), "--out", str(out)]) == 0
    result = analysis.report(str(out))
    names = {c["check"] for c in result["sanity_band"]["checks"]}
    assert "cluster_ph_band_12.4_13.9" in names   # merged from summary.json
    assert result["backend"] == "gems3k"
    for row in result["outputs"]:
        assert row["ledger_violations"] == []
        assert row["percolation"]["any"] in (True, False)
