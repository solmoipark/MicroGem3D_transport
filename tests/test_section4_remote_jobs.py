from __future__ import annotations

import importlib.util
from pathlib import Path

from tinn.config import TinnConfig


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sequential_schedule_boundaries_match_completed_accept_counts():
    module = _load_script("run_sequential_0d.py")
    cases = {
        "deschner_opc_q32_dt12_28d.json": 56,
        "deschner_opc_q32_fixed_dt05_log_outputs_28d.json": 1344,
        "deschner_opc_q32_log_l2_28d.json": 460,
    }
    for filename, expected in cases.items():
        cfg = TinnConfig.from_json_file(
            str(REPO / "examples" / "qualification" / filename))
        boundaries = module._schedule_boundaries(cfg, 672.0)
        assert len(boundaries) == expected
        assert boundaries[-1] == 672.0
        assert all(age in boundaries for age in module.AGES_H)


def test_prod64_config_contract():
    path = (REPO / "examples" / "qualification" / "spatial_sensitivity"
            / "prod64_v1_seed20260731_dt05_28d_20260806.json")
    cfg = TinnConfig.from_json_file(str(path))
    assert cfg.rve.grid_size == 64
    assert cfg.rve.voxel_size_um == 1.0
    assert cfg.rve.seed == 20260731
    assert cfg.schedule.dt_initial_h == 0.5
    assert cfg.schedule.dt_windows is None
    assert cfg.schedule.output_times_h == [
        8.0, 16.0, 24.0, 48.0, 72.0, 120.0, 168.0, 336.0, 504.0, 672.0]


def test_prod64_event8_dt12_config_contract():
    module = _load_script("run_sequential_0d.py")
    path = (REPO / "examples" / "qualification" / "spatial_sensitivity"
            / "prod64_v1_seed20260731_event8_dt12_28d_20260806.json")
    cfg = TinnConfig.from_json_file(str(path))
    assert cfg.rve.grid_size == 64
    assert cfg.rve.voxel_size_um == 1.0
    assert cfg.rve.seed == 20260731
    assert cfg.schedule.dt_initial_h == 8.0
    assert [(row.until_h, row.dt_h) for row in cfg.schedule.dt_windows] == [
        (24.0, 8.0), (672.0, 12.0)]
    assert cfg.schedule.output_times_h == [
        8.0, 16.0, 24.0, 48.0, 72.0, 120.0, 168.0, 336.0, 504.0, 672.0]
    boundaries = module._schedule_boundaries(cfg, 672.0)
    assert len(boundaries) == 57
    assert boundaries[:3] == [8.0, 16.0, 24.0]
    assert boundaries[-1] == 672.0
    assert all(age in boundaries for age in cfg.schedule.output_times_h)
