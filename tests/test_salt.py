"""E3 tests (PRD 1.2/4.1 v3.0): sulfate and alkali input channels — soluble
salt carriers in the registry, first-order dissolution, own particle
populations, backend gating, oxide diagnostics, pre-E3 restart refusal."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tinn import analysis
from tinn.config import TinnConfig
from tinn.engine import Engine
from tinn.kinetics import SALT_TAU_H_PRESETS, make_kinetics, salt_alpha
from tinn.registry import (ATOMIC_MASS_G_MOL, KINETIC_PHASE_IDS, SALT_PHASE_IDS,
                           SOLID_PHASE_IDS, default_registry)
from tinn.storage import StorageError, load_checkpoint, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
CNASH_BUNDLE = REPO / "gems_bundles" / "CNASH" / "Test-dat.lst"
GEMS_PYTHON = Path(os.environ.get(
    "TINN_GEMS_PYTHON",
    r"C:\Users\solmo\miniforge3\envs\py313-xgems\python.exe"))
needs_gems = pytest.mark.skipif(
    not (CNASH_BUNDLE.is_file() and GEMS_PYTHON.is_file()),
    reason="xgems worker interpreter or CNASH bundle not available")

REG = default_registry()

# bundle DCH V0 values (m3/mol -> cm3/mol), identical in PC and CNASH; anhydrite
# is absent from both bundles and carries the crystallographic value instead
BUNDLE_V0_CM3 = {"gypsum": 74.69, "hemihydrate": 61.73,
                 "arcanite": 65.50, "thenardite": 53.33}


def _salt_raw(**over):
    raw = json.loads((EXAMPLES / "opc_gypsum_cnash_32.json").read_text(
        encoding="utf-8"))
    raw["chemistry"]["gems_worker_python"] = str(GEMS_PYTHON)
    raw.update(over)
    return raw


# ---------------- registry ----------------

def test_salt_entries_match_bundle_molar_volumes():
    """The carriers' molar volumes are the bundle's own DCH V0 values, so a
    salt that dissolves and re-precipitates through GEMS keeps ONE volume
    definition (PRD 1.2 v3.0)."""
    dch = json.loads((REPO / "gems_bundles" / "PC" / "PC-dch.json").read_text(
        encoding="utf-8", errors="replace"))[0]["dch"]
    v0 = {str(n): float(v) for n, v in zip(dch["DCNL"], dch["V0"])}
    dc_of = {"gypsum": "Gp", "hemihydrate": "hemihydrate",
             "arcanite": "K2SO4", "thenardite": "Na2SO4"}
    for pid, dc in dc_of.items():
        entry = REG.get(pid)
        assert entry.kind == "salt"
        assert entry.molar_volume_cm3 == pytest.approx(v0[dc] * 1e6, rel=1e-6)
        assert entry.molar_volume_cm3 == pytest.approx(BUNDLE_V0_CM3[pid])
    # anhydrite: absent from the bundle, crystallographic M/rho (rho = 2.963)
    assert "Anh" not in v0 and "anhydrite" not in v0
    anh = REG.get("anhydrite")
    assert anh.density_g_cm3 == pytest.approx(2.963, rel=2e-3)


def test_salt_formulas_are_the_mineral_stoichiometries():
    f = {p: REG.get(p).formula for p in SALT_PHASE_IDS}
    assert f["gypsum"] == {"Ca": 1, "S": 1, "O": 6, "H": 4}        # CaSO4.2H2O
    assert f["hemihydrate"] == {"Ca": 1, "S": 1, "O": 4.5, "H": 1}  # .0.5H2O
    assert f["anhydrite"] == {"Ca": 1, "S": 1, "O": 4}
    assert f["arcanite"] == {"K": 2, "S": 1, "O": 4}
    assert f["thenardite"] == {"Na": 2, "S": 1, "O": 4}
    # molar masses follow from the formulas (no separate declaration to drift)
    m_gp = (ATOMIC_MASS_G_MOL["Ca"] + ATOMIC_MASS_G_MOL["S"]
            + 6 * ATOMIC_MASS_G_MOL["O"] + 4 * ATOMIC_MASS_G_MOL["H"])
    assert REG.get("gypsum").molar_mass_g_mol == pytest.approx(m_gp)


def test_salt_channels_appended_not_inserted():
    """Pre-E3 ledger positions must keep their meaning: the carriers are
    APPENDED to the kinetic/solid channel lists."""
    assert KINETIC_PHASE_IDS[:8] == ("C3S", "C2S", "C3A", "C4AF", "slag",
                                     "fly_ash", "metakaolin", "silica_fume")
    assert KINETIC_PHASE_IDS[8:] == SALT_PHASE_IDS
    assert SOLID_PHASE_IDS[:len(KINETIC_PHASE_IDS)] == KINETIC_PHASE_IDS
    assert SOLID_PHASE_IDS[-1] == "inert"


# ---------------- kinetics ----------------

def test_first_order_salt_alpha_math_and_ordering():
    assert salt_alpha(0.0, 3.0) == 0.0
    assert salt_alpha(-1.0, 3.0) == 0.0
    assert salt_alpha(3.0, 3.0) == pytest.approx(1.0 - math.exp(-1.0))
    assert salt_alpha(1e9, 3.0) == pytest.approx(1.0)
    # only the ORDERING of the presets is literature-grounded (PRD 1.2)
    t = SALT_TAU_H_PRESETS
    assert t["hemihydrate"] < t["gypsum"] < t["anhydrite"]
    assert t["arcanite"] == t["thenardite"] < t["hemihydrate"]


def test_pk_kinetics_drives_salt_channels_and_masks_absent_ones():
    cfg = TinnConfig.model_validate(_salt_raw())
    kin = make_kinetics(cfg)
    i = {p: KINETIC_PHASE_IDS.index(p) for p in SALT_PHASE_IDS}
    a0 = kin.alpha_at(0.0)
    assert float(a0[i["gypsum"]]) == 0.0
    a3 = kin.alpha_at(3.0)
    assert float(a3[i["gypsum"]]) == pytest.approx(1.0 - math.exp(-1.0))
    assert float(a3[i["arcanite"]]) > 0.99          # tau = 0.1 h
    # phases with no recipe mass stay at zero (existing masking contract)
    assert float(a3[i["hemihydrate"]]) == 0.0
    assert float(a3[i["anhydrite"]]) == 0.0
    # clinker integration is untouched by the new channels
    assert 0.0 < float(a3[KINETIC_PHASE_IDS.index("C3S")]) < 1.0


def test_salt_tau_override_and_validation():
    raw = _salt_raw()
    raw["kinetics"] = dict(raw["kinetics"], salt_tau_h={"gypsum": 8.0})
    cfg = TinnConfig.model_validate(raw)
    kin = make_kinetics(cfg)
    assert kin.salt_tau_h["gypsum"] == 8.0
    assert kin.salt_tau_h["arcanite"] == SALT_TAU_H_PRESETS["arcanite"]
    a = kin.alpha_at(8.0)[KINETIC_PHASE_IDS.index("gypsum")]
    assert float(a) == pytest.approx(1.0 - math.exp(-1.0))
    for bad in ({"C3S": 2.0}, {"gypsum": 0.0}, {"gypsum": -1.0}):
        rawb = _salt_raw()
        rawb["kinetics"] = dict(rawb["kinetics"], salt_tau_h=bad)
        with pytest.raises(Exception):
            TinnConfig.model_validate(rawb)
    # tabulated runs state alpha(t) directly, so the knob does not apply
    rawt = _salt_raw()
    rawt["kinetics"] = {"kind": "tabulated", "salt_tau_h": {"gypsum": 3.0},
                        "table": {"times_h": [0.0, 4.0],
                                  "alpha": {"C3S": [0.0, 0.1]}}}
    with pytest.raises(Exception, match="only applies to pk"):
        TinnConfig.model_validate(rawt)


# ---------------- geometry / config ----------------

def test_salt_carriers_get_their_own_populations():
    """Interground but distinct solids: each carrier is its own population with
    pure composition, never blended into the clinker particles."""
    from tinn.geometry import initialize_rve
    rve = initialize_rve(TinnConfig.model_validate(_salt_raw()), REG)
    mats = rve.report["materials"]
    assert list(mats)[0] == "clinker"
    assert set(list(mats)[1:]) == {"gypsum", "arcanite", "thenardite"}
    # each carrier population hits its own volume target and — being a
    # single-phase population — its dense channel holds exactly that volume,
    # which is what "not blended into the clinker particles" means
    for pid in ("gypsum", "arcanite", "thenardite"):
        row = mats[pid]
        assert row["rel_error"] < 0.01, (pid, row)
        chan = float(rve.anhydrous_fraction[SOLID_PHASE_IDS.index(pid)].sum())
        assert chan == pytest.approx(row["volume_achieved_vox"], rel=1e-9)
    # and the volume targets follow the recipe's volume shares
    fracs = _salt_raw()["binder"]["mass_fractions"]
    v = {p: f / REG.get(p).density_g_cm3 for p, f in fracs.items()}
    v["inert"] = (1.0 - sum(fracs.values())) / 3.15
    v_tot = sum(v.values())
    tot_target = sum(r["volume_target_vox"] for r in mats.values())
    assert mats["gypsum"]["volume_target_vox"] / tot_target == pytest.approx(
        v["gypsum"] / v_tot, rel=1e-6)
    # the carriers are FINER than the clinker population only if a PSD says so;
    # with a shared PSD their specific surface still differs by density alone
    assert mats["gypsum"]["ssa_est_m2_kg"] > 0.0


def test_material_psd_and_shape_accept_salt_keys():
    fine = {"bins": [{"d_lo_um": 1.0, "d_hi_um": 4.0, "volume_fraction": 1.0}]}
    raw = _salt_raw(material_psd={"gypsum": fine})
    h_psd = TinnConfig.model_validate(raw).config_hash()
    base = TinnConfig.model_validate(_salt_raw()).config_hash()
    assert h_psd != base                       # a distinct population is hashed
    raw2 = _salt_raw(material_shape={"gypsum": {"aspects": [1.4, 1.0, 0.8]}})
    assert TinnConfig.model_validate(raw2).config_hash() not in (base, h_psd)
    # a carrier without recipe mass is still refused
    with pytest.raises(Exception, match="no hemihydrate mass"):
        TinnConfig.model_validate(_salt_raw(material_psd={"hemihydrate": fine}))


def test_stoichiometric_backend_refuses_salt_recipes():
    raw = _salt_raw()
    raw["chemistry"] = {"backend": "stoichiometric"}
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0],
                                 "alpha": {"C3S": [0.0, 0.05],
                                           "C2S": [0.0, 0.0], "C3A": [0.0, 0.0],
                                           "C4AF": [0.0, 0.0],
                                           "gypsum": [0.0, 1.0],
                                           "arcanite": [0.0, 1.0],
                                           "thenardite": [0.0, 1.0]}}}
    raw["schedule"] = {"output_times_h": [4.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    with pytest.raises(Exception, match="gems3k"):
        TinnConfig.model_validate(raw)


def test_salt_free_config_hash_unchanged():
    """A salt-free config must keep its pre-E3 hash: salt_tau_h pops when it is
    None, and the carriers add nothing to the payload."""
    legacy = TinnConfig.from_json_file(str(EXAMPLES / "opc_cnash_32.json"))
    payload = legacy.model_dump(mode="json")
    assert payload["kinetics"].get("salt_tau_h") is None
    assert "gypsum" not in payload["binder"]["mass_fractions"]
    # the embedded-config hash contract that checkpoints rely on still holds
    assert legacy.config_hash() == TinnConfig.model_validate(
        json.loads((EXAMPLES / "opc_cnash_32.json").read_text(
            encoding="utf-8"))).config_hash()


# ---------------- diagnostics ----------------

def test_binder_oxide_diagnostics_math():
    cfg = TinnConfig.model_validate(_salt_raw())
    ox = analysis.binder_oxide_diagnostics(cfg, REG)
    # SO3 from the three carriers; hand-computed from the registry formulas
    so3 = 0.0
    for pid, frac in cfg.binder.mass_fractions.items():
        e = REG.get(pid)
        so3 += frac / e.molar_mass_g_mol * (e.formula or {}).get("S", 0.0)
    assert ox["so3_pct"] == pytest.approx(so3 * 80.06 * 100.0, rel=1e-12)
    assert 2.0 < ox["so3_pct"] < 4.0            # a normal Type I cement
    assert ox["na2o_eq_pct"] == pytest.approx(
        ox["na2o_pct"] + 0.658 * ox["k2o_pct"], rel=1e-12)
    assert 0.2 < ox["na2o_eq_pct"] < 1.0
    # a salt-free OPC reports exactly zero (nothing invented)
    plain = TinnConfig.from_json_file(str(EXAMPLES / "opc_cnash_32.json"))
    assert analysis.binder_oxide_diagnostics(plain, REG) == {
        "so3_pct": 0.0, "na2o_pct": 0.0, "k2o_pct": 0.0, "na2o_eq_pct": 0.0}


# ---------------- storage: pre-E3 refusal ----------------

def test_pre_e3_checkpoint_refused_with_named_channels(tmp_path):
    """A checkpoint written before E3 has no channel for the carriers; the
    refusal must NAME them instead of printing an opaque list diff."""
    raw = json.loads((EXAMPLES / "c3s_32.json").read_text(encoding="utf-8"))
    raw["kinetics"] = {"kind": "tabulated",
                       "table": {"times_h": [0.0, 4.0], "alpha": {"C3S": [0.0, 0.05]}}}
    raw["schedule"] = {"output_times_h": [2.0], "dt_initial_h": 2.0,
                       "dt_min_h": 0.01}
    cfg = TinnConfig.model_validate(raw)
    state, _ = Engine(cfg).run()
    save_checkpoint(state, str(tmp_path), "now")
    hdr = tmp_path / "now" / "header.json"
    payload = json.loads(hdr.read_text(encoding="utf-8"))
    # rewrite the header as a pre-E3 build would have written it
    payload["kinetic_phase_ids"] = list(KINETIC_PHASE_IDS[:8])
    payload["solid_phase_ids"] = list(KINETIC_PHASE_IDS[:8]) + ["inert"]
    hdr.write_text(json.dumps(payload), encoding="utf-8")
    import hashlib
    man = tmp_path / "now" / "manifest.json"
    manifest = json.loads(man.read_text(encoding="utf-8"))
    manifest["header.json"] = hashlib.sha256(hdr.read_bytes()).hexdigest()
    man.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageError, match="gypsum"):
        load_checkpoint(str(tmp_path / "now"), REG)


# ---------------- GEMS path: the actual point of E3 ----------------

@needs_gems
def test_gems_run_forms_sulfate_phases_and_closes(tmp_path):
    """The E3 payoff: with a real sulfate channel the equilibrium assemblage
    forms AFt/AFm instead of numerical dust, alkalis reach the pore solution,
    and every blocking ledger still closes."""
    raw = _salt_raw()
    raw["schedule"] = {"output_times_h": [6.0, 12.0], "dt_initial_h": 1.0,
                       "dt_min_h": 0.001}
    cfg = TinnConfig.model_validate(raw)
    eng = Engine(cfg)
    state, summary = eng.run(out_dir=str(tmp_path / "run"))
    from tinn import ledger
    rep = ledger.check_all(state, REG)
    assert rep.ok, rep.violations
    # sulfate actually entered the system as sulfate, not as inert filler
    from tinn.registry import ELEMENT_IDS
    s_col = ELEMENT_IDS.index("S")
    solid_s = float(state.hydrate_elements_ch[:, s_col].sum())
    liquid_s = float(state.cluster_inventory[:, s_col].sum()) \
        if state.cluster_inventory.size else 0.0
    assert solid_s + liquid_s > 0.0
    # an AFt/AFm-type host holds a MATERIAL SHARE of the system's sulfate —
    # pre-E3 the sulfate phases sat at ~1e-23 mol, i.e. numerical dust that no
    # relative test could pass (the gap this channel closes)
    hosts = {h: float(state.hydrate_elements_ch[i, s_col])
             for i, h in enumerate(state.hydrate_ids)
             if state.hydrate_elements_ch[i, s_col] > 0.0}
    assert hosts, "no solid phase hosts sulfate"
    assert max(hosts.values()) > 0.1 * (solid_s + liquid_s), hosts
    # and it is an ettringite/AFm-type host, not just re-precipitated gypsum
    assert any(("ettring" in h.lower() or h.startswith("C4A")
                or h.startswith("C6A")) for h in hosts), hosts
    # alkalis are in the pore solution (the pH comparison E3b unlocks)
    na = ELEMENT_IDS.index("Na")
    k = ELEMENT_IDS.index("K")
    if state.cluster_inventory.size:
        assert float(state.cluster_inventory[:, [na, k]].sum()) > 0.0
    ph = [row for row in summary["outputs"]
          if row.get("ledger_metrics", {}).get("cluster_ph")]
    assert ph, "no cluster pH recorded"
