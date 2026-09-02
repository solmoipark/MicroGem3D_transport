# -*- coding: utf-8 -*-
"""External sulfate attack qualification (RVE scale, 2026-09-02): per-
checkpoint observables of a `sulfate_attack_opc32` run - the boundary
sulfur uptake, the surface-sorbed sulfur, and per-layer volumes of the
sulfur-bearing hydrates along the exposed axis (layer 0 = the bath face).

    py -3 scripts/analyze_sulfate_attack.py [runs/sulfate_attack_opc32]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.registry import ELEMENT_IDS, registry_for      # noqa: E402
from tinn.storage import load_checkpoint                  # noqa: E402
from tinn.config import TinnConfig                        # noqa: E402

S_PHASES = ("ettringite", "SO4_CO3_AFt", "CO3_SO4_AFt", "SO4_OH_AFm",
            "OH_SO4_AFm", "Gypsum", "gypsum", "C4AsH12", "C4AsH14")
TRACK = ("Portlandite", "CSHQ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else REPO / "runs" / "sulfate_attack_opc32")
    ckpts = sorted(run_dir.glob("ckpt_*"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    cfg = TinnConfig.model_validate(json.loads(
        (REPO / "examples" / "qualification" / "sulfate_attack_opc32.json")
        .read_text(encoding="utf-8")))
    reg = registry_for(cfg)
    axis = {"z": 0, "y": 1, "x": 2}[cfg.transport.boundary.axis]
    other = tuple(a for a in range(3) if a != axis)
    s_idx = ELEMENT_IDS.index("S")
    rows = []
    for ck in ckpts:
        st = load_checkpoint(str(ck), reg)
        hyd = list(st.hydrate_ids)
        prof = {}
        for name in S_PHASES + TRACK:
            if name in hyd:
                prof[name] = st.hydrate_fraction[hyd.index(name)].sum(
                    axis=other).tolist()
        sorbed_s = (float(st.domain_sorbed_mol[:, s_idx].sum())
                    if st.domain_sorbed_mol.size else 0.0)
        rows.append({
            "ckpt": ck.name,
            "time_h": float(st.time_h),
            "S_in_from_bath_mol": float(st.boundary_exchanged_elements[s_idx]),
            "S_sorbed_mol": sorbed_s,
            "S_aqueous_mol": float(st.cluster_inventory[:, s_idx].sum()),
            "layer_profiles_vox": prof,
            "layer_liquid_vox": st.capillary_liquid.sum(axis=other).tolist(),
        })
    out = run_dir / "sulfate_attack_analysis.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    start = cfg.transport.boundary.start_h or 0.0
    print(f"{'t_h':>8} {'t_exp_h':>8} {'S_in':>10} {'S_sorbed':>10} "
          f"{'S_aq':>10}  AFt(layers 0-3 | 12-15 | 28-31)")
    for r in rows:
        aft = None
        for name in ("ettringite", "SO4_CO3_AFt", "CO3_SO4_AFt"):
            if name in r["layer_profiles_vox"]:
                v = np.asarray(r["layer_profiles_vox"][name])
                aft = v if aft is None else aft + v
        seg = ("n/a" if aft is None else
               f"{aft[0:4].sum():7.2f} | {aft[12:16].sum():7.2f} | "
               f"{aft[28:32].sum():7.2f}")
        print(f"{r['time_h']:8.2f} {r['time_h']-start:8.3f} "
              f"{r['S_in_from_bath_mol']:10.3e} {r['S_sorbed_mol']:10.3e} "
              f"{r['S_aqueous_mol']:10.3e}  {seg}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
