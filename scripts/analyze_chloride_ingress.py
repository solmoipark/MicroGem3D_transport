# -*- coding: utf-8 -*-
"""Chloride ingress qualification (RVE scale, RT-Cl-2): per-checkpoint
observables of a `chloride_ingress_opc32` run - the boundary chloride
uptake, the surface-sorbed chloride (Surf_c pool), the free chloride,
and per-layer volumes of the chloride AFm phases (Friedel's / Kuzel's
salt) along the exposed axis (layer 0 = the bath face).

    py -3 scripts/analyze_chloride_ingress.py [runs/chloride_ingress_opc32]
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

TRACK = ("Portlandite", "CSHQ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("cfg=")]
    run_dir = Path(args[0] if args else REPO / "runs" / "chloride_ingress_opc32")
    cfg_path = next((a[4:] for a in sys.argv[1:] if a.startswith("cfg=")),
                    str(REPO / "examples" / "qualification"
                        / "chloride_ingress_opc32.json"))
    ckpts = sorted(run_dir.glob("ckpt_*"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}")
    cfg = TinnConfig.model_validate(json.loads(
        Path(cfg_path).read_text(encoding="utf-8")))
    reg = registry_for(cfg)
    axis = {"z": 0, "y": 1, "x": 2}[cfg.transport.boundary.axis]
    other = tuple(a for a in range(3) if a != axis)
    cl_idx = ELEMENT_IDS.index("Cl")
    rows = []
    for ck in ckpts:
        st = load_checkpoint(str(ck), reg)
        hyd = list(st.hydrate_ids)
        cl_phases = tuple(h for h in hyd
                          if "Friedel" in h or "Kuzel" in h or "Cl" in h)
        prof = {}
        for name in cl_phases + TRACK:
            if name in hyd:
                prof[name] = st.hydrate_fraction[hyd.index(name)].sum(
                    axis=other).tolist()
        sorbed = (float(st.domain_sorbed_mol[:, cl_idx].sum())
                  if st.domain_sorbed_mol.size else 0.0)
        rows.append({
            "ckpt": ck.name,
            "time_h": float(st.time_h),
            "Cl_in_from_bath_mol": float(
                st.boundary_exchanged_elements[cl_idx]),
            "Cl_sorbed_mol": sorbed,
            "Cl_aqueous_mol": float(st.cluster_inventory[:, cl_idx].sum()),
            "cl_phases": list(cl_phases),
            "layer_profiles_vox": prof,
            "layer_liquid_vox": st.capillary_liquid.sum(axis=other).tolist(),
        })
    out = run_dir / "chloride_ingress_analysis.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    start = cfg.transport.boundary.start_h or 0.0
    print(f"{'t_h':>8} {'t_exp_h':>8} {'Cl_in':>10} {'Cl_sorbed':>10} "
          f"{'Cl_aq':>10}  Cl-AFm(layers 0-3 | 12-15 | 28-31)")
    for r in rows:
        afm = None
        for name in r["cl_phases"]:
            if name in r["layer_profiles_vox"]:
                v = np.asarray(r["layer_profiles_vox"][name])
                afm = v if afm is None else afm + v
        seg = ("n/a" if afm is None else
               f"{afm[0:4].sum():7.3f} | {afm[12:16].sum():7.3f} | "
               f"{afm[28:32].sum():7.3f}")
        print(f"{r['time_h']:8.2f} {r['time_h']-start:8.3f} "
              f"{r['Cl_in_from_bath_mol']:10.3e} {r['Cl_sorbed_mol']:10.3e} "
              f"{r['Cl_aqueous_mol']:10.3e}  {seg}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
