# -*- coding: utf-8 -*-
"""Re-pin the PRD 6.2 restart anchors after a FORMAT_VERSION break.

Runs anchor A (examples/c3s_32.json -> ckpt_003) and anchor B
(examples/qualification/deschner_opc_q32_smoke_24h.json -> ckpt_001)
from scratch and prints their dense_hash / full_hash. The dense hash
must equal the version-invariant PRD authority (the physics did not
move); the full hash is the value to record for the new format.

    py -3 scripts/repin_anchors.py [--out runs/repin]
"""
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinn.config import TinnConfig                      # noqa: E402
from tinn.engine import Engine                          # noqa: E402
from tinn.registry import registry_for                  # noqa: E402
from tinn.storage import FORMAT_VERSION, load_checkpoint  # noqa: E402

ANCHORS = {
    "A": ("examples/c3s_32.json", 3,
          "1490c624"),           # PRD 6.2 dense-hash authority prefix
    "B": ("examples/qualification/deschner_opc_q32_smoke_24h.json", 1,
          "83b0c6d9"),   # v6 (E=12) dense authority, PRD 6.2 (RT-Cl-1)
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out_root = REPO / "runs" / "repin"
    if "--out" in sys.argv:
        out_root = REPO / sys.argv[sys.argv.index("--out") + 1]
    print(f"FORMAT_VERSION {FORMAT_VERSION}")
    for tag, (cfg_path, k, dense_prefix) in ANCHORS.items():
        cfg = TinnConfig.model_validate(json.loads(
            (REPO / cfg_path).read_text(encoding="utf-8")))
        out = out_root / tag
        if out.exists():
            shutil.rmtree(out)
        t0 = time.perf_counter()
        Engine(cfg).run(out_dir=str(out))
        st = load_checkpoint(str(out / f"ckpt_{k:03d}"), registry_for(cfg))
        dense, full = st.dense_hash(), st.full_hash()
        ok = dense.startswith(dense_prefix)
        print(f"anchor {tag}: {cfg_path} ckpt_{k:03d}  wall {time.perf_counter()-t0:.0f} s")
        print(f"  dense {dense}  {'== PRD authority' if ok else '!! DIFFERS from PRD authority ' + dense_prefix}")
        print(f"  full  {full}")


if __name__ == "__main__":
    main()
