"""Vendor the per-species aqueous self-diffusion table for Tier 0 (RT-P0a).

Parses the SOLUTION_SPECIES block of a PHREEQC database (default: the
phreeqc.dat shipped with the installed phreeqpython package) and writes
gems_bundles/species_dw/species_dw.json with one row per species that
declares a `-dw` value (25 C tracer/self-diffusion coefficient, m^2/s).

The `aliases` field maps GEMS DC names (as they appear in the bundle DCH,
e.g. "SiO2@") to phreeqc.dat species names ("H4SiO4"). It is hand-curated
data: an existing aliases block in the output file is preserved verbatim
across regenerations; the DEFAULT_ALIASES below only seed a fresh file.
Only clear chemical identities are aliased — GEMS complexes with no
phreeqc analog stay unmapped on purpose and fall to the config's explicit
`default_dw_m2_s` (or a hard error when that is null; PRD 4.6.4, no
silent defaults).

Deterministic output: sorted keys, no timestamps. Run:

    py -3 scripts/make_species_dw_table.py
    py -3 scripts/make_species_dw_table.py --dat path/to/phreeqc.dat
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "gems_bundles" / "species_dw"
OUT_JSON = OUT_DIR / "species_dw.json"

# GEMS DC name -> phreeqc.dat species name. Identities only (same solute,
# different naming convention); approximations are NOT aliased.
DEFAULT_ALIASES = {
    "CO2@": "CO2",
    "CH4@": "CH4",
    "H2@": "H2",
    "O2@": "O2",
    "H2S@": "H2S",
    "SiO2@": "H4SiO4",          # neutral aqueous silica
    "Ca(HCO3)+": "CaHCO3+",
    "Ca(CO3)@": "CaCO3",
    "Ca(SO4)@": "CaSO4",
    "Mg(HCO3)+": "MgHCO3+",
    "Mg(CO3)@": "MgCO3",
    "MgSO4@": "MgSO4",
    "Na(CO3)-": "NaCO3-",
    "Na(HCO3)@": "NaHCO3",
    "Na(SO4)-": "NaSO4-",
    "K(SO4)-": "KSO4-",
}

_KEYWORD = re.compile(r"^[A-Z][A-Z0-9_]+$")          # e.g. PHASES
_CHARGE = re.compile(r"([+-])(\d*)$")


def species_charge(name: str) -> int:
    m = _CHARGE.search(name)
    if not m:
        return 0
    mag = int(m.group(2)) if m.group(2) else 1
    return mag if m.group(1) == "+" else -mag


def default_dat_path() -> Path:
    import phreeqpython
    return (Path(phreeqpython.__file__).parent / "database" / "phreeqc.dat")


def parse_dw(dat_path: Path):
    """[(species, z, dw_m2_s, source_line)] from the SOLUTION_SPECIES block."""
    rows = []
    in_block = False
    current = None
    for lineno, raw in enumerate(
            dat_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        bare = stripped.split("#", 1)[0].strip()
        if _KEYWORD.match(bare):
            in_block = bare == "SOLUTION_SPECIES"
            current = None
            continue
        if not in_block or not bare:
            continue
        if "=" in bare and not raw[:1].isspace() and not bare.startswith("-"):
            # reaction line: the defined (product) species is the RHS head
            current = bare.split("=", 1)[1].strip().split()[0]
            continue
        if bare.startswith("-dw") and current is not None:
            parts = bare.split()
            if len(parts) < 2:
                raise ValueError(f"malformed -dw line {lineno}: {raw!r}")
            dw = float(parts[1])
            if not (0.0 < dw < 1e-7):
                raise ValueError(
                    f"implausible dw {dw!r} for {current} (line {lineno})")
            rows.append((current, species_charge(current), dw, lineno))
    if not rows:
        raise ValueError(f"no -dw entries found in {dat_path}")
    # a species may appear once only; phreeqc.dat repeats none, keep it that way
    seen = {}
    for name, z, dw, lineno in rows:
        if name in seen and seen[name] != dw:
            raise ValueError(f"species {name} has two -dw values")
        seen[name] = dw
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dat", default=None,
                    help="source PHREEQC database (default: phreeqpython's "
                         "bundled phreeqc.dat)")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    dat = Path(args.dat) if args.dat else default_dat_path()
    out = Path(args.out)

    aliases = dict(DEFAULT_ALIASES)
    if out.is_file():   # hand-curated aliases survive regeneration
        prior = json.loads(out.read_text(encoding="utf-8"))
        if prior.get("aliases"):
            aliases = dict(prior["aliases"])

    rows = parse_dw(dat)
    names = {name for name, _, _, _ in rows}
    dangling = sorted(tgt for tgt in set(aliases.values()) if tgt not in names)
    if dangling:
        raise SystemExit(f"aliases point at species without -dw: {dangling}")

    try:
        import phreeqpython
        pkg_version = getattr(phreeqpython, "__version__", "unknown")
    except ImportError:
        pkg_version = "not_installed"
    payload = {
        "source": f"{dat.name} (phreeqpython {pkg_version})",
        "source_path": str(dat),
        "source_sha256": hashlib.sha256(dat.read_bytes()).hexdigest(),
        "temperature_C": 25.0,
        "species": [
            {"species": name, "z": z, "dw_25C_m2_s": dw, "source_line": lineno}
            for name, z, dw, lineno in sorted(rows)],
        "aliases": {k: aliases[k] for k in sorted(aliases)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(rows)} species, {len(aliases)} aliases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
