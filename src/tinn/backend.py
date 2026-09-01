"""ReactionBackend protocol + StoichiometricBackend (synthetic multiphase chemistry).

A backend receives, per liquid cluster, the newly released mol per kinetic phase
plus the existing dissolved-solution inventory and available water. It never sees
unreacted clinker and knows no coordinates. Fields it cannot compute are NaN with
status "not_available", never 0.

Products travel as Parcels carrying their OWN element vector and skeleton volume
(cm^3) — variable-composition solution phases (GEMS CSHQ) are never reinterpreted
as fixed formulas (PRD §2.3 product-parcel ledger).

Retryable failures (nonconvergence, timeouts) raise BackendTransientError, which
the engine turns into a trial reject; any other exception is a hard error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np

from .config import ReactionRule
from .registry import (ELEMENT_IDS, HYDRATE_PHASE_IDS, Registry, element_vector)

STATUS_OK = "ok"
STATUS_INSUFFICIENT_WATER = "insufficient_water"
PH_NOT_AVAILABLE = "not_available"


class BackendTransientError(RuntimeError):
    """Retryable backend failure — the engine rejects the trial (dt halving)."""


@dataclass
class Parcel:
    phase_id: str
    mol: float
    elements: np.ndarray      # (E,) over ELEMENT_IDS, total element mol
    skel_vol_cm3: float       # solid skeleton volume of this parcel
    # per-endmember (DC) mols of this parcel (E1, PRD 2.3 rev.3). None means
    # the phase is single-endmember: the engine books {phase_id: mol}. A dict
    # must sum to `mol` (worker-verified endmember-sum) — never invented.
    endmember_mol: Optional[Dict[str, float]] = None


@dataclass
class ReactionResult:
    status: str
    parcels: List[Parcel] = field(default_factory=list)
    water_consumed_mol: float = 0.0
    residual_inventory: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))
    injected_elements: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))  # seeds/solver floors
    ph: float = math.nan
    ph_status: str = PH_NOT_AVAILABLE
    # Read-only chemistry diagnostics carried through by backends that expose
    # them.  The engine does not use these to advance state; they make the
    # one-domain adapter directly auditable against standalone equilibrium.
    ionic_strength: float = math.nan
    ionic_strength_status: str = PH_NOT_AVAILABLE
    aqueous_h2o_mol: float = math.nan
    aqueous_elements: np.ndarray = field(
        default_factory=lambda: np.zeros(len(ELEMENT_IDS)))
    # Tier 0 (RT-P0a, PRD 4.6.4): per-species mols of the aqueous phase
    # (solvent H2O@ included), unscaled. None for backends without
    # speciation (synthetic); the engine only reads it when species
    # transport is configured.
    aqueous_species_mol: Optional[Dict[str, float]] = None


class ReactionBackend(Protocol):
    backend_id: str
    hydrate_ids: tuple  # fixed channel order of every parcel this backend emits
    # E1 (PRD 2.3 rev.3): per-channel endmember (DC) names in fixed order, and
    # each endmember's element row — both derived from the backend's own
    # definition source (bundle DCH / registry), never hardcoded.
    hydrate_endmembers: Dict[str, tuple]
    endmember_elements: Dict[str, np.ndarray]
    # "incremental": parcels are NEW precipitates appended to holdings.
    # "snapshot": parcels are the cluster's ENTIRE new assemblage (absolute
    # replacement; water_consumed_mol may be negative on re-dissolution).
    mode: str

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray, solid_elements=None) -> ReactionResult:
        ...


class StoichiometricBackend:
    """Deterministic fixed-stoichiometry backend: every released mol precipitates
    immediately per the config's element-balanced rules; the dissolved-solution
    inventory passes through unchanged (nothing accumulates in solution)."""

    backend_id = "stoichiometric"
    hydrate_ids = HYDRATE_PHASE_IDS
    mode = "incremental"

    def __init__(self, rules: Dict[str, ReactionRule], registry: Registry):
        self._rules = rules
        self._registry = registry
        # every stoichiometric hydrate is single-endmember: the channel IS the
        # endmember, with its registry formula as the element row
        self.hydrate_endmembers = {h: (h,) for h in HYDRATE_PHASE_IDS}
        self.endmember_elements = {
            h: element_vector(registry.get(h).formula, 1.0)
            for h in HYDRATE_PHASE_IDS}

    def react(self, released_mol: Dict[str, float], water_available_mol: float,
              inventory: np.ndarray, solid_elements=None) -> ReactionResult:
        if solid_elements is not None:
            raise RuntimeError(
                "StoichiometricBackend is incremental-only and cannot "
                "re-equilibrate existing solids (no silent fallback)")
        water_need = 0.0
        totals: Dict[str, float] = {}
        for phase_id, n_mol in released_mol.items():
            if n_mol <= 0.0:
                continue
            rule = self._rules.get(phase_id)
            if rule is None:
                raise RuntimeError(
                    f"stoichiometric backend has no reaction rule for released "
                    f"phase {phase_id!r} (SCM glasses and the E3 soluble salt "
                    f"carriers require the gems3k backend)")
            water_need += n_mol * rule.water_mol
            for hid, coeff in rule.products.items():
                totals[hid] = totals.get(hid, 0.0) + n_mol * coeff
        if water_need > water_available_mol:
            return ReactionResult(status=STATUS_INSUFFICIENT_WATER,
                                  residual_inventory=inventory.copy())
        parcels = []
        for hid in HYDRATE_PHASE_IDS:
            if hid not in totals:
                continue
            entry = self._registry.get(hid)
            mol = totals[hid]
            parcels.append(Parcel(
                phase_id=hid, mol=mol,
                elements=element_vector(entry.formula, mol),
                skel_vol_cm3=mol * entry.skeleton_molar_volume_cm3))
        return ReactionResult(status=STATUS_OK, parcels=parcels,
                              water_consumed_mol=water_need,
                              residual_inventory=inventory.copy())


# ------------------------------------------------------- Tier 1 (RT-S1a)
# PHREEQC SURFACE sorption operator (spec 3). Lives in backend.py by the
# module-cap rule (16/16). Phase-assemblage authority stays with GEMS:
# the operator runs SOLUTION + SURFACE only, and cemdata18.dat carries no
# surface chemistry (measured) - the reactions, constants and site
# densities are config-owned.

# neutral reactant decomposition of a ledger element vector (ported from
# the G0 spike, scripts/spike_phreeqc_crosscheck.py - scripts are not
# importable modules): (reactant formula, element, element per formula,
# O per formula)
_SORB_OXIDES = (
    ("CaO", "Ca", 1.0, 1.0), ("SiO2", "Si", 1.0, 2.0),
    ("Al2O3", "Al", 2.0, 3.0), ("Fe2O3", "Fe", 2.0, 3.0),
    ("SO3", "S", 1.0, 3.0), ("Na2O", "Na", 2.0, 1.0),
    ("K2O", "K", 2.0, 1.0), ("MgO", "Mg", 1.0, 1.0), ("CO2", "C", 1.0, 2.0),
)
_H2O_G_MOL = 18.015


def _decompose_to_reactants(elements: Dict[str, float]) -> Dict[str, float]:
    """elements -> {neutral reactant: mol}; exact-closure audited."""
    reactants: Dict[str, float] = {}
    o_used = 0.0
    for formula, el, n_el, n_o in _SORB_OXIDES:
        amount = elements.get(el, 0.0)
        if amount <= 0.0:
            continue
        mol = amount / n_el
        reactants[formula] = mol
        o_used += mol * n_o
    h = elements.get("H", 0.0)
    if h < -1e-15:
        raise ValueError(f"negative H in sorption input: {h}")
    h2o = h / 2.0
    o_left = elements.get("O", 0.0) - o_used - h2o
    scale = max((abs(v) for v in elements.values()), default=0.0) or 1.0
    if o_left < -1e-9 * scale:
        raise ValueError(
            f"element vector is not oxide-decomposable: O deficit {o_left}")
    if h2o > 0.0:
        reactants["H2O"] = h2o
    if o_left > 1e-15 * scale:
        reactants["O2"] = o_left / 2.0
    rebuilt: Dict[str, float] = {el: 0.0 for el in elements}
    for formula, el, n_el, n_o in _SORB_OXIDES:
        mol = reactants.get(formula, 0.0)
        rebuilt[el] = rebuilt.get(el, 0.0) + mol * n_el
        rebuilt["O"] = rebuilt.get("O", 0.0) + mol * n_o
    rebuilt["H"] = rebuilt.get("H", 0.0) + 2.0 * reactants.get("H2O", 0.0)
    rebuilt["O"] = (rebuilt.get("O", 0.0) + reactants.get("H2O", 0.0)
                    + 2.0 * reactants.get("O2", 0.0))
    for el, target in elements.items():
        if abs(rebuilt.get(el, 0.0) - target) > 1e-9 * scale + 1e-15:
            raise AssertionError(
                f"reactant decomposition broke {el}: "
                f"{rebuilt.get(el)} != {target}")
    return reactants


@dataclass
class SorptionResult:
    status: str                       # "ok" (nonconvergence raises instead)
    sorbed_mol: np.ndarray            # (E,) equilibrium surface-bound elements
    site_occupancy: Dict[str, float]  # bound species -> mol (diagnostics)


class SorptionOperator:
    """Pure-function equilibrium sorption: (pore-solution elements incl.
    the previously sorbed re-offer, water, site total) -> the equilibrium
    surface-bound element vector. Snapshot semantics, like the GEMS
    backend: the engine books the DELTA against its sorbed store, so a
    shrinking site total desorbs automatically. Deterministic: one
    IPhreeqc instance, numbered blocks fully redefined per call, exact
    input-hash memoization (LRU), dat audited before/after every call."""

    operator_id = "phreeqc_surface"

    def __init__(self, config, temperature_k: float):
        from collections import OrderedDict
        if config.surface_model != "no_edl":
            raise RuntimeError(
                "surface_model 'ddl' is declared but not implemented yet - "
                "the electrostatic model needs real area/mass parameters "
                "(no silent placeholder physics)")
        self._cfg = config
        self.temperature_k = float(temperature_k)
        self._dat = str(Path(config.phreeqc_dat).resolve())
        from .gems import audit_bundle
        self._audit = audit_bundle
        self.baseline_audit = audit_bundle(self._dat)
        self._pp = None
        self._memo: "OrderedDict[str, SorptionResult]" = OrderedDict()
        self.memo_cap = 4096
        # per-reaction element rows over ELEMENT_IDS, config-declared
        self._bound_species = [rx.bound_species
                               for rx in config.surface_species]
        if len(set(self._bound_species)) != len(self._bound_species):
            raise RuntimeError("surface reactions bind duplicate species")
        self._rows = np.zeros((len(config.surface_species), len(ELEMENT_IDS)))
        for i, rx in enumerate(config.surface_species):
            for el, v in rx.sorbed_elements.items():
                self._rows[i, ELEMENT_IDS.index(el)] = float(v)
        self._whitelist = tuple(config.elements)

    # ------------------------------------------------------------- private
    def _instance(self):
        if self._pp is None:
            from phreeqpython import PhreeqPython
            dat = Path(self._dat)
            self._pp = PhreeqPython(database=dat.name,
                                    database_directory=dat.parent)
            lines = ["SURFACE_MASTER_SPECIES",
                     "    Surf_s Surf_sOH",
                     "SURFACE_SPECIES",
                     "    Surf_sOH = Surf_sOH",
                     "        log_k 0"]
            for rx in self._cfg.surface_species:
                lines += [f"    {rx.reaction}",
                          f"        log_k {rx.log_k!r}"]
            punch_tot = " ".join(self._whitelist)
            punch_mol = " ".join(self._bound_species)
            lines += ["SELECTED_OUTPUT 1",
                      "    -reset false",
                      "    -high_precision true",
                      "    -water true",
                      f"    -totals {punch_tot}",
                      f"    -molalities {punch_mol}",
                      "END"]
            self._pp.ip.run_string("\n".join(lines))
        return self._pp

    # -------------------------------------------------------------- public
    def sorb(self, aqueous_elements: np.ndarray, water_mol: float,
             sorbent_sites_mol: float,
             temperature_k: Optional[float] = None) -> SorptionResult:
        e = np.asarray(aqueous_elements, dtype=np.float64)
        zeros = np.zeros(len(ELEMENT_IDS))
        if (sorbent_sites_mol <= 0.0 or water_mol <= 0.0
                or not np.any(e > 0.0)):
            # null fast path: no sites / no water / no solutes - identical
            # to the operator being absent (the S1a null gate)
            return SorptionResult("ok", zeros, {})
        before = self._audit(self._dat)
        if before != self.baseline_audit:
            raise RuntimeError(
                "sorption dat changed since operator construction - "
                "read-only input violated (PROVENANCE audit rule)")
        t_k = (self.temperature_k if temperature_k is None
               else float(temperature_k))
        key = json.dumps({"e": [v.hex() for v in e],
                          "w": float(water_mol).hex(),
                          "s": float(sorbent_sites_mol).hex(),
                          "t": t_k.hex()})
        hit = self._memo.get(key)
        if hit is not None:
            self._memo.move_to_end(key)
            return SorptionResult(hit.status, hit.sorbed_mol.copy(),
                                  dict(hit.site_occupancy))

        # equilibrium is intensive: scale (solution, water, sites) jointly
        # to the canonical magnitude where IPhreeqc converges - RVE ledger
        # amounts are ~1e-10 mol and PHREEQC's absolute convergence criteria
        # fail there (measured: mass-balance residuals ~1e-14 with
        # 'Numerical method failed'). The GEMS backend does exactly this
        # (CANONICAL_MAX_ELEMENT_MOL); the partition scales back exactly.
        peak = float(e.max())
        s_fac = 1e-2 / peak
        e_s = e * s_fac
        elements = {el: float(e_s[i]) for i, el in enumerate(ELEMENT_IDS)
                    if e_s[i] > 0.0}
        reactants = _decompose_to_reactants(elements)
        water_kg = (water_mol * s_fac) * _H2O_G_MOL / 1000.0
        lines = ["SOLUTION 1",
                 f"    temp {t_k - 273.15:.6f}",
                 f"    water {water_kg!r} kg",
                 "REACTION 1"]
        lines += [f"    {f} {mol!r}" for f, mol in sorted(reactants.items())]
        # PHREEQC REACTION semantics: moles added = coefficient x amount.
        # Coefficients above ARE the absolute ledger mols, so the amount is
        # exactly 1.0 (the spike README's measured convention).
        lines += ["    1.0 moles",
                  "SURFACE 1",
                  f"    Surf_sOH {sorbent_sites_mol * s_fac!r} 600.0 1.0",
                  "    -no_edl",
                  "END"]
        pp = self._instance()
        try:
            pp.ip.run_string("\n".join(lines))
            rows = pp.ip.get_selected_output_array()
        except Exception as exc:               # IPhreeqc raises plain errors
            raise BackendTransientError(
                f"PHREEQC sorption call failed: {exc}") from exc
        header = [str(h).strip() for h in rows[0]]
        col = {h: i for i, h in enumerate(header)}
        last = rows[-1]
        kgw = float(last[col["mass_H2O"]])
        occupancy: Dict[str, float] = {}
        sorbed = np.zeros(len(ELEMENT_IDS))
        for i, sp in enumerate(self._bound_species):
            n_i = float(last[col[f"m_{sp}(mol/kgw)"]]) * kgw / s_fac
            occupancy[sp] = n_i
            sorbed += n_i * self._rows[i]
        # closure witness: for every whitelisted element the config-declared
        # rows must reproduce PHREEQC's own solution balance - a wrong
        # sorbed_elements declaration is refused, never absorbed
        scale = max(float(np.abs(e).max()), float(np.abs(sorbed).max()),
                    1e-30)
        for el in self._whitelist:
            k = ELEMENT_IDS.index(el)
            aq_after = float(last[col[f"{el}(mol/kgw)"]]) * kgw / s_fac
            drift = abs((e[k] - aq_after) - sorbed[k])
            if drift > 1e-8 * scale + 1e-18:
                raise RuntimeError(
                    f"sorbed_elements for {el} disagrees with PHREEQC's "
                    f"solution balance by {drift:.3e} mol (input {e[k]!r}, "
                    f"aqueous after {aq_after!r}, declared row gives "
                    f"{sorbed[k]!r}) - fix the config rows")
        after = self._audit(self._dat)
        if after != self.baseline_audit:
            raise RuntimeError("sorption dat changed during the call")
        result = SorptionResult("ok", sorbed, occupancy)
        self._memo[key] = SorptionResult("ok", sorbed.copy(),
                                         dict(occupancy))
        if len(self._memo) > self.memo_cap:
            self._memo.popitem(last=False)
        return result
