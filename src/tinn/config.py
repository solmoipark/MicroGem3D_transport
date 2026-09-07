"""Pydantic configuration schema with unit/range validation and config hashing.

All models forbid unknown fields (PRD §5: no guessing, no silent fallback).
Field names carry units explicitly (_um, _K, _h).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from .registry import (ELEMENT_IDS, KINETIC_PHASE_IDS, SALT_PHASE_IDS,
                       SCM_PHASE_IDS, default_registry)

# The rasterizer's periodic bounding box needs d/h + sqrt(3) + 2 voxels
# (half-diagonal halo on each side plus floor granularity); keep in sync with
# geometry._rasterize_sphere.
RASTER_HALO_VOX = 2.0 + math.sqrt(3.0)

_STRICT = ConfigDict(extra="forbid")

PK_PRESETS = ("pk_elakneswaran_2018", "pk_cemgems_2021")


class BinderRecipe(BaseModel):
    model_config = _STRICT
    # Phase mass fractions over the kinetic phases; the remainder to 1.0 is the
    # unassigned inert residual (kept as-is, never invented into chemistry).
    mass_fractions: Dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> "BinderRecipe":
        unknown = set(self.mass_fractions) - set(KINETIC_PHASE_IDS)
        if unknown:
            raise ValueError(
                f"unknown binder phases {sorted(unknown)}; allowed: {KINETIC_PHASE_IDS}"
            )
        for p, f in self.mass_fractions.items():
            if not (0.0 <= f <= 1.0):
                raise ValueError(f"mass fraction of {p} must be in [0, 1], got {f}")
        total = sum(self.mass_fractions.values())
        if total <= 0.0:
            raise ValueError("binder recipe has zero total mass fraction")
        if total > 1.0 + 1e-9:
            raise ValueError(f"binder mass fractions sum to {total} > 1")
        return self

    @property
    def unassigned(self) -> float:
        return max(0.0, 1.0 - sum(self.mass_fractions.values()))


class PSDBin(BaseModel):
    model_config = _STRICT
    d_lo_um: float = Field(gt=0.0)
    d_hi_um: float = Field(gt=0.0)
    volume_fraction: float = Field(ge=0.0, le=1.0)


class RosinRammler(BaseModel):
    """Rosin-Rammler(Weibull) fit of a measured PSD: F(d) = 1 - exp(-(d/d')^n),
    discretized over [d_min, d_max] and renormalized to the window mass (the
    excluded tail fraction is reported by geometry, never hidden)."""
    model_config = _STRICT
    d_prime_um: float = Field(gt=0.0)
    n: float = Field(gt=0.0)
    d_min_um: float = Field(gt=0.0)
    d_max_um: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "RosinRammler":
        if self.d_max_um <= self.d_min_um:
            raise ValueError("rosin_rammler needs d_max_um > d_min_um")
        return self

    def cdf(self, d_um: float) -> float:
        return 1.0 - math.exp(-((d_um / self.d_prime_um) ** self.n))


# Rosin-Rammler discretization density: fixed (deterministic), log-spaced
_RR_BINS_PER_DECADE = 8


class PSD(BaseModel):
    """Particle-size distribution (PRD 1.2 rev.2): exactly ONE of
    - `bins`            — log-interval volume fractions (the canonical form),
    - `cumulative`      — measured cumulative volume curve [(d_um, F)] as it
                          comes off a laser-diffraction report; consecutive
                          points become bins (F must be non-decreasing, start
                          at 0 and end at 1 — nothing is invented),
    - `rosin_rammler`   — measured RR fit parameters (CEMHYD3D convention).
    Whichever is given is converted to `bins` at validation; geometry only
    ever sees bins. `truncate_to_grid` opts into CEMHYD3D-style truncation of
    the coarse tail that cannot be rasterized (default: hard reject)."""
    model_config = _STRICT
    bins: Optional[List[PSDBin]] = Field(default=None, min_length=1)
    cumulative: Optional[List[Tuple[float, float]]] = Field(default=None,
                                                            min_length=2)
    rosin_rammler: Optional[RosinRammler] = None
    truncate_to_grid: bool = False

    @model_validator(mode="after")
    def _check(self) -> "PSD":
        # `bins` is the CANONICAL DERIVED form: when a measured input is
        # present, bins are (re)computed from it — this makes serialized
        # configs (which carry both the measured input and the converted
        # bins) revalidate cleanly on checkpoint load, and grid truncation
        # re-applies idempotently in the cross-validator
        if self.cumulative is not None and self.rosin_rammler is not None:
            raise ValueError(
                "PSD takes cumulative or rosin_rammler, not both")
        if self.cumulative is None and self.rosin_rammler is None \
                and self.bins is None:
            raise ValueError(
                "PSD needs one of bins / cumulative / rosin_rammler")
        if self.cumulative is not None:
            pts = self.cumulative
            ds = [p[0] for p in pts]
            fs = [p[1] for p in pts]
            if any(d <= 0.0 for d in ds):
                raise ValueError("cumulative PSD diameters must be positive")
            if any(b <= a for a, b in zip(ds, ds[1:])):
                raise ValueError("cumulative PSD diameters must be strictly ascending")
            if any(b < a for a, b in zip(fs, fs[1:])):
                raise ValueError("cumulative PSD fractions must be non-decreasing")
            if any(f < 0.0 or f > 1.0 + 1e-9 for f in fs):
                raise ValueError("cumulative PSD fractions must be in [0, 1]")
            # real laser-diffraction curves rarely start at exactly F=0 or end
            # at exactly F=1 (e.g. SRM 114q: 5.1 % below the first reported
            # size) — the measured window is renormalized and the excluded
            # mass is REPORTED (geometry report), exactly like the RR window;
            # nothing is invented outside the measured curve
            window = fs[-1] - fs[0]
            if window <= 0.0:
                raise ValueError("cumulative PSD window carries no mass")
            self.bins = [PSDBin(d_lo_um=ds[i], d_hi_um=ds[i + 1],
                                volume_fraction=(fs[i + 1] - fs[i]) / window)
                         for i in range(len(pts) - 1)]
        if self.rosin_rammler is not None:
            rr = self.rosin_rammler
            n_bins = max(1, math.ceil(
                _RR_BINS_PER_DECADE * math.log10(rr.d_max_um / rr.d_min_um)))
            ratio = (rr.d_max_um / rr.d_min_um) ** (1.0 / n_bins)
            edges = [rr.d_min_um * ratio ** i for i in range(n_bins + 1)]
            edges[-1] = rr.d_max_um  # exact endpoint, no float drift
            window = rr.cdf(rr.d_max_um) - rr.cdf(rr.d_min_um)
            if window <= 0.0:
                raise ValueError("rosin_rammler window carries no mass")
            self.bins = [PSDBin(d_lo_um=lo, d_hi_um=hi,
                                volume_fraction=(rr.cdf(hi) - rr.cdf(lo)) / window)
                         for lo, hi in zip(edges, edges[1:])]
        for b in self.bins:
            if b.d_hi_um <= b.d_lo_um:
                raise ValueError(f"PSD bin needs d_hi > d_lo, got [{b.d_lo_um}, {b.d_hi_um}]")
        for a, b in zip(self.bins, self.bins[1:]):
            if b.d_lo_um < a.d_hi_um - 1e-12:
                raise ValueError("PSD bins must be ascending and non-overlapping")
        total = sum(b.volume_fraction for b in self.bins)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"PSD volume fractions must sum to 1, got {total}")
        return self

    @property
    def d_max_um(self) -> float:
        return self.bins[-1].d_hi_um

    def measured_window_excluded(self) -> float:
        """Mass fraction of the measured input outside the represented window
        (below the first / above the last cumulative point, or outside the RR
        [d_min, d_max]); 0 for direct bins."""
        if self.rosin_rammler is not None:
            rr = self.rosin_rammler
            return 1.0 - (rr.cdf(rr.d_max_um) - rr.cdf(rr.d_min_um))
        if self.cumulative is not None:
            return 1.0 - (self.cumulative[-1][1] - self.cumulative[0][1])
        return 0.0

    @classmethod
    def synthetic_default(cls) -> "PSD":
        edges_vf = [(0.5, 1.0, 0.06), (1.0, 2.0, 0.12), (2.0, 4.0, 0.22),
                    (4.0, 8.0, 0.30), (8.0, 16.0, 0.30)]
        return cls(bins=[PSDBin(d_lo_um=lo, d_hi_um=hi, volume_fraction=vf)
                         for lo, hi, vf in edges_vf])


class TabulatedTable(BaseModel):
    model_config = _STRICT
    times_h: List[float] = Field(min_length=2)
    alpha: Dict[str, List[float]]

    @model_validator(mode="after")
    def _check(self) -> "TabulatedTable":
        if self.times_h[0] < 0.0:
            raise ValueError("times_h must start at >= 0")
        for a, b in zip(self.times_h, self.times_h[1:]):
            if b <= a:
                raise ValueError("times_h must be strictly increasing")
        if not self.alpha:
            raise ValueError("alpha table is empty")
        unknown = set(self.alpha) - set(KINETIC_PHASE_IDS)
        if unknown:
            raise ValueError(f"unknown phases in alpha table: {sorted(unknown)}")
        for p, vals in self.alpha.items():
            if len(vals) != len(self.times_h):
                raise ValueError(f"alpha[{p}] length must match times_h")
            for v in vals:
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"alpha[{p}] values must be in [0, 1]")
            for a, b in zip(vals, vals[1:]):
                if b < a:
                    raise ValueError(f"alpha[{p}] must be non-decreasing")
        return self


class KineticsConfig(BaseModel):
    model_config = _STRICT
    kind: Literal["tabulated", "pk"]
    preset: Optional[str] = None
    table: Optional[TabulatedTable] = None
    blaine_m2_kg: Optional[float] = None
    # numerical policy for the P&K explicit-Euler integration (PRD §4.1)
    # >= 1e-15 keeps 1 - seed representable so the Jander denominator stays finite
    pk_alpha_seed: float = Field(default=1e-8, ge=1e-15, lt=1.0)
    pk_max_substep_days: float = Field(default=0.01, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "KineticsConfig":
        if self.kind == "pk":
            if self.preset not in PK_PRESETS:
                raise ValueError(f"pk kinetics requires preset in {PK_PRESETS}, got {self.preset!r}")
            if self.table is not None:
                raise ValueError("pk kinetics does not take a table")
            if self.blaine_m2_kg is None or self.blaine_m2_kg <= 0.0:
                raise ValueError("pk kinetics requires a positive blaine_m2_kg")
        else:
            if self.table is None:
                raise ValueError("tabulated kinetics requires a table")
            if self.preset is not None:
                raise ValueError("tabulated kinetics does not take a preset")
            if self.blaine_m2_kg is not None:
                raise ValueError("blaine_m2_kg only applies to pk kinetics")
        return self


class ReactionRule(BaseModel):
    """Fixed synthetic stoichiometry: mol H2O consumed and product mol per mol dissolved."""
    model_config = _STRICT
    water_mol: float = Field(ge=0.0)
    products: Dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> "ReactionRule":
        reg = default_registry()
        for pid, n in self.products.items():
            if pid not in reg:
                raise ValueError(f"unknown product phase {pid!r}")
            if reg.get(pid).kind != "hydrate":
                raise ValueError(f"product {pid!r} is not a hydrate phase")
            if n <= 0.0:
                raise ValueError(f"product coefficient for {pid} must be > 0")
        return self


def _default_rules() -> Dict[str, "ReactionRule"]:
    # Element-balanced synthetic reactions (checked against registry formulas):
    #  C3S  + 5.3 H2O -> C1.7SH4 + 1.3 CH
    #  C2S  + 4.3 H2O -> C1.7SH4 + 0.3 CH
    #  C3A  + 6.0 H2O -> C3AH6
    #  C4AF + 10  H2O -> C3AH6 + CH + FH3
    return {
        "C3S": ReactionRule(water_mol=5.3, products={"CSH": 1.0, "CH": 1.3}),
        "C2S": ReactionRule(water_mol=4.3, products={"CSH": 1.0, "CH": 0.3}),
        "C3A": ReactionRule(water_mol=6.0, products={"C3AH6": 1.0}),
        "C4AF": ReactionRule(water_mol=10.0, products={"C3AH6": 1.0, "CH": 1.0, "FH3": 1.0}),
    }


def _check_rule_element_balance(phase_id: str, rule: ReactionRule) -> None:
    """Reactant + water elements must equal product elements (PRD §6.1 is blocking)."""
    reg = default_registry()
    lhs: Dict[str, float] = dict(reg.get(phase_id).formula)
    for el, n in reg.get("H2O").formula.items():
        lhs[el] = lhs.get(el, 0.0) + rule.water_mol * n
    rhs: Dict[str, float] = {}
    for pid, coeff in rule.products.items():
        for el, n in reg.get(pid).formula.items():
            rhs[el] = rhs.get(el, 0.0) + coeff * n
    for el in set(lhs) | set(rhs):
        if abs(lhs.get(el, 0.0) - rhs.get(el, 0.0)) > 1e-9:
            raise ValueError(
                f"stoichiometric rule for {phase_id} is not element-balanced: "
                f"{el} lhs={lhs.get(el, 0.0)} rhs={rhs.get(el, 0.0)}"
            )


DEFAULT_GEMS_BUNDLE_LST = "gems_bundles/CNASH/Test-dat.lst"
# declared gel porosity of the C-S-H solid solutions (both models treated
# alike); legacy configs that omitted the map hashed the pre-CNASH form, so
# config_hash canonicalizes this exact map back to it (see config_hash)
DEFAULT_GEMS_GEL_POROSITY = {"CSHQ": 0.28, "CNASH": 0.28}
_LEGACY_GEL_POROSITY_HASH_FORM = {"CSHQ": 0.28}


class ChemistryConfig(BaseModel):
    model_config = _STRICT
    backend: Literal["stoichiometric", "gems3k"]
    # None for gems3k (kept out of config_hash); defaults filled for stoichiometric.
    stoichiometric_rules: Optional[Dict[str, ReactionRule]] = None
    gems_bundle_lst: Optional[str] = None      # default: CNASH Test bundle
    gems_worker_python: Optional[str] = None   # interpreter with xgems installed
    # declared gel porosity per GEMS solid phase; phases absent from the map are
    # crystalline (0.0). GEMS volumes are solid skeletons; envelope = skel/(1-eps).
    gems_gel_porosity: Optional[Dict[str, float]] = None
    # RT-S3 (2026-09-03): config-declared GEMS exclusions beyond the clinker
    # phases. The ledger's O/H are oxide-exact, so the GEM system sits at the
    # H2/H2O redox floor and reduces sulfate: with the PC bundle the reduced
    # sulfur leaves as pyrite (measured 3-35 vox), with the user's PC-Cl
    # export (no sulfide phases) it stays as 53 mM HS- in the pore solution
    # and the sorption operator re-oxidised it into a 95 % sulfate store.
    # Cement thermodynamics treats sulfur as S(VI): declare the reduced
    # species (e.g. HS-, H2S@, S-2, S2O3-2, HSO3-, SO3-2, H2S, Sulfur) and
    # sulfide phases (Pyrite, Troilite) here. Names are typo-guarded against
    # the bundle; a suppressed name that still precipitates is an error.
    # None keeps every earlier config hash.
    suppressed_species: Optional[List[str]] = None
    suppressed_phases: Optional[List[str]] = None

    @model_validator(mode="after")
    def _check(self) -> "ChemistryConfig":
        for label, lst in (("suppressed_species", self.suppressed_species),
                           ("suppressed_phases", self.suppressed_phases)):
            if lst is not None:
                if not lst or len(set(lst)) != len(lst) or any(
                        not isinstance(x, str) or not x for x in lst):
                    raise ValueError(
                        f"chemistry.{label} must be a non-empty list of "
                        f"distinct names (omit it instead of an empty list)")
        if self.backend == "gems3k":
            if self.stoichiometric_rules is not None:
                raise ValueError("gems3k backend does not take stoichiometric_rules")
            if not self.gems_bundle_lst:
                self.gems_bundle_lst = DEFAULT_GEMS_BUNDLE_LST
            if self.gems_gel_porosity is None:
                self.gems_gel_porosity = dict(DEFAULT_GEMS_GEL_POROSITY)
            for phase, eps in self.gems_gel_porosity.items():
                if not (0.0 <= eps < 1.0):
                    raise ValueError(f"gel porosity of {phase} must be in [0, 1)")
            return self
        if (self.gems_bundle_lst is not None or self.gems_worker_python is not None
                or self.gems_gel_porosity is not None
                or self.suppressed_species is not None
                or self.suppressed_phases is not None):
            raise ValueError("gems_* / suppressed_* fields only apply to the "
                             "gems3k backend")
        if self.stoichiometric_rules is None:
            self.stoichiometric_rules = _default_rules()
        unknown = set(self.stoichiometric_rules) - set(KINETIC_PHASE_IDS)
        if unknown:
            raise ValueError(f"stoichiometric rules for unknown phases: {sorted(unknown)}")
        for phase_id, rule in self.stoichiometric_rules.items():
            _check_rule_element_balance(phase_id, rule)
        return self


class RVEConfig(BaseModel):
    model_config = _STRICT
    # 128^3 @ 0.25 um serves the resolution-convergence track (PRD 1.2 rev.2);
    # below ~0.1 um voxels the capillary/gel-pore split would double-count
    # C-S-H gel porosity, so finer grids are out of scope by design
    grid_size: Literal[32, 64, 128]
    voxel_size_um: float = Field(ge=0.25, le=1.0)
    seed: int = Field(ge=0)


class TimeStepWindow(BaseModel):
    """Piecewise-constant external timestep cap up to ``until_h``."""
    model_config = _STRICT
    until_h: float = Field(gt=0.0)
    dt_h: float = Field(gt=0.0)


class ScheduleConfig(BaseModel):
    model_config = _STRICT
    output_times_h: List[float] = Field(min_length=1)
    dt_initial_h: float = Field(default=0.01, gt=0.0)
    dt_min_h: float = Field(default=1e-4, gt=0.0)
    max_retries: int = Field(default=8, ge=1)
    dt_windows: Optional[List[TimeStepWindow]] = None

    @model_validator(mode="after")
    def _check(self) -> "ScheduleConfig":
        if self.output_times_h[0] <= 0.0:
            raise ValueError("output times must be > 0")
        for a, b in zip(self.output_times_h, self.output_times_h[1:]):
            if b <= a:
                raise ValueError("output_times_h must be strictly increasing")
        if self.dt_min_h > self.dt_initial_h:
            raise ValueError("dt_min_h must be <= dt_initial_h")
        if self.dt_windows is not None:
            if not self.dt_windows:
                raise ValueError("dt_windows must be non-empty when provided")
            for a, b in zip(self.dt_windows, self.dt_windows[1:]):
                if b.until_h <= a.until_h:
                    raise ValueError("dt_windows until_h values must increase")
            if self.dt_windows[-1].until_h < self.output_times_h[-1] - 1e-12:
                raise ValueError(
                    "last dt_window must cover the last output time")
            if any(w.dt_h < self.dt_min_h for w in self.dt_windows):
                raise ValueError("every dt_window dt_h must be >= dt_min_h")
            if self.dt_windows[0].dt_h != self.dt_initial_h:
                raise ValueError(
                    "dt_initial_h must equal the first dt_window dt_h")
        return self

    def dt_cap_at(self, time_h: float) -> float:
        if self.dt_windows is None:
            return self.dt_initial_h
        for window in self.dt_windows:
            if time_h < window.until_h - 1e-12:
                return window.dt_h
        return self.dt_windows[-1].dt_h

    def next_window_end_after(self, time_h: float) -> Optional[float]:
        if self.dt_windows is None:
            return None
        for window in self.dt_windows:
            if window.until_h > time_h + 1e-12:
                return window.until_h
        return None


class ParticleShape(BaseModel):
    """Per-material particle shape (PRD 1.2 rev.2). Semi-axis ratios a:b:c on
    any positive scale — rasterization volume-normalizes them (abc -> 1) so
    the PSD keeps its volume-equivalent-diameter meaning. Orientation is
    sampled uniformly per particle from the seeded RNG."""
    model_config = _STRICT
    kind: Literal["ellipsoid"] = "ellipsoid"
    aspects: Tuple[float, float, float]

    @model_validator(mode="after")
    def _check(self) -> "ParticleShape":
        if any(a <= 0.0 for a in self.aspects):
            raise ValueError("shape aspects must all be positive")
        return self

    def normalized_axes(self) -> Tuple[float, float, float]:
        g = (self.aspects[0] * self.aspects[1] * self.aspects[2]) ** (1.0 / 3.0)
        return (self.aspects[0] / g, self.aspects[1] / g, self.aspects[2] / g)

    def elongation(self) -> float:
        return max(self.normalized_axes())


class ScmComposition(BaseModel):
    """A measured SCM glass composition, replacing the built-in one for that
    phase id (PRD 1.2 v3.0). Oxide wt% is the form a mill certificate or an
    XRF table reports; the formula unit is those oxides per 100 g, exactly as
    registry.scm_entry builds the built-ins. Only the LISTED oxides enter the
    chemistry — an unlisted residue (LOI, unburnt carbon) is simply absent,
    never invented into reactive mass. The density fixes the molar volume."""
    model_config = _STRICT
    oxides_wt_pct: Dict[str, float]
    density_g_cm3: float = Field(gt=0.0, le=8.0)

    @model_validator(mode="after")
    def _check(self) -> "ScmComposition":
        if not self.oxides_wt_pct:
            raise ValueError("scm_composition needs at least one oxide")
        for ox, wt in self.oxides_wt_pct.items():
            if not (0.0 <= wt <= 100.0):
                raise ValueError(f"oxide {ox} wt% must be in [0, 100], got {wt}")
        total = sum(self.oxides_wt_pct.values())
        if total > 100.0 + 1e-9:
            raise ValueError(
                f"listed oxides sum to {total} wt% > 100 - a glass composition "
                f"is per 100 g of material")
        return self


class BoundaryReservoirConfig(BaseModel):
    """RT-W3 (PRD 4.6.3): fixed-composition bath on declared exposed
    face(s) = a continuously-renewed reservoir imposed as a surface
    boundary condition. Declaring ANY side de-periodizes labeling and the
    domain graph along the whole axis (the seam is one identification;
    an un-coupled face becomes a sealed no-flux wall). Solutes only —
    boundary_water_mol stays 0."""
    model_config = _STRICT
    axis: Literal["z", "y", "x"]           # dense arrays are (z, y, x)
    side: Literal["low", "high", "both"]   # low = index 0
    # exposure start time. None = 0 (bath active from mixing). A positive
    # value runs SEALED (fully periodic, bit-identical to a boundary-free
    # config) until then - the mature-paste protocol: hydrate first, then
    # expose. Must coincide with an output time so no step straddles the
    # switch (validated cross-section).
    start_h: Optional[float] = Field(default=None, ge=0.0)
    # dissolved ELEMENT concentrations, mol per m^3 of capillary water
    # (volumetric — the solver's c = n/W basis; ~molarity*1000 for dilute
    # baths). {} = pure (deionized, continuously renewed) water. Compose
    # ONLY from neutral formula units (NaOH c -> {Na: c, O: c, H: c};
    # the element state has no charge coordinate, so Na alone would be
    # metallic sodium). Solvent H/O are NOT part of the composition.
    composition_mol_per_m3: Dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> "BoundaryReservoirConfig":
        unknown = set(self.composition_mol_per_m3) - set(ELEMENT_IDS)
        if unknown:
            raise ValueError(
                f"unknown bath elements {sorted(unknown)}; allowed: "
                f"{list(ELEMENT_IDS)}")
        for el, c in self.composition_mol_per_m3.items():
            if isinstance(c, bool) or not math.isfinite(c) or c < 0.0:
                raise ValueError(
                    f"bath concentration of {el} must be a finite number "
                    f">= 0 mol/m^3, got {c!r}")
        return self


class SpeciesDiffusionConfig(BaseModel):
    """Tier 0 (RT-P0, PRD 4.6.4): per-element effective diffusivities from
    aqueous speciation and per-species self-diffusion coefficients under the
    Nernst-Planck zero-current projection. Mutually exclusive with the
    scalar d0_m2_s unless diagnostics_only."""
    model_config = _STRICT
    # vendored table (scripts/make_species_dw_table.py output) with the
    # GEMS-DC alias map; audited by sha256 in the run summary
    dw_table: str
    # dw for GEMS species with no table mapping. None = a run whose bundle
    # declares an unmapped aqueous species REFUSES to start (no silent
    # defaults); a value is recorded per species in the run summary.
    default_dw_m2_s: Optional[float] = Field(default=None, gt=0.0)
    # porous-medium reduction applied uniformly on top of the graph's
    # geometric resistance. REQUIRED and explicit (d0 precedent): 1.0 means
    # the graph carries all geometry, exactly like the scalar path.
    geometry_factor: float = Field(gt=0.0)
    phi_clamp_report: bool = True
    # RT-P0a: keep the scalar d0 physics and only REPORT the NP effective
    # diffusivities per step. False (active transport) lands with RT-P0b.
    diagnostics_only: bool = False
    # RT-01 (review 2026-09-07): the BE step advances element columns
    # independently, so the flux it APPLIES can carry net charge even when
    # the frozen projection was zero-current and no clamp fired. The
    # residual is always reported (np_applied_charge_rel_max); a threshold
    # here rejects the trial (dt halves) when it is exceeded. None = report
    # only (every earlier hash unchanged).
    applied_charge_rtol: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "SpeciesDiffusionConfig":
        if not math.isfinite(self.geometry_factor):
            raise ValueError("geometry_factor must be finite")
        if (self.applied_charge_rtol is not None
                and not math.isfinite(self.applied_charge_rtol)):
            raise ValueError("applied_charge_rtol must be finite")
        if (self.default_dw_m2_s is not None
                and not math.isfinite(self.default_dw_m2_s)):
            raise ValueError("default_dw_m2_s must be finite")
        return self


class DomainPartitionConfig(BaseModel):
    """Mode C (PRD 4.6.2): sub-cluster equilibration domains from a static
    axis-aligned tiling, coupled by implicit diffusion on the domain graph."""
    model_config = _STRICT
    # tile edge in voxels; must divide rve.grid_size. == grid_size is the
    # degenerate 1-domain-per-cluster limit (bit-identical to mode full).
    tile_vox: int = Field(ge=2)
    # per-axis tile edges (z, y, x), overriding the cubic tile_vox (W3
    # design review: a planar leaching front is transverse-uniform, so
    # BANDED tiles - transverse = grid, axial 2-4 - are the dominant cost
    # lever AND better physics per GEM call). Each entry divides the grid.
    # None keeps the cubic tiling (and every earlier config hash).
    tile_zyx: Optional[Tuple[int, int, int]] = None
    # common effective free-solution diffusivity for all elements. REQUIRED
    # and explicit - no invented default. Bulk ionic self-diffusivities at
    # 25 C span 0.8e-9 (Ca2+) to 5.3e-9 (OH-) m2/s; 1e-9 is the
    # conventional single-D compromise (PRD 4.6.2: the elemental state
    # carries no speciation, so a per-element D would be an invented
    # speciation; the graph carries the geometry, D0 is the bulk scale).
    # Optional since RT-P0 ONLY to admit the species alternative below; a
    # config without `species` (or with diagnostics_only) still REQUIRES it
    # (validated) — every earlier config carries the field, hash unchanged.
    d0_m2_s: Optional[float] = Field(default=None, gt=0.0)
    # Tier 0 species-resolved diffusion (PRD 4.6.4). None keeps the scalar
    # path and every earlier config hash (the transport-section None sweep).
    species: Optional[SpeciesDiffusionConfig] = None
    # GEM-call economy (PRD 4.6.2). dirty_rtol = 0.0 equilibrates every
    # wet domain every step - the pure mode-C reference.
    dirty_rtol: float = Field(default=0.0, ge=0.0)
    eq_max_age_steps: int = Field(default=16, ge=1)
    max_gem_calls_per_step: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "DomainPartitionConfig":
        if self.d0_m2_s is not None and not math.isfinite(self.d0_m2_s):
            raise ValueError("d0_m2_s must be finite")
        if not math.isfinite(self.dirty_rtol):
            raise ValueError("dirty_rtol must be finite")
        if self.tile_zyx is not None and any(t < 1 for t in self.tile_zyx):
            raise ValueError("tile_zyx entries must be >= 1")
        # d0 XOR active species (PSD precedent: one source of truth, both =
        # error); diagnostics ride the scalar physics so they need d0 too
        active = self.species is not None and not self.species.diagnostics_only
        if active and self.d0_m2_s is not None:
            raise ValueError(
                "d0_m2_s and active transport.domains.species are mutually "
                "exclusive - the per-element conductances replace the scalar")
        if not active and self.d0_m2_s is None:
            raise ValueError(
                "transport.domains needs d0_m2_s (species diagnostics and "
                "the scalar path both run on it; no invented default)")
        return self

    def tiles(self, grid_size: int) -> Tuple[int, int, int]:
        if self.tile_zyx is not None:
            return self.tile_zyx
        return (self.tile_vox, self.tile_vox, self.tile_vox)


class TransportConfig(BaseModel):
    """v4.0/RT chemistry-transport modes (PRD 1.4/4.6). Absence of the
    section — or None in every field — preserves the exact current engine,
    bit for bit, and every earlier config hash."""
    model_config = _STRICT
    # Mode C: sub-cluster equilibration domains (PRD 4.6.2). None => one
    # well-mixed reactor per connected cluster (exact legacy path). Allowed
    # with BOTH backends: with stoichiometric it still changes real physics
    # (per-domain dissolution/placement), which lets the partition gates
    # run GEMS-free.
    domains: Optional[DomainPartitionConfig] = None
    # RT-W3 boundary reservoir (PRD 4.6.3). None => sealed periodic RVE.
    boundary: Optional[BoundaryReservoirConfig] = None
    # Mode B: rate-limited re-equilibration (PRD 4.6.1). Per step, only the
    # fraction f = min(1, dt/tau) of each domain's owned solid-solution
    # inventory is offered to the equilibrium; the withheld remainder keeps
    # its stored composition and is chemically inert that step (turnover
    # n/tau — the declared C-S-H recrystallization/exchange time, literature
    # months to years). Single-endmember crystalline channels default to
    # tau = 0 (always fully offered: their dissolution/growth is
    # surface-controlled, and withholding CH would silently break pH
    # buffering). None => full re-equilibration (exact legacy code path).
    exchange_tau_h: Optional[float] = Field(default=None, gt=0.0)
    # Per-channel overrides in hours, keyed by bundle hydrate ids (validated
    # against the actual bundle at Engine construction — config cannot know
    # it). 0.0 = always fully offered.
    exchange_tau_h_per_phase: Optional[Dict[str, float]] = None

    @field_validator("exchange_tau_h", "exchange_tau_h_per_phase",
                     mode="before")
    @classmethod
    def _no_boolean_taus(cls, v):
        # lax coercion would turn a typo'd JSON true/false into 1.0/0.0 hours
        # - a physical exchange time, silently orders of magnitude off
        vals = v.values() if isinstance(v, dict) else (v,)
        for item in vals:
            if isinstance(item, bool):
                raise ValueError(
                    "boolean is not an exchange time - give hours")
        return v

    @model_validator(mode="after")
    def _check(self) -> "TransportConfig":
        if self.exchange_tau_h is not None and not math.isfinite(
                self.exchange_tau_h):
            raise ValueError("exchange_tau_h must be finite (an infinite "
                             "exchange time is the tau -> inf limit; it is "
                             "not representable in the config hash)")
        if self.exchange_tau_h_per_phase is not None:
            if not self.exchange_tau_h_per_phase:
                raise ValueError(
                    "exchange_tau_h_per_phase must not be empty - omit it")
            for key, val in self.exchange_tau_h_per_phase.items():
                if not math.isfinite(val) or val < 0.0:
                    # NaN passes a bare `< 0.0` check and would silently
                    # disable the rate limit downstream (review finding)
                    raise ValueError(
                        f"exchange_tau_h_per_phase[{key!r}] must be a finite "
                        f"number >= 0 (0 = always fully offered), got {val}")
            if (self.exchange_tau_h is None
                    and all(v == 0.0
                            for v in self.exchange_tau_h_per_phase.values())):
                raise ValueError(
                    "exchange_tau_h_per_phase contains only 0.0 (= fully "
                    "offered) and there is no global exchange_tau_h - the "
                    "section declares rate limiting but limits nothing")
        if self.boundary is not None and self.domains is None:
            raise ValueError(
                "transport.boundary needs transport.domains - the bath "
                "couples to the domain graph (tile_vox == grid_size gives "
                "one reactor per cluster)")
        return self

    def rate_limited(self) -> bool:
        return (self.exchange_tau_h is not None
                or self.exchange_tau_h_per_phase is not None)


_SPECIES_CHARGE_RE = re.compile(r"^(.*?)([+-]\d*)$")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?|\(|\))(\d*\.?\d*)")


def _parse_species_elements(name: str) -> Tuple[Dict[str, float], float]:
    """PHREEQC species name -> (element counts, charge). Surface species
    keep their site as a pseudo-element 'Surf_x' so the site balances like
    any other element (RT-03). Charge is the trailing +/-N; a bare +/- is
    +/-1. Parentheses nest (Al(OH)4-). Anything unparsable is an error -
    a reaction the platform cannot audit is refused, never guessed."""
    m = _SPECIES_CHARGE_RE.match(name.strip())
    base, q = (m.group(1), m.group(2)) if m and m.group(2) else (name.strip(), "")
    charge = 0.0 if not q else (float(q) if len(q) > 1 else float(q + "1"))
    counts: Dict[str, float] = {}
    if base.startswith("Surf_"):
        site, base = base[:6], base[6:]          # 'Surf_s' / 'Surf_c'
        counts[site] = 1.0
    pos = 0
    stack: List[Dict[str, float]] = [counts]
    while pos < len(base):
        tok = _FORMULA_TOKEN_RE.match(base, pos)
        if tok is None or tok.end() == pos:
            raise ValueError(f"cannot parse species {name!r} at {base[pos:]!r}")
        sym, num = tok.group(1), tok.group(2)
        pos = tok.end()
        if sym == "(":
            stack.append({})
            continue
        if sym == ")":
            grp = stack.pop()
            mult = float(num) if num else 1.0
            for el, v in grp.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + v * mult
            continue
        stack[-1][sym] = stack[-1].get(sym, 0.0) + (float(num) if num else 1.0)
    if len(stack) != 1:
        raise ValueError(f"unbalanced parentheses in species {name!r}")
    return counts, charge


def surface_reaction_removal(reaction: str) -> Dict[str, float]:
    """Element mols REMOVED from solution per mol of reaction, derived from
    the equation itself: aqueous reactants minus aqueous products. Refuses
    an element- or charge-unbalanced equation and elements outside the
    ledger (RT-03: the declared row must reproduce this, O/H included)."""
    lhs, rhs = (side.strip() for side in reaction.split("=", 1))
    totals = {"lhs": ({}, 0.0), "rhs": ({}, 0.0)}
    removal: Dict[str, float] = {}
    for side, text in (("lhs", lhs), ("rhs", rhs)):
        el_sum: Dict[str, float] = {}
        q_sum = 0.0
        for term in text.split(" + "):
            term = term.strip()
            coef, sp = 1.0, term
            parts = term.split(" ", 1)
            if len(parts) == 2 and re.fullmatch(r"\d*\.?\d+", parts[0]):
                coef, sp = float(parts[0]), parts[1].strip()
            counts, charge = _parse_species_elements(sp)
            q_sum += coef * charge
            for el, v in counts.items():
                el_sum[el] = el_sum.get(el, 0.0) + coef * v
            if not sp.startswith("Surf_"):
                sign = 1.0 if side == "lhs" else -1.0
                for el, v in counts.items():
                    removal[el] = removal.get(el, 0.0) + sign * coef * v
        totals[side] = (el_sum, q_sum)
    (l_el, l_q), (r_el, r_q) = totals["lhs"], totals["rhs"]
    for el in set(l_el) | set(r_el):
        if abs(l_el.get(el, 0.0) - r_el.get(el, 0.0)) > 1e-9:
            raise ValueError(
                f"surface reaction {reaction!r} is not element-balanced "
                f"({el}: {l_el.get(el, 0.0)} vs {r_el.get(el, 0.0)})")
    if abs(l_q - r_q) > 1e-9:
        raise ValueError(
            f"surface reaction {reaction!r} is not charge-balanced "
            f"({l_q} vs {r_q})")
    bad = sorted(el for el in removal if el not in ELEMENT_IDS)
    if bad:
        raise ValueError(
            f"surface reaction {reaction!r} moves elements outside the "
            f"ledger: {bad}")
    return {el: v for el, v in removal.items() if abs(v) > 1e-12}


class SurfaceReaction(BaseModel):
    """One PHREEQC SURFACE_SPECIES reaction (RT-S1, spec 3). cemdata18.dat
    defines NO surface chemistry (measured: zero SURFACE blocks), so the
    reactions and constants are config-owned - literature values go HERE,
    never invented in code (d0 precedent). sorbed_elements declares the
    element mols REMOVED from solution per mol of the bound species formed
    (negative = released, e.g. the OH- freed by ligand exchange); the
    operator cross-checks it against PHREEQC's own solution totals on
    every call."""
    model_config = _STRICT
    reaction: str            # e.g. "Surf_sOH + SO4-2 = Surf_sSO4- + OH-"
    log_k: float
    sorbed_elements: Dict[str, float]   # per mol bound, ELEMENT_IDS keys

    @model_validator(mode="after")
    def _check(self) -> "SurfaceReaction":
        if not math.isfinite(self.log_k):
            raise ValueError("surface reaction log_k must be finite")
        pools = [p for p in ("Surf_s", "Surf_c") if p in self.reaction]
        if "=" not in self.reaction or not pools:
            raise ValueError(
                "surface reaction must be a PHREEQC equation over the "
                "Surf_s site (e.g. 'Surf_sOH + SO4-2 = Surf_sSO4- + OH-') "
                "or the Surf_c pool (RT-Cl-2)")
        if len(pools) == 2:
            raise ValueError(
                f"surface reaction {self.reaction!r} spans both site pools "
                "- one reaction, one pool (Surf_s or Surf_c)")
        unknown = sorted(set(self.sorbed_elements) - set(ELEMENT_IDS))
        if unknown:
            raise ValueError(
                f"sorbed_elements has unknown elements {unknown}")
        if not any(v != 0.0 for v in self.sorbed_elements.values()):
            raise ValueError("sorbed_elements is all zero - the reaction "
                             "would sorb nothing (omit it instead)")
        for el, v in self.sorbed_elements.items():
            if not math.isfinite(v):
                raise ValueError(f"sorbed_elements[{el!r}] must be finite")
        # RT-03 (external review 2026-09-07): the declared row must equal
        # the removal vector the equation itself implies, O/H INCLUDED -
        # the operator's closure witness audits solutes only and the ledger
        # closes with any row applied consistently, so a wrong O/H row was
        # invisible to both (reproduced: S:1,O:0,H:0 was accepted).
        derived = surface_reaction_removal(self.reaction)
        for el in set(derived) | set(self.sorbed_elements):
            d, s = derived.get(el, 0.0), self.sorbed_elements.get(el, 0.0)
            if abs(d - s) > 1e-9:
                raise ValueError(
                    f"sorbed_elements for {self.reaction!r} disagree with the "
                    f"equation: declared {dict(self.sorbed_elements)}, derived "
                    f"{derived} (element {el}: {s} vs {d})")
        return self

    @property
    def bound_species(self) -> str:
        """Product surface species name (first Surf_s term on the RHS).
        Terms split on the ' + ' separator, NOT on every '+', so charged
        species names like Surf_sOCa+ survive intact (RT-S2a fix)."""
        rhs = self.reaction.split("=", 1)[1]
        for term in rhs.split(" + "):
            term = term.strip()
            if term.startswith("Surf_"):
                return term
        raise ValueError(f"no Surf_ product in {self.reaction!r}")

    @property
    def site_pool(self) -> str:
        """'s' (silanol, Surf_s) or 'c' (RT-Cl-2 second pool, Surf_c)."""
        return "c" if "Surf_c" in self.reaction else "s"


class SorptionConfig(BaseModel):
    """Tier 1 (RT-S1): PHREEQC SURFACE sorption operator on C-S-H sites.
    Phase-assemblage authority stays with GEMS (mechanism table, spec 1):
    the operator runs SOLUTION + SURFACE only. Absence of this section =
    no S stage, bit-identical engine, unchanged config hash."""
    model_config = _STRICT
    operator: Literal["phreeqc_surface"]
    phreeqc_dat: str
    surface_model: Literal["no_edl", "ddl"] = "no_edl"
    # ddl only (RT-S2a): Gouy-Chapman needs the physical surface area the
    # site total lives on. Literature-derived, config-owned (d0 precedent):
    # Divet SSA 350 m2/g / 2.79 mmol sites/g = 1.254e5 m2 per mol sites.
    specific_area_m2_per_mol_site: Optional[float] = None
    # ddl only: intrinsic charging reactions (silanol deprotonation, Ca
    # complexation) - these SET the surface potential every ion feels; the
    # sorbed_elements rows book their solution deltas in the same ledger.
    charging_reactions: List[SurfaceReaction] = Field(default_factory=list)
    # RT-S1d: pH buffer for the S-stage subsystem. The snapshot re-offer
    # returns the store as SO3 while the base it released has gone to
    # solids in the R stage, so without a buffer the operator solves an
    # acid pseudo-solution (measured pH 0.7 / -0.3, PRD 4.6.5 S1-OPEN-2).
    # "Portlandite" includes the reactor's OWNED CH as an equilibrium
    # phase; its delta is booked solution <-> CH pool for the R stage.
    buffer_phase: Optional[Literal["Portlandite"]] = None
    # sites per mol of each C-S-H endmember (bundle DC names; REQUIRED, no
    # defaults - the engine validates the keys against the bundle at S1b).
    # Uncovered pool mass takes the global endmember fractions (the E2
    # feeding fallback), so the coverage gap adds no new assumption.
    site_density_mol_per_mol: Dict[str, float]
    # RT-Cl-2: OPTIONAL second, independent site pool Surf_c - the
    # Ca-decorated silanol sub-population that binds Cl- (Elakneswaran
    # 2009 eq. 6). Hirao 2005 saturates C-S-H at 0.616 mmol/g = 22 % of
    # the silanol total, so ONE pool cannot reproduce the isotherm shape.
    # Same keys/units as site_density_mol_per_mol; None = no pool, and
    # every earlier sorption config keeps its hash.
    site_density_c_mol_per_mol: Optional[Dict[str, float]] = None
    surface_species: List[SurfaceReaction] = Field(min_length=1)
    # elements the operator may move (solutes only; O/H ride implicitly as
    # the protonation frame). A delta outside this list is a hard error.
    elements: List[str] = Field(min_length=1)
    alkali_exchange: bool = False

    @model_validator(mode="after")
    def _check(self) -> "SorptionConfig":
        unknown = sorted(set(self.elements) - set(ELEMENT_IDS))
        if unknown:
            raise ValueError(f"sorption.elements has unknown elements "
                             f"{unknown}; allowed: {ELEMENT_IDS}")
        if any(el in ("O", "H") for el in self.elements):
            raise ValueError(
                "sorption.elements lists solutes only - O/H ride "
                "implicitly as the surface protonation frame")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("sorption.elements has duplicates")
        if not self.site_density_mol_per_mol:
            raise ValueError("site_density_mol_per_mol must not be empty")
        for k, v in self.site_density_mol_per_mol.items():
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"site_density_mol_per_mol[{k!r}] must be finite >= 0")
        c_rxs = [rx for rx in [*self.surface_species, *self.charging_reactions]
                 if rx.site_pool == "c"]
        if self.site_density_c_mol_per_mol is not None:
            if not self.site_density_c_mol_per_mol:
                raise ValueError(
                    "site_density_c_mol_per_mol must not be empty")
            for k, v in self.site_density_c_mol_per_mol.items():
                if not math.isfinite(v) or v < 0.0:
                    raise ValueError(
                        f"site_density_c_mol_per_mol[{k!r}] must be finite >= 0")
            if not c_rxs:
                raise ValueError(
                    "site_density_c_mol_per_mol given but no reaction uses "
                    "the Surf_c pool (no silent ignore)")
            if self.surface_model == "ddl":
                raise ValueError(
                    "the Surf_c pool is no_edl only - its charging is not "
                    "modelled under 'ddl' (refused, no silent ignore)")
        elif c_rxs:
            raise ValueError(
                f"reaction {c_rxs[0].reaction!r} uses the Surf_c pool but "
                "site_density_c_mol_per_mol is not declared")
        allowed = set(self.elements) | {"O", "H"}
        for rx in [*self.surface_species, *self.charging_reactions]:
            bad = sorted(set(rx.sorbed_elements) - allowed)
            if bad:
                raise ValueError(
                    f"surface reaction {rx.reaction!r} moves {bad} - "
                    f"outside the declared elements whitelist")
        if self.surface_model == "ddl":
            a = self.specific_area_m2_per_mol_site
            if a is None or not math.isfinite(a) or a <= 0.0:
                raise ValueError(
                    "surface_model 'ddl' requires a finite positive "
                    "specific_area_m2_per_mol_site (the Gouy-Chapman "
                    "charge density needs real area - no placeholder)")
        else:
            if self.specific_area_m2_per_mol_site is not None:
                raise ValueError(
                    "specific_area_m2_per_mol_site is meaningless without "
                    "surface_model 'ddl' - remove it (no silent ignore)")
            if self.charging_reactions:
                raise ValueError(
                    "charging_reactions need surface_model 'ddl' - without "
                    "an electrostatic model the surface charge they create "
                    "has no effect (no silent ignore)")
        names = [rx.bound_species
                 for rx in [*self.surface_species, *self.charging_reactions]]
        if len(set(names)) != len(names):
            raise ValueError("surface reactions bind duplicate species")
        if (any(el in ("Na", "K") for el in self.elements)
                and not self.alkali_exchange):
            raise ValueError(
                "Na/K sorption requires alkali_exchange: true (and a C-S-H "
                "model without structural alkali uptake - CSHQ bundle only, "
                "engine-checked; the CNASH bundle binds alkalis "
                "thermodynamically and double-counting is refused)")
        return self


class TinnConfig(BaseModel):
    model_config = _STRICT
    binder: BinderRecipe
    psd: PSD = Field(default_factory=PSD.synthetic_default)
    # per-material particle-size distributions (PRD v2.2): keys are "clinker"
    # or an SCM id present in the recipe; materials absent from the map use
    # `psd`. None keeps every legacy config (and its hash) unchanged.
    material_psd: Optional[Dict[str, PSD]] = None
    # per-material particle shapes (PRD 1.2 rev.2): same keys as material_psd;
    # materials absent from the map stay spherical. None keeps every earlier
    # config (and its hash, and its RNG stream) unchanged.
    material_shape: Optional[Dict[str, ParticleShape]] = None
    # measured SCM glass compositions replacing the built-in ones (PRD 1.2
    # v3.0): keys are SCM phase ids. The channel layout is untouched, so
    # only the hash records the swap. None keeps every earlier config.
    scm_composition: Optional[Dict[str, ScmComposition]] = None
    w_c: float = Field(gt=0.0, le=2.0)
    temperature_K: float = Field(ge=273.15, le=372.15)
    kinetics: KineticsConfig
    rve: RVEConfig
    chemistry: ChemistryConfig
    schedule: ScheduleConfig
    # v4.0/RT chemistry-transport modes (PRD 4.6). None keeps every earlier
    # config hash (and the engine's exact legacy code path) unchanged.
    transport: Optional[TransportConfig] = None
    # Tier 1 (RT-S1): PHREEQC surface-sorption operator. None = no S stage,
    # bit-identical engine and unchanged hash (explicit pop in config_hash -
    # top-level optionals are NOT auto-swept).
    sorption: Optional[SorptionConfig] = None
    # coarse-tail volume fraction removed per PSD by truncate_to_grid, keyed
    # "__shared__" (the top-level psd) or the material_psd key — diagnostics
    # for the geometry report, never part of the hash/serialized payload
    _psd_truncation: Dict[str, float] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "TinnConfig":
        d_max_allowed_um = (self.rve.grid_size - RASTER_HALO_VOX) * self.rve.voxel_size_um
        for label, psd in (("__shared__", self.psd),
                           *((k, v)
                             for k, v in (self.material_psd or {}).items())):
            if psd.d_max_um <= d_max_allowed_um:
                continue
            if not psd.truncate_to_grid:
                raise ValueError(
                    f"largest {label} PSD diameter {psd.d_max_um} um exceeds "
                    f"the rasterizable maximum {d_max_allowed_um:.3f} um for a "
                    f"{self.rve.grid_size}^3 periodic RVE at "
                    f"{self.rve.voxel_size_um} um/voxel (set the PSD's "
                    f"truncate_to_grid to opt into CEMHYD3D-style truncation)"
                )
            # CEMHYD3D-style coarse-tail truncation: drop whole bins that
            # cannot be rasterized, renormalize the rest, and RECORD the
            # removed fraction (surfaced in the geometry report — PRD 1.2)
            kept = [b for b in psd.bins if b.d_hi_um <= d_max_allowed_um]
            if not kept:
                raise ValueError(
                    f"{label} PSD lies entirely above the rasterizable "
                    f"maximum {d_max_allowed_um:.3f} um - truncation would "
                    f"leave no particles")
            kept_vf = sum(b.volume_fraction for b in kept)
            if kept_vf <= 0.0:
                raise ValueError(
                    f"{label} PSD has no volume below the rasterizable "
                    f"maximum - truncation would leave no particles")
            self._psd_truncation[label] = 1.0 - kept_vf
            psd.bins = [PSDBin(d_lo_um=b.d_lo_um, d_hi_um=b.d_hi_um,
                               volume_fraction=b.volume_fraction / kept_vf)
                        for b in kept]
        if self.material_psd is not None:
            allowed = {"clinker", *SCM_PHASE_IDS, *SALT_PHASE_IDS}
            unknown = set(self.material_psd) - allowed
            if unknown:
                raise ValueError(
                    f"unknown material_psd keys {sorted(unknown)}; allowed: "
                    f"{sorted(allowed)}")
            for key in self.material_psd:
                if key != "clinker" and self.binder.mass_fractions.get(key, 0.0) <= 0.0:
                    raise ValueError(
                        f"material_psd[{key!r}] given but the recipe has no {key} mass")
        if self.material_shape is not None:
            allowed = {"clinker", *SCM_PHASE_IDS, *SALT_PHASE_IDS}
            unknown = set(self.material_shape) - allowed
            if unknown:
                raise ValueError(
                    f"unknown material_shape keys {sorted(unknown)}; allowed: "
                    f"{sorted(allowed)}")
            for key, shape in self.material_shape.items():
                if key != "clinker" and self.binder.mass_fractions.get(key, 0.0) <= 0.0:
                    raise ValueError(
                        f"material_shape[{key!r}] given but the recipe has no "
                        f"{key} mass")
                # the LONGEST semi-axis is what must fit the periodic RVE
                psd_eff = (self.material_psd or {}).get(key, self.psd)
                d_eff = psd_eff.d_max_um * shape.elongation()
                if d_eff > d_max_allowed_um:
                    raise ValueError(
                        f"material_shape[{key!r}] stretches the largest particle "
                        f"to {d_eff:.3f} um along its major axis, exceeding the "
                        f"rasterizable maximum {d_max_allowed_um:.3f} um")
        active = {p for p, f in self.binder.mass_fractions.items() if f > 0.0}
        if self.kinetics.kind == "tabulated":
            # salt carriers are solubility-controlled: the engine offers the
            # equilibrium whatever water reaches and GEMS decides (PRD 4.2
            # v3.0/E3). A tabulated alpha for them would be silently ignored,
            # so it is refused rather than accepted and dropped.
            scheduled_salts = set(self.kinetics.table.alpha) & set(SALT_PHASE_IDS)
            if scheduled_salts:
                raise ValueError(
                    f"tabulated kinetics carries alpha series for the soluble "
                    f"salt carriers {sorted(scheduled_salts)}, which have no "
                    f"rate law - their dissolution is decided by the GEMS "
                    f"equilibrium, so the schedule would be ignored")
            missing = (active - set(self.kinetics.table.alpha)) - set(SALT_PHASE_IDS)
            if missing:
                raise ValueError(
                    f"tabulated kinetics has no alpha series for binder phases "
                    f"{sorted(missing)} - every reacting phase needs a schedule"
                )
            horizon = self.kinetics.table.times_h[-1]
            if self.schedule.output_times_h[-1] > horizon:
                raise ValueError(
                    f"output time {self.schedule.output_times_h[-1]} h is beyond the "
                    f"tabulated kinetics horizon {horizon} h - no extrapolation"
                )
        if self.chemistry.backend == "stoichiometric":
            missing = active - set(self.chemistry.stoichiometric_rules)
            if missing:
                raise ValueError(
                    f"stoichiometric backend has no reaction rule for binder phases "
                    f"{sorted(missing)} - SCM glasses and the E3 soluble salt "
                    f"carriers dissolve into solution instead of precipitating a "
                    f"fixed product, so they require the gems3k backend"
                )
        if (self.transport is not None and self.transport.rate_limited()
                and self.chemistry.backend != "gems3k"):
            raise ValueError(
                "transport.exchange_tau_h rate-limits the re-equilibration of "
                "owned hydrates, which only the gems3k snapshot backend "
                "performs - with the stoichiometric backend it would be a "
                "silent no-op, so it is refused")
        if self.transport is not None:
            # scientific-review recommendation 1 (2026-08-20): a tau at or
            # below dt_min guarantees f = 1 at EVERY reachable dt - mode B
            # declared but permanently reverted to full re-equilibration.
            # Same silent-no-op class as above, refused. (tau between
            # dt_min and the cruise dt engages on halved retries and is a
            # legitimate, if unusual, configuration.)
            floor = self.schedule.dt_min_h
            tau_g = self.transport.exchange_tau_h
            if tau_g is not None and tau_g <= floor:
                raise ValueError(
                    f"exchange_tau_h {tau_g} h <= schedule.dt_min_h {floor} "
                    f"h keeps f = 1 at every reachable dt - mode B would "
                    f"silently run full re-equilibration")
            for key, val in (self.transport.exchange_tau_h_per_phase
                             or {}).items():
                if 0.0 < val <= floor:
                    raise ValueError(
                        f"exchange_tau_h_per_phase[{key!r}] = {val} h <= "
                        f"schedule.dt_min_h {floor} h keeps that channel at "
                        f"f = 1 always (use 0.0 for an explicit fully-"
                        f"offered channel)")
        if (self.transport is not None and self.transport.domains is not None
                and self.transport.domains.species is not None):
            # Tier 0 (PRD 4.6.4): the speciation source is the GEMS worker
            # response - the stoichiometric backend has none (no silent
            # empty-speciation run, diagnostics included)
            if self.chemistry.backend != "gems3k":
                raise ValueError(
                    "transport.domains.species needs the gems3k backend - "
                    "aqueous speciation comes from the GEMS responses")
            # RT-P0d: a solute-bearing reservoir is speciated ONCE at engine
            # init (GEMS aqueous-only equilibrium, frozen for the run, the
            # 0D suppression witness guarding it) - no config-level refusal
            # any more; a bath GEMS cannot speciate fails loudly at init.
        if self.transport is not None and self.transport.domains is not None:
            for name, t in zip(("tile_zyx[z]", "tile_zyx[y]", "tile_zyx[x]"),
                               self.transport.domains.tiles(
                                   self.rve.grid_size)):
                if self.rve.grid_size % t != 0:
                    raise ValueError(
                        f"transport.domains {name if self.transport.domains.tile_zyx else 'tile_vox'} "
                        f"= {t} does not divide the grid size "
                        f"{self.rve.grid_size} - tiles must wrap "
                        f"periodically")
        if (self.transport is not None
                and self.transport.boundary is not None
                and self.transport.boundary.start_h is not None
                and self.transport.boundary.start_h > 0.0
                and not any(abs(t - self.transport.boundary.start_h) <= 1e-9
                            for t in self.schedule.output_times_h)):
            raise ValueError(
                f"transport.boundary.start_h "
                f"{self.transport.boundary.start_h} must coincide with an "
                f"output time so no step straddles the sealed->exposed "
                f"switch")
        if self.sorption is not None:
            # RT-04B (review 2026-09-07): the S stage hands the reactor's
            # OWNED buffer amount to PHREEQC, but a per-phase rate limit on
            # that phase restricts what the R stage offers - the buffer
            # would reach mass the reaction transaction withholds. The
            # global exchange_tau_h case is refused by the engine already.
            per = (self.transport.exchange_tau_h_per_phase or {}
                   if self.transport is not None else {})
            if (self.sorption.buffer_phase is not None
                    and self.sorption.buffer_phase in per):
                raise ValueError(
                    f"sorption.buffer_phase {self.sorption.buffer_phase!r} "
                    f"cannot carry a transport.exchange_tau_h_per_phase entry "
                    f"- the S-stage buffer would use mass the rate-limited R "
                    f"offer withholds (refused, no silent inconsistency)")
            if self.chemistry.backend != "gems3k":
                raise ValueError(
                    "sorption needs the gems3k backend - sorbent sites come "
                    "from the C-S-H endmember ledger, which only the "
                    "snapshot backend maintains")
            if (self.sorption.alkali_exchange
                    and self.chemistry.gems_bundle_lst is None):
                raise ValueError(
                    "sorption.alkali_exchange needs an EXPLICIT "
                    "gems_bundle_lst (the omitted default is the CNASH "
                    "bundle, whose C-S-H binds alkalis thermodynamically - "
                    "surface exchange on top would double-count; the engine "
                    "re-checks the actual bundle)")
        if (self.transport is not None and self.transport.boundary is not None
                and self.chemistry.backend != "gems3k"):
            # the stoichiometric backend's solution ledger is identically
            # empty (residual = inventory, initially zeros): a bath is a
            # literal no-op or injects elements no chemistry can consume
            raise ValueError(
                "transport.boundary needs the gems3k backend - the "
                "stoichiometric backend carries no solution inventory, so "
                "a bath would be a silent no-op, which is refused")
        return self

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # the worker interpreter path is machine infrastructure, not physics —
        # excluding it keeps checkpoints restartable after an env move
        # (override at runtime with the TINN_GEMS_PYTHON environment variable)
        if payload.get("chemistry"):
            payload["chemistry"].pop("gems_worker_python", None)
            # RT-S3: undeclared GEMS exclusions keep the earlier hash
            for key in ("suppressed_species", "suppressed_phases"):
                if payload["chemistry"].get(key) is None:
                    payload["chemistry"].pop(key, None)
            # legacy hash compatibility: configs that omit the gel-porosity map
            # hashed the pre-CNASH default, and the CNASH entry is inert for
            # bundles without that phase — canonicalize the built-in default
            # back to the legacy form so every old checkpoint stays restartable
            if payload["chemistry"].get("gems_gel_porosity") == DEFAULT_GEMS_GEL_POROSITY:
                payload["chemistry"]["gems_gel_porosity"] = dict(_LEGACY_GEL_POROSITY_HASH_FORM)
        # absent material_psd stays out of the payload so every pre-v2.2
        # config keeps its hash
        if payload.get("material_psd") is None:
            payload.pop("material_psd", None)
        # same contract for material_shape (rev.2)
        if payload.get("material_shape") is None:
            payload.pop("material_shape", None)
        # and for the declared SCM compositions (v3.0): a config that inherits
        # the built-in glasses keeps its earlier hash
        if payload.get("scm_composition") is None:
            payload.pop("scm_composition", None)
        # RT-S1: a sorption-free config keeps its pre-Tier-1 hash
        if payload.get("sorption") is None:
            payload.pop("sorption", None)
        # RT-S2a: no_edl configs written before the ddl fields keep their
        # hash (material_psd pattern - unset optionals pop)
        sorp = payload.get("sorption")
        if sorp is not None:
            if sorp.get("specific_area_m2_per_mol_site") is None:
                sorp.pop("specific_area_m2_per_mol_site", None)
            if not sorp.get("charging_reactions"):
                sorp.pop("charging_reactions", None)
            if sorp.get("buffer_phase") is None:
                sorp.pop("buffer_phase", None)
            # RT-Cl-2: the optional second site pool follows the same rule
            if sorp.get("site_density_c_mol_per_mol") is None:
                sorp.pop("site_density_c_mol_per_mol", None)
        # Piecewise timesteps were added after checkpoint format v3. An absent
        # schedule keeps every legacy config/checkpoint hash unchanged.
        if payload.get("schedule", {}).get("dt_windows") is None:
            payload["schedule"].pop("dt_windows", None)
        # v4.0/RT transport section: EVERY None-valued field pops (generic on
        # purpose — a hard-coded name tuple would silently change every
        # mode-B hash the day RT-W2 adds `domains`, review finding), so
        # "transport": null, {}, and {"exchange_tau_h": null} all collapse
        # to the legacy hash — a config without RT modes is the same physics
        # it always was
        tr = payload.get("transport")
        if isinstance(tr, dict):
            for key in [k for k, v in tr.items() if v is None]:
                tr.pop(key)
            # nested RT sections follow the same rule: every None-valued
            # field pops, so later Optional additions (start_h, tile_zyx,
            # ...) never shift the hash of a config that does not use them
            for sub in ("domains", "boundary"):
                d = tr.get(sub)
                if isinstance(d, dict):
                    for key in [k for k, v in d.items() if v is None]:
                        d.pop(key)
                    # the species block nests one level deeper (RT-01:
                    # applied_charge_rtol None keeps every P0 hash)
                    sp = d.get("species")
                    if isinstance(sp, dict):
                        for key in [k for k, v in sp.items() if v is None]:
                            sp.pop(key)
        if not payload.get("transport"):
            payload.pop("transport", None)
        # PSD measured-input fields (rev.2): default-valued keys pop so every
        # bins-only legacy PSD keeps its hash; a truncated PSD hashes its
        # TRUNCATED bins plus the original measured input — physics-faithful

        def _canon_psd(d):
            if not isinstance(d, dict):
                return
            for key, default in (("cumulative", None), ("rosin_rammler", None),
                                 ("truncate_to_grid", False)):
                if d.get(key) == default:
                    d.pop(key, None)

        _canon_psd(payload.get("psd"))
        for v in (payload.get("material_psd") or {}).values():
            _canon_psd(v)
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def from_json_file(cls, path: str) -> "TinnConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))
