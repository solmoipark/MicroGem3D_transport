"""Pydantic configuration schema with unit/range validation and config hashing.

All models forbid unknown fields (PRD §5: no guessing, no silent fallback).
Field names carry units explicitly (_um, _K, _h).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import (BaseModel, ConfigDict, Field, PrivateAttr,
                      field_validator, model_validator)

from .registry import (KINETIC_PHASE_IDS, SALT_PHASE_IDS, SCM_PHASE_IDS,
                       default_registry)

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

    @model_validator(mode="after")
    def _check(self) -> "ChemistryConfig":
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
                or self.gems_gel_porosity is not None):
            raise ValueError("gems_* fields only apply to the gems3k backend")
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


class ScheduleConfig(BaseModel):
    model_config = _STRICT
    output_times_h: List[float] = Field(min_length=1)
    dt_initial_h: float = Field(default=0.01, gt=0.0)
    dt_min_h: float = Field(default=1e-4, gt=0.0)
    max_retries: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "ScheduleConfig":
        if self.output_times_h[0] <= 0.0:
            raise ValueError("output times must be > 0")
        for a, b in zip(self.output_times_h, self.output_times_h[1:]):
            if b <= a:
                raise ValueError("output_times_h must be strictly increasing")
        if self.dt_min_h > self.dt_initial_h:
            raise ValueError("dt_min_h must be <= dt_initial_h")
        return self


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


class TransportConfig(BaseModel):
    """v4.0/RT chemistry-transport modes (PRD 1.4/4.6). Absence of the
    section — or None in every field — preserves the exact current engine,
    bit for bit, and every earlier config hash."""
    model_config = _STRICT
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
        return self

    def rate_limited(self) -> bool:
        return (self.exchange_tau_h is not None
                or bool(self.exchange_tau_h_per_phase))


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
        return self

    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # the worker interpreter path is machine infrastructure, not physics —
        # excluding it keeps checkpoints restartable after an env move
        # (override at runtime with the TINN_GEMS_PYTHON environment variable)
        if payload.get("chemistry"):
            payload["chemistry"].pop("gems_worker_python", None)
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
