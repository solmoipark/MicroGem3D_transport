"""Pydantic configuration schema with unit/range validation and config hashing.

All models forbid unknown fields (PRD §5: no guessing, no silent fallback).
Field names carry units explicitly (_um, _K, _h).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import KINETIC_PHASE_IDS, default_registry

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


class PSD(BaseModel):
    model_config = _STRICT
    bins: List[PSDBin] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "PSD":
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

    @model_validator(mode="after")
    def _check(self) -> "KineticsConfig":
        if self.kind == "pk":
            if self.preset not in PK_PRESETS:
                raise ValueError(f"pk kinetics requires preset in {PK_PRESETS}, got {self.preset!r}")
            if self.table is not None:
                raise ValueError("pk kinetics does not take a table")
        else:
            if self.table is None:
                raise ValueError("tabulated kinetics requires a table")
            if self.preset is not None:
                raise ValueError("tabulated kinetics does not take a preset")
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


class ChemistryConfig(BaseModel):
    model_config = _STRICT
    backend: Literal["stoichiometric", "gems3k"]
    stoichiometric_rules: Dict[str, ReactionRule] = Field(default_factory=_default_rules)

    @model_validator(mode="after")
    def _check(self) -> "ChemistryConfig":
        unknown = set(self.stoichiometric_rules) - set(KINETIC_PHASE_IDS)
        if unknown:
            raise ValueError(f"stoichiometric rules for unknown phases: {sorted(unknown)}")
        return self


class RVEConfig(BaseModel):
    model_config = _STRICT
    grid_size: Literal[32, 64]
    voxel_size_um: float = Field(ge=0.5, le=1.0)
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


class TinnConfig(BaseModel):
    model_config = _STRICT
    binder: BinderRecipe
    psd: PSD = Field(default_factory=PSD.synthetic_default)
    w_c: float = Field(gt=0.0, le=2.0)
    temperature_K: float = Field(ge=273.15, le=372.15)
    kinetics: KineticsConfig
    rve: RVEConfig
    chemistry: ChemistryConfig
    schedule: ScheduleConfig

    @model_validator(mode="after")
    def _check(self) -> "TinnConfig":
        extent_um = self.rve.grid_size * self.rve.voxel_size_um
        if self.psd.d_max_um >= extent_um:
            raise ValueError(
                f"largest PSD diameter {self.psd.d_max_um} um must be smaller than the "
                f"periodic RVE extent {extent_um} um"
            )
        return self

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_json_file(cls, path: str) -> "TinnConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))
