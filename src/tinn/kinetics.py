"""KineticsModel: returns target per-phase alpha for a time — no geometry/chemistry.

TabulatedKinetics interpolates a validated alpha(t) table. ParrotKilloh integrates
the four-phase Parrot--Killoh model (PRD §4.1) with two published presets:

  pk_elakneswaran_2018  T0=293.15 K, Blaine ratio scales ALL rates
                        (Elakneswaran et al. 2018, doi:10.3390/app8122597)
  pk_cemgems_2021       T0=298.15 K, Blaine ratio scales nucleation/growth only
                        (Kulik et al. 2021, doi:10.21809/rilemtechlett.2021.140)

Canonical units: rates day^-1, temperature K, activation energy J/mol,
Blaine m^2/kg (published reference 385). alpha_at(t) is a pure function of t
(each call integrates 0 -> t with the configured substep), so straight runs and
checkpoint restarts see bit-identical targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import numpy as np

from .config import KineticsConfig, TinnConfig
from .registry import CLINKER_PHASE_IDS, KINETIC_PHASE_IDS, SCM_PHASE_IDS

GAS_CONSTANT_J_MOL_K = 8.314
REFERENCE_BLAINE_M2_KG = 385.0
WATER_RETARDATION_SLOPE = 3.333
RH_CUTOFF = 0.55

# SCM reaction schedules alpha(t) = D + (A - D) / (1 + (t/C)^B)^G, t in days
# (PRD v2.1; parameters from InverseGems configs/scm_reaction.yaml — the D here
# is the REFERENCE-blend value, rescaled per recipe by the availability modifier)
SCM_LOGISTIC_PRESETS: Dict[str, Tuple[float, float, float, float, float]] = {
    #             A     B      C     D     G
    "slag":        (0.0, 0.75, 20.0, 0.55, 1.0),
    "fly_ash":     (0.0, 1.05, 35.0, 0.40, 1.0),
    "metakaolin":  (0.0, 0.95, 5.0, 0.55, 1.0),
    "silica_fume": (0.0, 0.80, 3.0, 0.85, 1.0),
}

# CH-availability modifier (InverseGems configs/c3s_c2s_availability.yaml,
# availability_modifier.py — ON by default in the source pipeline):
# D_eff = clamp(min(absolute_max_D, D_ref * R^eta), 0, 1) with
# R = availability(mix)/availability(reference), availability = supply/demand,
# supply = C3S mass + 0.30 * C2S mass, demand = sum SCM mass * demand_coeff.
SCM_C2S_WEIGHT = 0.30
SCM_DEMAND_COEFF = {"slag": 0.35, "fly_ash": 0.75, "metakaolin": 1.00,
                    "silica_fume": 1.20}
SCM_ETA = {"slag": 0.35, "fly_ash": 0.60, "metakaolin": 0.90, "silica_fume": 1.00}
SCM_ABSOLUTE_MAX_D = {"slag": 0.75, "fly_ash": 0.60, "metakaolin": 0.95,
                      "silica_fume": 0.95}
# reference system: OPC 60 / SCM 40 with InverseGems' default OPC Bogue
# (C3S 66.219 %, C2S 8.3097 % of OPC): 60*(0.66219 + 0.3*0.083097)/40
SCM_REFERENCE_AVAILABILITY = 60.0 * (0.66219 + SCM_C2S_WEIGHT * 0.083097) / 40.0


def scm_effective_params(phase_mass_fractions: Dict[str, float]
                         ) -> Dict[str, Tuple[float, float, float, float, float]]:
    """Per-recipe SCM logistic parameters with the CH-availability-scaled D."""
    from .registry import SCM_PHASE_IDS
    present = {p: phase_mass_fractions.get(p, 0.0) for p in SCM_PHASE_IDS
               if phase_mass_fractions.get(p, 0.0) > 0.0}
    effective = dict(SCM_LOGISTIC_PRESETS)
    if not present:
        return effective
    supply = (phase_mass_fractions.get("C3S", 0.0)
              + SCM_C2S_WEIGHT * phase_mass_fractions.get("C2S", 0.0))
    demand = sum(m * SCM_DEMAND_COEFF[p] for p, m in present.items())
    r = 0.0
    if demand > 0.0:
        r = max(0.0, (supply / demand) / SCM_REFERENCE_AVAILABILITY)
    for p in present:
        a, b, c, d_ref, g = SCM_LOGISTIC_PRESETS[p]
        d_eff = min(SCM_ABSOLUTE_MAX_D[p], d_ref * r ** SCM_ETA[p])
        effective[p] = (a, b, c, max(0.0, min(1.0, d_eff)), g)
    return effective


def scm_alpha(t_days: float, params: Tuple[float, float, float, float, float]) -> float:
    a, b, c, d, g = params
    if t_days <= 0.0:
        return max(0.0, min(1.0, a))
    alpha = d + (a - d) / (1.0 + (t_days / c) ** b) ** g
    return max(0.0, min(1.0, alpha))


class KineticsModel(Protocol):
    def alpha_at(self, t_h: float) -> np.ndarray:
        """Target alpha vector over KINETIC_PHASE_IDS at absolute time t_h."""
        ...


class TabulatedKinetics:
    """Monotone piecewise-linear interpolation of a validated alpha(t) table.

    Phases absent from the table never react (config validation guarantees every
    reacting binder phase is present). Times beyond the table horizon are rejected
    at config validation; before the first knot, alpha holds the first value.
    """

    def __init__(self, kin: KineticsConfig):
        assert kin.kind == "tabulated" and kin.table is not None
        self._times = np.asarray(kin.table.times_h, dtype=np.float64)
        self._alpha = np.zeros((len(KINETIC_PHASE_IDS), len(self._times)))
        for i, p in enumerate(KINETIC_PHASE_IDS):
            if p in kin.table.alpha:
                self._alpha[i] = np.asarray(kin.table.alpha[p], dtype=np.float64)

    def alpha_at(self, t_h: float) -> np.ndarray:
        if t_h > self._times[-1] + 1e-9:
            raise ValueError(
                f"t={t_h} h is beyond the tabulated horizon {self._times[-1]} h"
            )
        return np.array([np.interp(t_h, self._times, self._alpha[i])
                         for i in range(self._alpha.shape[0])])


# --------------------------------------------------------------------- Parrot-Killoh

@dataclass(frozen=True)
class PKPreset:
    name: str
    # per-phase rows in KINETIC_PHASE_IDS order: (K1, N1, K2, K3, N3, H, Ea_J_mol)
    params: Tuple[Tuple[float, float, float, float, float, float, float], ...]
    reference_temperature_k: float
    surface_scales_all_rates: bool
    source: str


PK_PRESETS: Dict[str, PKPreset] = {
    "pk_elakneswaran_2018": PKPreset(
        name="pk_elakneswaran_2018",
        params=(
            (1.5, 0.70, 0.050, 1.10, 3.3, 1.80, 41570.0),   # C3S
            (0.5, 1.00, 0.006, 0.20, 5.0, 1.35, 20785.0),   # C2S
            (1.0, 0.85, 0.040, 1.00, 3.2, 1.60, 54040.0),   # C3A
            (0.37, 0.70, 0.015, 0.40, 3.7, 1.45, 34087.0),  # C4AF
        ),
        reference_temperature_k=293.15,
        surface_scales_all_rates=True,
        source="Elakneswaran, Owaki, Nawa (2018), doi:10.3390/app8122597, Appendix A.2",
    ),
    "pk_cemgems_2021": PKPreset(
        name="pk_cemgems_2021",
        params=(
            (1.5, 0.70, 0.050, 1.10, 3.3, 2.00, 41570.0),   # C3S
            (0.5, 1.00, 0.020, 0.70, 5.0, 1.55, 20785.0),   # C2S
            (1.0, 0.85, 0.040, 1.00, 3.2, 1.80, 54040.0),   # C3A
            (0.37, 0.70, 0.020, 0.40, 3.7, 1.65, 34087.0),  # C4AF
        ),
        reference_temperature_k=298.15,
        surface_scales_all_rates=False,
        source="Kulik et al. (2021), doi:10.21809/rilemtechlett.2021.140, Table SB1",
    ),
}


class ParrotKilloh:
    """Explicit-Euler four-phase P&K integration (PRD §4.1).

    The low-water retardation uses the mass-weighted TOTAL clinker alpha
    (sum f_m alpha_m / sum f_m over the four P&K phases only), applied after
    selecting the controlling rate, with the bracket clipped to [0, 1] so an
    Euler overshoot can never make hydration re-accelerate.
    """

    def __init__(self, preset_name: str, w_c: float, temperature_k: float,
                 blaine_m2_kg: float, phase_mass_fractions: Dict[str, float],
                 relative_humidity: float = 1.0,
                 alpha_seed: float = 1e-8, max_substep_days: float = 0.01):
        if preset_name not in PK_PRESETS:
            raise ValueError(
                f"unknown P&K preset {preset_name!r}; choose one of {sorted(PK_PRESETS)}")
        for name, value in (("w_c", w_c), ("temperature_k", temperature_k),
                            ("blaine_m2_kg", blaine_m2_kg),
                            ("max_substep_days", max_substep_days)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        p = PK_PRESETS[preset_name]
        self.preset = p
        rows = np.asarray(p.params)  # 4 clinker rows; SCM phases use logistic curves
        self._k1, self._n1, self._k2, self._k3, self._n3, self._h, self._ea = rows.T
        self.w_c = w_c
        self.temperature_k = temperature_k
        self.blaine_ratio = blaine_m2_kg / REFERENCE_BLAINE_M2_KG
        self.relative_humidity = relative_humidity
        self.alpha_seed = alpha_seed
        self.max_substep_days = max_substep_days
        self._weights = np.array([phase_mass_fractions.get(pid, 0.0)
                                  for pid in KINETIC_PHASE_IDS])
        if self._weights.sum() <= 0.0:
            raise ValueError("P&K needs at least one kinetic phase with mass")
        self._n_clinker = len(CLINKER_PHASE_IDS)
        # availability-scaled D per SCM (recipe-dependent, InverseGems default)
        effective = scm_effective_params(phase_mass_fractions)
        self._scm_params = [effective[pid] for pid in SCM_PHASE_IDS]
        self.scm_effective_d = {pid: effective[pid][3] for pid in SCM_PHASE_IDS}
        # fixed-grid prefix memo for alpha_at: {n_full: (clinker_state, t)}
        self._prefix_cache: Dict[int, tuple] = {}

    # -- correction factors ------------------------------------------------
    def humidity_factor(self) -> float:
        if self.relative_humidity <= RH_CUTOFF:
            return 0.0
        return ((self.relative_humidity - RH_CUTOFF) / (1.0 - RH_CUTOFF)) ** 4

    def temperature_factors(self) -> np.ndarray:
        t0 = self.preset.reference_temperature_k
        return np.exp((self._ea / GAS_CONSTANT_J_MOL_K)
                      * (1.0 / t0 - 1.0 / self.temperature_k))

    def water_retardation_factors(self, total_alpha: float) -> np.ndarray:
        """Per-CLINKER-phase factors (length 4; SCM schedules are closed-form)."""
        threshold = self._h * self.w_c
        factors = np.ones(len(self._h))
        active = total_alpha > threshold
        bracket = 1.0 + WATER_RETARDATION_SLOPE * (threshold - total_alpha)
        factors[active] = np.clip(bracket[active], 0.0, 1.0) ** 4
        return factors

    def total_clinker_alpha(self, alpha: np.ndarray) -> float:
        """Mass-weighted alpha over the 4 P&K clinker phases only (PRD 4.1)."""
        n = self._n_clinker
        w = self._weights[:n]
        if w.sum() <= 0.0:
            return 0.0
        return float((alpha[:n] * w).sum() / w.sum())

    # -- rates ---------------------------------------------------------------
    def rate_components(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Corrected (R_ng, R_df, R_hs) in day^-1 (without water retardation)."""
        a = np.clip(alpha, self.alpha_seed, 1.0 - np.finfo(np.float64).eps)
        remaining = 1.0 - a
        transformed = -np.log1p(-a)
        r_ng = (self._k1 / self._n1) * remaining * transformed ** (1.0 - self._n1)
        r_df = self._k2 * remaining ** (2.0 / 3.0) / (1.0 - np.cbrt(remaining))
        r_hs = self._k3 * remaining ** self._n3

        common = self.humidity_factor() * self.temperature_factors()
        r_ng, r_df, r_hs = r_ng * common, r_df * common, r_hs * common
        if self.preset.surface_scales_all_rates:
            r_ng, r_df, r_hs = (r_ng * self.blaine_ratio, r_df * self.blaine_ratio,
                                r_hs * self.blaine_ratio)
        else:
            r_ng = r_ng * self.blaine_ratio
        return r_ng, r_df, r_hs

    def controlling_rates(self, alpha: np.ndarray) -> np.ndarray:
        """Corrected min(R_ng, R_df, R_hs) in day^-1 (without water retardation)."""
        r_ng, r_df, r_hs = self.rate_components(alpha)
        # every component must be finite BEFORE selection — an inf in a
        # non-controlling rate signals a degenerate configuration, not health
        for comp in (r_ng, r_df, r_hs):
            if not np.all(np.isfinite(comp)):
                raise FloatingPointError("P&K rate evaluation produced a non-finite value")
        rates = np.minimum(np.minimum(r_ng, r_df), r_hs)
        rates[np.asarray(alpha) >= 1.0] = 0.0
        return rates

    def _step(self, clinker_alpha: np.ndarray, t_days: float,
              dt_days: float) -> np.ndarray:
        """One Euler step for the 4 clinker phases (rate machinery is
        clinker-length; SCM schedules are closed-form and never enter here)."""
        full = np.zeros(len(KINETIC_PHASE_IDS))
        full[:self._n_clinker] = clinker_alpha
        rates = self.controlling_rates(clinker_alpha)
        water = self.water_retardation_factors(self.total_clinker_alpha(full))
        return np.clip(clinker_alpha + dt_days * rates * water, clinker_alpha, 1.0)

    def alpha_at(self, t_h: float) -> np.ndarray:
        """Pure function of t, integrated on a FIXED absolute grid (full
        max_substep_days steps plus one remainder step). Different query times
        share the same grid prefix, so targets do not jitter between adjacent
        engine steps and exact substep multiples are float-stable."""
        t_days = t_h / 24.0
        alpha = np.zeros(len(KINETIC_PHASE_IDS))
        if t_days <= 0.0:
            return alpha
        h = self.max_substep_days
        n_full = int(math.floor(t_days / h + 1e-9))
        remainder = t_days - n_full * h
        # fixed-grid prefix memo: the SAME _step chain (identical float ops in
        # identical order, including the accumulated t), just not recomputed
        # from zero on every query — late-age queries were integrating tens of
        # thousands of substeps per call. Bitwise-neutral by construction;
        # restarts rebuild the cache and land on the same values.
        cache = self._prefix_cache
        best = -1
        for n in cache:
            if best < n <= n_full:
                best = n
        if best >= 0:
            clinker, t = cache[best]
            clinker = clinker.copy()
        else:
            clinker = np.zeros(self._n_clinker)
            t = 0.0
            best = 0
        for _ in range(best, n_full):
            clinker = self._step(clinker, t, h)
            t += h
        if n_full not in cache:
            cache[n_full] = (clinker.copy(), t)
            if len(cache) > 16:
                cache.pop(min(cache))
        if remainder > 1e-12 * max(1.0, t_days):
            clinker = self._step(clinker, t, remainder)
        alpha[:self._n_clinker] = clinker
        # SCM schedules are closed-form logistic curves (PRD v2.1)
        for j, params in enumerate(self._scm_params):
            alpha[self._n_clinker + j] = scm_alpha(t_days, params)
        # phases with no mass in the recipe have no meaningful alpha target
        return np.where(self._weights > 0.0, alpha, 0.0)


def make_kinetics(config: TinnConfig) -> KineticsModel:
    kin = config.kinetics
    if kin.kind == "tabulated":
        return TabulatedKinetics(kin)
    return ParrotKilloh(
        preset_name=kin.preset,
        w_c=config.w_c,
        temperature_k=config.temperature_K,
        blaine_m2_kg=kin.blaine_m2_kg,
        phase_mass_fractions=config.binder.mass_fractions,
        alpha_seed=kin.pk_alpha_seed,
        max_substep_days=kin.pk_max_substep_days,
    )
