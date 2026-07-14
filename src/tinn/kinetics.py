"""KineticsModel: returns target per-phase alpha for a time — no geometry/chemistry involved.

M1 ships TabulatedKinetics; ParrotKilloh presets arrive with M2.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import KineticsConfig, TinnConfig
from .registry import KINETIC_PHASE_IDS


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


def make_kinetics(config: TinnConfig) -> KineticsModel:
    if config.kinetics.kind == "tabulated":
        return TabulatedKinetics(config.kinetics)
    raise NotImplementedError(
        f"kinetics preset {config.kinetics.preset!r} arrives with milestone M2"
    )
