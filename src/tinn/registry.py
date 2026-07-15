"""Phase/component registry: IDs, formulas, molar masses, molar volumes + basis, gel porosity.

Molar quantities are authoritative; densities are derived (g/cm3 = molar mass / molar volume)
except for the placement-only "inert" pseudo-phase which carries an explicit density.
Unknown phases, elements, or a missing volume basis are hard errors — no guessing (PRD §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ATOMIC_MASS_G_MOL: Dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
    "Na": 22.9898,
    "Mg": 24.305,
    "Al": 26.9815,
    "Si": 28.0855,
    "S": 32.06,
    "K": 39.0983,
    "Ca": 40.078,
    "Fe": 55.845,
}

CLINKER_PHASE_IDS: Tuple[str, ...] = ("C3S", "C2S", "C3A", "C4AF")
# SCM glasses (PRD v2.1): compositions/densities from InverseGems materials.yaml,
# reaction schedules are logistic curves (kinetics.SCM_LOGISTIC_PRESETS)
SCM_PHASE_IDS: Tuple[str, ...] = ("slag", "fly_ash", "metakaolin", "silica_fume")
KINETIC_PHASE_IDS: Tuple[str, ...] = CLINKER_PHASE_IDS + SCM_PHASE_IDS
INERT_PHASE_ID = "inert"
# Order of solid channels in the dense anhydrous_fraction array (fixed).
SOLID_PHASE_IDS: Tuple[str, ...] = KINETIC_PHASE_IDS + (INERT_PHASE_ID,)
# Hydrate channels of the STOICHIOMETRIC backend (a gems3k run derives its
# channel set from the bundle's solid phases; the run's channels live in
# SimulationState.hydrate_ids and in the checkpoint header).
HYDRATE_PHASE_IDS: Tuple[str, ...] = ("CSH", "CH", "C3AH6", "FH3")
# Order of the element ledger vector (fixed; covers the PC bundle's elements).
ELEMENT_IDS: Tuple[str, ...] = ("Ca", "Si", "Al", "Fe", "S", "Na", "K", "Mg",
                                "C", "H", "O")


def element_vector(formula: dict, mol: float = 1.0):
    """Element mol vector over ELEMENT_IDS for `mol` of a formula unit."""
    import numpy as np
    v = np.zeros(len(ELEMENT_IDS))
    for el, count in formula.items():
        v[ELEMENT_IDS.index(el)] += count * mol
    return v

VALID_BASIS = ("solid_skeleton", "bulk_envelope")
VALID_KINDS = ("clinker", "scm", "hydrate", "liquid", "inert")

# oxide -> element counts for SCM glass composition conversion
_OXIDES = {
    "SiO2": {"Si": 1, "O": 2}, "Al2O3": {"Al": 2, "O": 3},
    "Fe2O3": {"Fe": 2, "O": 3}, "CaO": {"Ca": 1, "O": 1},
    "MgO": {"Mg": 1, "O": 1}, "SO3": {"S": 1, "O": 3},
    "Na2O": {"Na": 2, "O": 1}, "K2O": {"K": 2, "O": 1},
}


def _oxide_molar_mass(oxide: str) -> float:
    return sum(ATOMIC_MASS_G_MOL[el] * n for el, n in _OXIDES[oxide].items())


def scm_entry(phase_id: str, oxide_mass_percent: dict, density_g_cm3: float,
              gel_porosity: float = 0.0) -> "PhaseEntry":
    """SCM glass as a registry phase: the formula unit is the LISTED oxides per
    100 g of material (measured composition, never invented; unlisted residue
    like LOI is simply absent). Molar volume follows from the material density,
    so density_g_cm3 round-trips exactly."""
    formula: dict = {}
    for oxide, wt in oxide_mass_percent.items():
        if oxide not in _OXIDES:
            raise RegistryError(f"{phase_id}: unknown oxide {oxide!r}")
        mol = wt / _oxide_molar_mass(oxide)  # mol oxide per 100 g material
        for el, n in _OXIDES[oxide].items():
            formula[el] = formula.get(el, 0.0) + mol * n
    molar_mass = sum(ATOMIC_MASS_G_MOL[el] * n for el, n in formula.items())
    return PhaseEntry(phase_id, "scm", formula, molar_mass / density_g_cm3,
                      "solid_skeleton", gel_porosity=gel_porosity)


class RegistryError(ValueError):
    """Invalid or missing registry data (hard error, never guessed around)."""


@dataclass(frozen=True)
class PhaseEntry:
    phase_id: str
    kind: str  # clinker | hydrate | liquid | inert
    formula: Optional[Dict[str, float]] = None  # element -> atoms per formula unit
    molar_volume_cm3: Optional[float] = None
    basis: Optional[str] = None  # required for solids (clinker/hydrate)
    gel_porosity: float = 0.0
    density_override_g_cm3: Optional[float] = None  # only for formula-less pseudo-phases

    @property
    def molar_mass_g_mol(self) -> Optional[float]:
        if self.formula is None:
            return None
        return sum(ATOMIC_MASS_G_MOL[el] * n for el, n in self.formula.items())

    @property
    def density_g_cm3(self) -> float:
        if self.density_override_g_cm3 is not None:
            return self.density_override_g_cm3
        assert self.formula is not None and self.molar_volume_cm3 is not None
        return self.molar_mass_g_mol / self.molar_volume_cm3

    @property
    def skeleton_molar_volume_cm3(self) -> float:
        if self.basis == "solid_skeleton":
            return self.molar_volume_cm3
        if self.basis == "bulk_envelope":
            return self.molar_volume_cm3 * (1.0 - self.gel_porosity)
        raise RegistryError(f"{self.phase_id}: no volume basis")

    @property
    def envelope_molar_volume_cm3(self) -> float:
        """Bulk envelope = skeleton / (1 - gel porosity) (PRD §4.4)."""
        if self.basis == "bulk_envelope":
            return self.molar_volume_cm3
        if self.basis == "solid_skeleton":
            return self.molar_volume_cm3 / (1.0 - self.gel_porosity)
        raise RegistryError(f"{self.phase_id}: no volume basis")


class Registry:
    def __init__(self, entries: Tuple[PhaseEntry, ...]):
        self._entries: Dict[str, PhaseEntry] = {}
        for e in entries:
            if e.phase_id in self._entries:
                raise RegistryError(f"duplicate phase id: {e.phase_id!r}")
            if e.kind not in VALID_KINDS:
                raise RegistryError(f"{e.phase_id}: unknown kind {e.kind!r}")
            if e.kind in ("clinker", "scm", "hydrate"):
                if e.basis not in VALID_BASIS:
                    raise RegistryError(
                        f"{e.phase_id}: solid phase requires basis in {VALID_BASIS}, "
                        f"got {e.basis!r}"
                    )
            if e.formula is not None:
                unknown = set(e.formula) - set(ATOMIC_MASS_G_MOL)
                if unknown:
                    raise RegistryError(f"{e.phase_id}: unknown elements {sorted(unknown)}")
                if e.molar_volume_cm3 is None or e.molar_volume_cm3 <= 0:
                    raise RegistryError(f"{e.phase_id}: molar_volume_cm3 required and > 0")
            elif e.density_override_g_cm3 is None or e.density_override_g_cm3 <= 0:
                raise RegistryError(
                    f"{e.phase_id}: needs a formula+molar volume or an explicit density"
                )
            if not (0.0 <= e.gel_porosity < 1.0):
                raise RegistryError(f"{e.phase_id}: gel_porosity must be in [0, 1)")
            self._entries[e.phase_id] = e

    def get(self, phase_id: str) -> PhaseEntry:
        try:
            return self._entries[phase_id]
        except KeyError:
            raise KeyError(
                f"unknown phase {phase_id!r}; registered: {sorted(self._entries)}"
            ) from None

    def __contains__(self, phase_id: str) -> bool:
        return phase_id in self._entries

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._entries)


def default_registry() -> Registry:
    return Registry((
        # -- clinker phases (anhydrous, no gel porosity) --
        PhaseEntry("C3S", "clinker", {"Ca": 3, "Si": 1, "O": 5}, 72.4, "solid_skeleton"),
        PhaseEntry("C2S", "clinker", {"Ca": 2, "Si": 1, "O": 4}, 52.4, "solid_skeleton"),
        PhaseEntry("C3A", "clinker", {"Ca": 3, "Al": 2, "O": 6}, 89.1, "solid_skeleton"),
        PhaseEntry("C4AF", "clinker", {"Ca": 4, "Al": 2, "Fe": 2, "O": 10}, 130.3,
                   "solid_skeleton"),
        # -- SCM glasses (PRD v2.1; oxide wt% + densities from InverseGems
        #    configs/materials.yaml — measured compositions, nothing invented) --
        scm_entry("slag", {"SiO2": 36.49, "Al2O3": 12.26, "CaO": 41.79,
                           "MgO": 7.48, "SO3": 1.98}, 2.90),
        scm_entry("fly_ash", {"SiO2": 51.8, "Al2O3": 23.4, "Fe2O3": 7.2,
                              "CaO": 10.8, "MgO": 2.7, "SO3": 1.1,
                              "Na2O": 1.3, "K2O": 1.6}, 2.20),
        scm_entry("metakaolin", {"SiO2": 54.1, "Al2O3": 43.6, "Fe2O3": 1.1,
                                 "CaO": 0.2, "MgO": 0.2, "SO3": 0.1,
                                 "Na2O": 0.1, "K2O": 0.5}, 2.50),
        scm_entry("silica_fume", {"SiO2": 99.3, "Al2O3": 0.1, "CaO": 0.1,
                                  "MgO": 0.1, "Na2O": 0.1, "K2O": 0.2}, 2.20),
        # Unassigned clinker residual: placement-only inert solid, no chemistry invented.
        PhaseEntry(INERT_PHASE_ID, "inert", density_override_g_cm3=3.15),
        # -- liquid --
        PhaseEntry("H2O", "liquid", {"H": 2, "O": 1}, 18.05),
        # -- hydrates (for the stoichiometric backend, M1+) --
        # CSH = C1.7SH4 (1.7CaO.SiO2.4H2O); skeleton volume, gel porosity 0.28
        PhaseEntry("CSH", "hydrate", {"Ca": 1.7, "Si": 1, "O": 7.7, "H": 8}, 107.3,
                   "solid_skeleton", gel_porosity=0.28),
        PhaseEntry("CH", "hydrate", {"Ca": 1, "O": 2, "H": 2}, 33.1, "solid_skeleton"),
        # hydrogarnet C3AH6 = Ca3Al2(OH)12
        PhaseEntry("C3AH6", "hydrate", {"Ca": 3, "Al": 2, "O": 12, "H": 12}, 150.1,
                   "solid_skeleton"),
        # FH3 = Fe2O3.3H2O
        PhaseEntry("FH3", "hydrate", {"Fe": 2, "O": 6, "H": 6}, 69.8, "solid_skeleton"),
    ))
