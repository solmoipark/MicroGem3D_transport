# PC-Cl bundle (GEMS3K export, 2026-09-02)

Source: `C:\Users\solmo\GEMS data files\20260902 TINN_v4\MySystem-*` — GEM-Selektor
project "TINN_v4" (ID_key `TINN_1 G MySystem`), cemdata18, exported by the user on
2026-09-02 with **NaCl as the background electrolyte**. Files copied verbatim
(`MySystem-dat.lst` references them by name).

What it adds over `gems_bundles/PC` (measured from the DCH):
- independent components: Al C Ca **Cl** Fe H K Mg Na O S Si Zz (nIC 13; PC has no Cl)
- aqueous Cl species: Cl-, ClO4-, FeCl+, FeCl+2, FeCl2+, FeCl3@
- chloride phases: **Friedels** (C4AClH10), **Kuzels** (C4AsClH12)
- CSHQ solid solution with 6 endmembers: JenD/JenH/TobD/TobH + **KSiOH/NaSiOH**
  (alkali uptake endmembers, Si 0.2 each) — site densities for the sorption
  operator must list them (0.4766 x Si)
- hydrates-only (no clinker phases): the engine's suppression filter handles that
  (P0a fix); Si-hydrogarnet C3(AF)S0.84H present and NOT suppressed (user decision)
- MgAl-OH-LDH (3 DCs) replaces hydrotalc-pyro; Na-oxide carrier present; no zeolites

Exported equilibrium (dbr-0-0000, 25 C, B = user's MySystem bulk): pH 12.607,
I 0.0597, Portlandite 0.0200 mol, CSHQ 0.0206, brucite 0.00228, AFm/AFt traces;
Friedels/Kuzels absent at this Cl load (1.7e-5 mol) — the 0D smoke reproduces
this state as the bundle's acceptance gate.
