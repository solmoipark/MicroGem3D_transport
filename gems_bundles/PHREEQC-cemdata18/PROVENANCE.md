# gems_bundles/PHREEQC-cemdata18/cemdata18.dat — provenance

- **What**: CEMDATA18 thermodynamic database in PHREEQC format. Official
  Empa/PSI export — the file header states it was "Exported to PHREEQC format
  using ThermoMatch (https://bitbucket.org/gems4/thermomatch) reactions
  generator and export modules", based on CEMDATA18 version 01 (09.10.2017)
  and PSI/Nagra 12/07, with named contacts Barbara Lothenbach (Empa) and
  G. Dan Miron (PSI). Internal update log runs to 16.01.2019.
- **Retrieved**: 2026-09-01, from the HYDCEM v4.01 distribution which vendors
  this file unmodified:
  https://raw.githubusercontent.com/BrucieHolmes/HYDCEMv4.01_Executable/main/DataHYDCEM/cemdata18.dat
  (HYDCEM: N. Holmes et al., cement hydration model coupled to PHREEQC.)
- **sha256**: 48542d7bf5086c8865c709d238a68bd44174fea2471aac0d659a6e9c21c37485
- **Cite**: B. Lothenbach, D.A. Kulik, T. Matschei, M. Balonis, L. Baquerizo,
  B. Dilnesa, G.D. Miron, R.J. Myers, "Cemdata18: A chemical thermodynamic
  database for hydrated Portland cements and alkali-activated materials",
  Cement and Concrete Research 115 (2019) 472–506.
- **Validity**: 0–100 °C (three-term analytical log K expressions; the GEMS
  bundles integrate Cp directly, so off-25 °C comparisons test the fit too).
- **Model content**: aqueous model + PHASES including the CSHQ endmembers
  (CSHQ-TobH/TobD/JenH/JenD), the CNASH_ss endmembers (TobH/T2C/T5C-CNASHss,
  5CA, 5CNA, INFCA, INFCN, INFCNA), ECSH1/ECSH2, AFt/AFm series, hydrogarnets,
  hydrotalcite, salts. Solid solutions are NOT pre-declared — the input file
  assembles them (see scripts/spike_phreeqc_crosscheck.py).
- **License note**: distributed by Empa for research use; for anything beyond
  internal research/publication cite the paper and check with Empa
  (barbara.lothenbach@empa.ch). This copy is vendored for reproducibility of
  the cross-check spike only.
- **Audit rule**: treat like the GEMS bundles — the file is read-only input;
  scripts must never mutate it (sha256 above is the check).
