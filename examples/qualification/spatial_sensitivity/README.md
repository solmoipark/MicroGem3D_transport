# Section 4.1 spatial-sensitivity matrix

All cases preserve the sealed, initially saturated OPC engineering baseline:
pure water, w/c 0.50, 20 °C, 1 bar, Parrot–Killoh kinetics, and the frozen
PC/Cemdata18.1 xGEMS bundle. The production timestep is fixed at 0.5 h.

## Four-level axes

- Domain-size sensitivity at 1.0 µm: 32³, 64³, 96³, and 128³.
- Voxel sensitivity at a fixed 32 µm domain: 32³ @ 1.0 µm,
  64³ @ 0.5 µm, 128³ @ 0.25 µm, and 320³ @ 0.1 µm.
- Stochastic sensitivity at the 64 µm production candidate:
  64³ @ 1.0 µm with four frozen independent seeds.
- Screening output ages: 1, 3, and 7 days.
- Selective extension age: 28 days.

The earlier `q32_fixed_dt05_log_28d` result has no exact 72 h checkpoint.
The 32³ @ 1.0 µm, seed 20260731 reference is therefore rerun with the same
four output ages as every other coupled case; no temporal interpolation is used.

The three additional stochastic seeds were generated once with
`numpy.random.SeedSequence(20260731).generate_state(3)` and then frozen:
2857866572, 1001334556, and 3821638640.

The 0.1 µm level is geometry/topology-only. At a fixed 32 µm domain it requires
320³ voxels. A PC/xGEMS state would allocate dense hydrate fields for roughly
64 solid channels plus transactional copies, exceeding the supported coupled
runtime memory envelope. It is therefore used to test initial-interface and
discretization trends, not as a fourth chemistry trajectory. Coupled chemistry
remains qualified through 0.25 µm.

## Staged execution policy

1. Geometry-only screen for every unique config.
2. Complete the already-running 64³ @ 1.0 µm, seed 20260731 baseline through
   28 days as the local late-age anchor.
3. Run all remaining domain, voxel, and seed levels through 1, 3, and 7 days.
4. At 7 days, rank each axis using absolute phase inventories, total hydrate,
   water compartments, capillary porosity, clinker reaction degree, CSHQ
   authority, and available morphology descriptors.
5. For each axis, extend the baseline and the largest-deviation or extreme case
   to 28 days. Extend further cases only if the 7-day ranking changes at 28 days
   or late sulfate/pore repartition grows materially.

Execution placement: the primary computer finishes its already-running Q64
28-day anchor only. All remaining geometry and 7-day domain/voxel/seed screens
are assigned to the frozen remote package. Returned artifacts are compared on
the primary computer after their hashes and qualification evidence are checked.

The original four-output 28-day configs are preserved for selective extension.
Files ending in `_7d.json` are the authoritative screening configs and end at
168 h. A 7-day pass is not a claim of 28-day convergence: the timestep study
showed that capillary-pore and sulfate-phase differences can amplify after 7 days.

These runs qualify numerical and spatial sensitivity only; they do not claim
independent experimental validation.
