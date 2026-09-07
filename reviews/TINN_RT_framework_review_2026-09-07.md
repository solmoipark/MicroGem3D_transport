# TINN Reactive Transport Framework Review

**Review date:** 2026-09-07  
**Reviewed revision:** `38e57c45338f6c2967b049606cb94fbacee73966`  
**Scope:** Framework architecture, transport, PHREEQC surface sorption, GEMS coupling, configuration safeguards, and reproducibility.  
**Change policy:** The review did not modify implementation, configuration, databases, or existing results. This document records the findings and recommendations; no fixes have been applied.

## Overall assessment

The separation of the conservation ledger, chemistry backend, microstructure, and transport is a sound foundation. Explicit element and endmember accounting, antisymmetric transport transfers, transactional state updates, and restart checks are valuable strengths.

The next priority should be consistency between these components before extending the framework to additional reaction mechanisms. In particular, element conservation alone does not establish charge conservation, correct reaction stoichiometry, or equilibrium consistency at the end of a coupled step.

Several implementation gaps were reproduced with small numerical examples or initialization checks. Other findings concern documented model limitations and validation work that remains necessary. These evidence levels are distinguished below.

## Current division of responsibilities

| Component | Responsibility in the reviewed implementation |
|---|---|
| GEMS | Mineral and solid-solution equilibrium; aqueous speciation |
| TINN transport | Domain-graph diffusion using scalar diffusivity or a species-informed Nernst–Planck approximation |
| PHREEQC | `SURFACE` sorption and optional Portlandite equilibrium buffering |
| `exchange_be` | Diffusive exchange between transport domains |
| `exchange_tau_h` / `exchange_tau_h_per_phase` | Rate-limited re-equilibration of existing solids |
| `sorption.alkali_exchange` | A configuration permission flag for alkali sorption |

The current implementation does **not** generate PHREEQC's native `EXCHANGE` assemblages or their associated exchange-reaction definitions. Ligand exchange expressed through a `SURFACE_SPECIES` reaction is distinct from native PHREEQC ion exchange. Native `EXCHANGE` uses separately defined exchange sites, species, and equilibrium constants. See the [USGS EXCHANGE documentation](https://water.usgs.gov/water-resources/software/PHREEQC/documentation/phreeqc3-html/phreeqc3-14.htm).

## Findings and recommended priorities

| ID | Priority | Finding | Evidence |
|---|---|---|---|
| RT-01 | High | The NP charge diagnostic does not validate the flux actually applied by backward Euler | Numerical reproduction |
| RT-02 | High | Gel-only connections are included in reported diffusivity but excluded from the RT graph | Numerical reproduction |
| RT-03 | High | Incorrect O/H bookkeeping in surface reactions can pass the current checks | PHREEQC operator reproduction |
| RT-04 | High | Configuration safeguards miss structural alkali uptake and a per-phase buffering conflict | Initialization checks |
| RT-05 | Medium | PHREEQC transient failures bypass the engine's normal step-retry path | Code inspection |
| RT-06 | High for quantitative applications | Sorption–mineral splitting and calibration limits require additional validation | Code inspection and existing project records |
| RT-07 | High for sulfate-bearing chloride runs | The documented sulfur redox correction is not consistently applied to examples | Configuration comparison and existing project records |

Priority indicates recommended work order, not a measured error magnitude in a production simulation.

### RT-01 — Validate charge conservation on the applied transport flux

**Observation.** The NP kernel calculates a zero-current species flux from the frozen starting speciation. It converts that flux into element-specific effective diffusivities. The backward Euler solver then advances the element columns independently using those diffusivities.

The reported `np_charge_flux_rel_max` checks the initial projected species flux. It does not check the flux subsequently applied by the implicit solver. Independent implicit updates can change the relative ion fluxes and therefore break the original zero-current relationship, even when no diffusivity clamp is activated.

**Reproduction.** A synthetic two-domain, three-ion example used unit domain water volumes and unit graph conductance, charge vector `[+1, -1, +1]`, an identity species-to-element map, diffusivities `[9.31, 2.03, 1.96]`, a time step of `0.1`, and initial inventories:

```text
Domain A: [3.0, 4.0, 1.0]
Domain B: [1.0, 1.5, 0.5]
```

Both initial domains are electrically neutral. The result was:

| Quantity | Result |
|---|---:|
| Activated diffusivity clamps | 0 |
| Reported relative charge residual | approximately 5.32e-17 |
| Relative charge imbalance of the applied flux | approximately 4.64% |
| Final domain charge inventories | approximately +0.04434 and -0.04434 |

The applied-flux residual is `abs(sum(z_i * F_i)) / sum(abs(z_i * F_i))`. These are normalized synthetic inputs, not measured cement pore-solution conditions. The result demonstrates a diagnostic gap; it does not establish a 4.64% error in a production case.

**Recommendation.** Evaluate charge balance on the actual accepted transport update. Use a controlled error response, such as smaller time steps or a coupled iteration, when the residual is excessive. If the approximation remains inadequate, retain ion coupling explicitly in the transport solve. Clamping statistics alone are insufficient because the discrepancy also occurs without clamping.

**Code:** [NP effective diffusivity and diagnostic](src/tinn/transport.py#L342), [backward Euler transport](src/tinn/transport.py#L474).

### RT-02 — Make gel connectivity consistent between transport and reporting

**Observation.** The diffusivity report includes gel-bearing hydrate voxels as conducting cells. The RT cluster labels, however, are constructed only from voxels with capillary liquid above `LIQ_EPS`. A voxel containing gel but no capillary liquid receives no transport-domain label and cannot connect two wet domains.

Consequently, sharing the same local conductance formula does not make the two networks equivalent: their connectivity differs.

**Reproduction.** A small `4 x 4 x 4` example placed liquid layers on opposite sides of two gel-only layers. With gel relative diffusivity `0.0025`:

| Quantity | Result |
|---|---:|
| RT liquid clusters | 2 |
| RT graph edges connecting the domains | 0 |
| Reported relative diffusivity through the gel bridge | approximately 0.00499 |

**Implication.** The discrepancy can matter for mature paste, dense microstructures, and precipitation-induced pore blockage. Reported finite diffusivity may coexist with a completely disconnected RT path.

**Recommendation.** Define a consistent treatment of gel-mediated connectivity. This should also specify whether gel water stores dissolved solutes or only contributes conductance. If the two models intentionally represent different physics, their outputs and applicability need to state that distinction explicitly.

**Code:** [capillary-liquid cluster labeling](src/tinn/transport.py#L31), [domain graph construction](src/tinn/transport.py#L191), [reported diffusivity network](src/tinn/analysis.py#L661).

### RT-03 — Independently validate O/H stoichiometry in surface reactions

**Observation.** The operator compares the declared `sorbed_elements` with PHREEQC aqueous totals for the whitelisted solute elements. O and H are excluded from this whitelist and travel as a signed protonation/water frame.

This allows incorrect O/H bookkeeping to pass even when the PHREEQC reaction itself is correctly balanced.

**Reproduction.** For:

```text
Surf_sOH + SO4-2 = Surf_sSO4- + OH-
```

the correct solution-removal vector per mole of bound species is:

```text
S: 1, O: 3, H: -1
```

Changing the declared vector in memory to `S: 1, O: 0, H: 0` was accepted. The operator returned the same surface occupancy while recording different O/H transfers. No repository configuration was changed for this check.

**Implication.** A global ledger can remain closed because it applies the same incorrect vector to both stores. Ledger closure therefore cannot substitute for an independent stoichiometric check.

**Recommendation.** Derive the solution-removal vector from the reaction definition, or independently validate all its coefficients, including O/H. Keep this validation distinct from the same-float transfer checks used by the conservation ledger.

**Code:** [surface reaction schema](src/tinn/config.py#L695), [operator closure witness](src/tinn/backend.py#L486).

### RT-04 — Strengthen configuration compatibility checks

Two combinations were accepted during initialization checks.

**A. Structural alkali uptake plus additional alkali sorption.** The PC-Cl CSHQ solid solution includes `KSiOH` and `NaSiOH`, but a configuration adding Na surface sorption with `alkali_exchange: true` passed initialization. The current safeguard mainly checks for a `CNASH` phase name. A `CSHQ` name does not establish that the solid solution is free of structural alkali uptake.

**Recommendation.** Check the actual endmembers and their assigned mechanisms. Permit combined structural and surface uptake only when the distinction and parameterization are explicit, so that the same uptake contribution is not counted twice.

**B. Portlandite buffering plus per-phase rate limitation.** The engine rejects a sorption buffer combined with global `exchange_tau_h`, but accepted:

```json
{
  "exchange_tau_h_per_phase": {
    "Portlandite": 100.0
  }
}
```

with `sorption.buffer_phase: "Portlandite"`. The S stage obtains the owned buffer amount, while the R stage's offered solid amount can be restricted by the per-phase rate setting.

**Recommendation.** Apply compatibility checks to the effective per-phase offer policy, rather than only the global configuration field. The buffer must not access an amount excluded from the corresponding reaction transaction.

These checks establish that the combinations are accepted. Their downstream quantitative effects were not measured in long simulations.

**Code:** [sorption and alkali safeguards](src/tinn/engine.py#L363), [buffer compatibility check](src/tinn/engine.py#L392), [buffer amount supplied to PHREEQC](src/tinn/engine.py#L1210).

### RT-05 — Route PHREEQC failures through the transaction and retry mechanism

**Observation.** A failed PHREEQC calculation raises `BackendTransientError`. The S-stage call does not catch that exception and convert it into a `StepReject`. The handling around GEMS reaction calls does not enclose the earlier sorption call.

**Implication.** A PHREEQC transient failure can terminate the run instead of reaching the existing time-step reduction and retry mechanism.

**Recommendation.** Give the S stage the same transactional failure contract as the other retryable operators. Preserve the committed state and return a rejection with useful domain and input diagnostics. Configuration and stoichiometry errors should remain distinguishable from retryable convergence failures.

**Code:** [PHREEQC exception conversion](src/tinn/backend.py#L455), [S-stage call](src/tinn/engine.py#L1214), [step retry loop](src/tinn/engine.py#L2139).

### RT-06 — Validate sorption–mineral splitting and respect calibration limits

**Observation.** The `T -> S -> R` sequence evaluates sorption using the existing sorbent sites and buffer inventory. GEMS then changes the aqueous composition, mineral assemblage, and potentially the amount of sorbent. The surface state at the end of the step is therefore not guaranteed to be in equilibrium with the final solution and solids.

This matters when adsorption, mineral transformation, and pH changes are strongly coupled. It also creates a risk of interpreting repeated numerical equilibration as physical relaxation. Once Portlandite is exhausted, the current CH-buffering correction is no longer available.

The project already records that the SO4 `no_edl` parameterization is a local calibration around pH 12.9–13.4. Its behavior should not be assumed valid for decalcified, SCM-rich, or high-alkali conditions without additional evidence. Existing project records also identify remaining differences between the S-stage buffered subsystem and the full pore-solution chemistry.

**Recommendation.** First compare otherwise identical runs at `dt`, `dt/2`, and `dt/4`, tracking free ions, sorbed inventories, pH, and CH depletion time. Separate this numerical convergence check from experimental calibration. If necessary, introduce an S–R iteration with explicit convergence criteria for the coupled chemical state. Evaluate applicability both before and after CH depletion.

**Code and records:** [S-stage ordering and site use](src/tinn/engine.py#L1167), [SO4 calibration limitations](PRD.md#L1040), [buffering limitations and existing measurements](PRD.md#L1078).

### RT-07 — Apply the sulfur redox policy consistently across examples

**Observation.** The sulfate-attack example declares suppression of reduced sulfur species and relevant sulfide phases. The chloride-ingress example still lacks that declaration, although it includes sulfate surface sorption.

The PRD documents an earlier artifact in which GEMS produced reduced aqueous sulfur and the sorption operator reconstructed that sulfur as sulfate, generating an inflated sorbed store. The current chloride-ingress example remains exposed to the conditions behind that documented issue. This review did not rerun the full scenario to measure its present magnitude.

**Recommendation.** Represent the sulfur oxidation-state policy consistently in shared chemistry configuration or validated example construction. Identify whether each published or stored result predates or includes the correction. Do not treat older and corrected results as interchangeable validation evidence.

**Configuration and records:** [chloride-ingress example](examples/qualification/chloride_ingress_opc32.json#L25), [sulfate-attack example](examples/qualification/sulfate_attack_opc32.json), [documented RT-S3 artifact and correction](PRD.md#L1202).

## Broader framework improvements

### Separate stage responsibilities inside the engine

`try_step()` spans approximately 1,500 lines and handles transport, chemistry selection, solid ownership, sorption, reaction, morphology, remapping, and conservation checks. This makes cross-stage contracts difficult to inspect and contributes to gaps such as the missing S-stage retry handling.

Consider explicit stage inputs, outputs, and transaction records. The existing fixed module-count constraint should not prevent clearer responsibility boundaries. This is a maintainability recommendation, not a request to rewrite the framework wholesale.

### Persist the chemical environment needed for reproducibility

The runtime database audits are useful, but checkpoint reproducibility should also identify the original contents of external thermodynamic inputs. Matching paths, species names, or endmember stoichiometry does not detect a changed equilibrium constant under the same name.

Store content hashes for the GEMS bundle and PHREEQC database, together with chemistry-engine and relevant dependency versions. Check those identities on restart. See [checkpoint metadata](src/tinn/storage.py#L118) and [configuration hashing](src/tinn/config.py#L1093).

### Report the magnitude of numerical fallback actions

Domain freezing, bath flushing, and dryout surrender preserve accounting but can still affect transport or reaction histories. Event counts alone are insufficient to assess their importance.

Report cumulative affected amounts per element, their fraction of the relevant inventory or transport flux, and the location and duration of frozen regions. This makes it possible to distinguish negligible numerical cleanup from a fallback that materially influences the predicted front. See [bath sweep](src/tinn/engine.py#L805), [nonconvergence freezing](src/tinn/engine.py#L1414), and [sorbed-store surrender](src/tinn/engine.py#L1940).

## Verification performed and limits of this review

The review inspected the implementation, configuration examples, PRD, previous review and PHREEQC integration notes, and selected existing result files.

Verification completed:

- **29 existing tests passed**, covering selected transport kernels, boundary behavior, NP behavior, PHREEQC sorption, and configuration/ledger checks.
- **1 existing integration test passed:** `test_sorption_engine_run_closes_and_restarts`, exercising GEMS–PHREEQC coupling, conservation, checkpoint persistence, and restart identity.
- Small in-memory reproductions confirmed the applied-flux charge diagnostic gap, the gel-connectivity mismatch, and acceptance of incorrect O/H bookkeeping.
- Initialization checks confirmed the two configuration compatibility gaps described in RT-04.

Passing existing tests is compatible with these findings: the tested contracts do not cover all the physical and cross-stage conditions identified here.

The numerical probes and test artifacts were confined to memory and temporary directories. No implementation, configuration, thermodynamic database, or existing result was edited. Full-duration chloride ingress, sulfate attack, and leaching simulations were **not** rerun. Accordingly, the review establishes specific implementation gaps and validation needs, but does not quantify their combined effect on long-term application predictions.
