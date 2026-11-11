# Lean LFM-to-Dirac Derivation

This repository is the production proof environment for an attempted formal
derivation of the free Dirac equation from a local two-field cubic-lattice
substrate.

Only reviewed proof dependencies belong here. Historical experiments,
unsuccessful candidates, numerical search scripts and answer-shaped scaffolds
remain outside this repository.

## Current result

| Gate | Status | What is established |
|---|---|---|
| `DIRAC-001` | **PASS** | One public proof root and a closed, auditable dependency chain. |
| `DIRAC-002` | **PASS** | The noncanonical two-field complex-C3 action contains the positive complex-χ covariance shape square and the positive native χ-overlap incidence square. Lean proves the common action is nonnegative and proves all six prepared face products. |
| `DIRAC-003` | **PASS** | Lean differentiates that exact action, including every covariance, overlap and neighbor contribution, to obtain reciprocal Ψ and χ equations. |
| `DIRAC-004` | **PASS** | The centered native update is rebuilt coordinate by coordinate from those exact action variations. Lean proves both equation families and exact reversibility. No uniqueness claim for a separately discretized action is made for the incidence correction. |
| `DIRAC-005` | **PASS** | Lean proves the covariance repair is stationary on the prepared running unit cell and the incidence residual cancels on that exact background. The period-two cancellation is not used as evidence for surrounding-cell transport. |
| `DIRAC-006` | **PASS** | Lean proves that the nondegenerate surrounding-cell Bloch reduction of the same action produces a nonzero signed cubical-incidence transport from the oriented face subset of C19. |
| `DIRAC-007` | **PASS** | At every nonzero timestep, the complete pair of committed native Ψ/χ layers is exactly equivalent to the current native state plus the backward interval velocity already used by the action. Lean proves both full-state round trips and an explicit equivalence between the thirty-two sector amplitudes and the running-sector state without selecting four components. |
| `DIRAC-008` | **PASS** | The incidence-action update and lossless two-layer readout give an exact reversible first-order map on the complete running Ψ/χ state. No spinor or sector projection is inserted. |
| `DIRAC-009` | **PASS** | Lean evaluates the native incidence residual on an actual repeated infinite-lattice Ψ/χ state, reduces it to signed cube incidence, and then proves three equal-square, mutually anticommuting directions and two retained rank-four copies. Parity is retained only as a geometric grading; its identification with physical mass is deferred to DIRAC-010. A nondegenerate 4x4x4 stability ensemble is required by this gate. |

These passes do **not** yet establish the Dirac equation, an electron,
physical spin one-half, particle formation, or a canonical LFM framework.
The derivation is conditional on an already-running prepared background; no
origin story is part of the present theorem.

### DIRAC-009 corrective-action result

DIRAC-002 now uses the original combined-density plus exterior-area action
with two generic positive squares.  The covariance square supplies the tested
departure stiffness.  The new incidence square uses only native complex-χ
overlaps, the oriented face subset of C19 and the composite `chi wedge Psi`.
It contains no prepared ray, spinor, projector, Dirac matrix, held track or
runtime constraint.

Lean proves that the complete action is positive and differentiates it for
independent Ψ, χ and velocity variations. The resulting reciprocal equations
generate the reversible update used through DIRAC-008. The period-two unit
cell makes opposite neighbors coincide and therefore cannot test the new odd
transport. DIRAC-009 now constructs the repeated native lattice fields, keeps
the surrounding-cell Bloch phases distinct, and proves that the zeroth
component of their exact action residual is gauge-equivalent to oriented
cubical incidence.

That native incidence supplies three directions with the required square and
anticommutation relations. A native flavor involution leaves two rank-four
copies; neither copy is deleted or declared physical. Cube parity also has the
right geometric anticommutation, but this gate does not call it physical mass.
The mass identification is the next derivation.

Fresh numerical checks reproduce the algebra to machine precision. The
nondegenerate stability test repeats the background on a 4x4x4 periodic C19
lattice, where positive and negative face neighbors are distinct, and applies
full-field perturbations so the incidence energy and force are nonzero. Its
recorded 100-cycle timestep ensemble measures all pattern defects and relative
energy deviation after every update and is part of the gate evidence. Standard
Dirac identification and the final continuum, density/current and
physical-electron claims remain downstream.

## Frozen DIRAC-002 action

The only native field families are

```text
Psi_i in C^3
chi_i in C^3
```

The declared candidate action contains ordinary positive kinetic energy, the local potential

```text
(rho_chi - chi0^2)^2 / 4
+ rho_chi rho_Psi / 2
+ rho_Psi^2 / 4
```

and a positive neighbor energy comparing the oriented area
`chi wedge Psi` across the declared C19 bonds. It also contains the positive
shape energy `1/2 sum_i |chi_i wedge (C_i chi_i)|^2`, where `C_i` is the
normalized complex-χ covariance of the complete C19 shell, and the positive
incidence energy `1/2 sum_i |D_chi (chi wedge Psi)_i|^2`, where `D_chi`
uses native complex-χ overlaps and oriented C19 faces. There is no independent
spinor, projector, link field, held track, Kato kinetic rule, Dirac matrix or
frame target. DIRAC-002 proves positivity and the complete prepared geometry;
it does not claim that this candidate action or its normalization is uniquely
forced by a still deeper substrate law.

The old scalar flat-octic potential is not part of this lowest-zoom action.
Recovering its exact higher-zoom scalar response remains open.

## Fifteen-stage derivation

| # | ID | Stage | Status | Completion criterion |
|---:|---|---|---|---|
| 1 | `DIRAC-001` | Authority boundary | **PASS** | One public root; no experimental or superseded dependency enters the production graph. |
| 2 | `DIRAC-002` | Native action | **PASS** | Declare the noncanonical two-field candidate action, prove positivity and all six prepared face products, and state explicitly that deeper uniqueness is not claimed. |
| 3 | `DIRAC-003` | Native equations | **PASS** | Derive the Ψ and χ force and conjugate-momentum families by independent arbitrary variations of the same local action. |
| 4 | `DIRAC-004` | Native time | **PASS** | Build the centered update from the actual action derivatives and prove reversibility and both equations for arbitrary field variations. |
| 5 | `DIRAC-005` | Prepared running background | **PASS** | Prove the covariance correction is stationary on the declared unit-cell background and distinguish its exact period-two cancellation from the surrounding-cell response. |
| 6 | `DIRAC-006` | Native local transport symbol | **PASS** | Prove the exact nondegenerate Bloch reduction of the action yields a nonzero cubical-incidence transport from native C19 faces. |
| 7 | `DIRAC-007` | Paired readout | **PASS** | Prove an exact two-way equivalence between the two committed native layers and current displacement plus backward interval velocity; construct the explicit sector-to-coordinate equivalence and preserve all thirty-two complex unit-cell coordinates in both directions. |
| 8 | `DIRAC-008` | Native first-order evolution map | **PASS** | Conjugate the corrected reversible two-layer update through the lossless running-state readout and prove a complete first-order map without a spinor or sector projection. |
| 9 | `DIRAC-009` | Native Clifford algebra | **PASS** | Connect the positive action to the three equal-square anticommuting cube directions, retain both rank-four copies, and pass a nondegenerate repeated-lattice stability ensemble without supplied matrices or a held track. |
| 10 | `DIRAC-010` | Dirac identification | OPEN | Identify the derived native algebra with standard Dirac notation downstream. |
| 11 | `DIRAC-011` | Readout rebuild | OPEN | Rebuild both frequency readouts and the public theorem from the repaired chain. |
| 12 | `DIRAC-012` | Continuum limit | OPEN | Prove the uniform-chi free-Dirac limit with vanishing space/time remainders. |
| 13 | `DIRAC-013` | Density and current | OPEN | Derive Dirac density, current and continuity through the same embedding. |
| 14 | `DIRAC-014` | Final audit | OPEN | Clear every critical/high circularity, dependency, placeholder and axiom finding. |
| 15 | `DIRAC-015` | Publication | OPEN | Reproduce the complete audited derivation from a clean checkout. |

```text
001 -> 002 -> 003 -> 004 -> 005 -> 006 -> 007 -> 008 -> 009 -> 010 -> 011
                                                                          |
                                                                          +-> 012 --+
                                                                          +-> 013 --+-> 014 -> 015
```

## Proof layout

```text
LeanDirac.lean                         sole public facade
LeanDirac/Publication.lean             highest admitted public theorem
LeanDirac/Dirac008Incidence.lean       complete reversible native first-order map
LeanDirac/Dirac007Incidence.lean       lossless state-plus-velocity readout
LeanDirac/Dirac006ActionTransport.lean completed nondegenerate action transport certificate
LeanDirac/Dirac006Incidence.lean       native C19 transport lineage
LeanDirac/Dirac005Incidence.lean       invariant prepared unit-cell sector
LeanDirac/Dirac004Incidence.lean       action-derived centered update
LeanDirac/ChiIncidenceVariation.lean   reciprocal two-field action variation
LeanDirac/ChiIncidence.lean            positive incidence action
LeanDirac/Dirac009Geometry.lean         cubical square-root geometry and controlled lattice remainder
LeanDirac/Dirac009.lean                 prepared-link algebraic scaffold; not the publication certificate
LeanDirac/Dirac009ActionBridge.lean     exact action-to-incidence Bloch reduction
LeanDirac/Dirac009Closed.lean           admitted native incidence/Clifford certificate
LeanDirac/Dirac002.lean                DIRAC-002 contracts
LeanDirac/PreparedBackground.lean      conditional eight-corner state
LeanDirac/NativeAction.lean            frozen action and positivity
LeanDirac/ExteriorArea.lean            derived wedge transport
LeanDirac/Dirac001.lean                DIRAC-001 native-framework contract
LeanDirac/CubeGeometry.lean            DIRAC-002 eight-corner geometry
LeanDirac/C19Geometry.lean             DIRAC-001 substrate geometry
LeanDirac/Foundation.lean              two fields and kinetic law
LeanDiracAudit.lean                    axiom report
project/                               immutable status and provenance
scripts/production_guard.py            manual dependency/hash/status audit; not a commit hook
scripts/adversarial_guard_test.py       negative tests for the audit logic
scripts/derivation_contract_audit.py    read-only DIRAC-001--009 status report
experiments/dirac009_native_coupling/   runnable nondegenerate 4x4x4 evidence
```

## Rebuild

Install Lean through `elan`, then run:

```bash
lake update
lake build
lake build LeanDiracAudit
python scripts/production_guard.py
python scripts/adversarial_guard_test.py
python scripts/derivation_contract_audit.py --strict
```

To regenerate the nondegenerate numerical evidence, install the optional
Python dependencies and run:

```bash
python -m pip install -r requirements.txt
python experiments/dirac009_native_coupling/candidate_007_l4_nondegenerate_stability.py
```

The generated production audit imports every locked Lean module and verifies
the resulting module coverage, including `LeanDiracAudit` itself. Lean inspects
every compiled production declaration, including declarations unused by the
publication theorem. Every dependency returned by `collectAxioms` must be one
of Lean's standard foundations: `propext`, `Classical.choice` or `Quot.sound`.
The audit fails on every other dependency, on `sorry`, or on a custom `axiom`.
Compiler-based adversarial fixtures cover interpolated terms, escaped
identifiers, raw strings, imported nonstandard axioms, duplicate reports,
incomplete metrics and non-finite timestep data. This avoids treating a partial
Python tokenizer as a Lean proof-security boundary.

## Claim boundary

This repository records a noncanonical mathematical research candidate.
`DIRAC-003` is a local variational theorem: it identifies the two native force
and momentum families as derivatives of the same declared action. It does not
claim formation, background selection or a continuum trajectory.
The public cumulative certificate explicitly retains the DIRAC-001 framework,
DIRAC-002 candidate-action/geometry certificate and DIRAC-003 local variation
certificate rather than relying on imports alone. `DIRAC-004` constructs the
centered update directly from the action derivatives and proves exact
reversibility and both equation families for every field direction. The
incidence chain does not claim a separate unique discrete-action theorem.
`DIRAC-005` proves that
the period-two eight-site tetrahedral ray bundle is invariant under that
unrestricted update and that its exact normal force block vanishes. Lean also
proves that amplitude extraction and state reconstruction are exact inverses,
so the bundle has sixteen unique complex coordinates on each state slice:
eight Ψ amplitudes and eight χ amplitudes. The two-layer centered state has
thirty-two.
No four-component split is supplied by DIRAC-005. `DIRAC-006` keeps the
surrounding-cell phases that the period-two cell loses and proves that the
linear coefficient of the actual incidence residual is the nonzero signed
cube transport. It does not identify either rank-four flavor copy as a
physical spinor. `DIRAC-007` proves that the two
committed native layers can be replaced exactly by the current state and the
backward interval velocity, and rebuilt with no information loss. This
velocity belongs to the interval from the previous layer to the current layer;
it is not asserted to be a same-time continuum derivative. Lean constructs an
explicit equivalence between the running sector and its thirty-two complex
coordinate functions. It does not select four components or define their
evolution. `DIRAC-008` conjugates the action-derived centered two-layer
update through that lossless readout. Lean proves the complete nonlinear
first-order map, its exact inverse, the explicit Ψ/χ displacement and
interval-velocity formulas, preservation of the admitted sector, and the
equivalent reversible map on all thirty-two complex coordinates. No matrix or
projector is supplied. `DIRAC-009` then proves that the action-derived
incidence directions square equally and anticommute and that a native flavor
involution retains two rank-four copies. It does not select one copy or derive
the physical mass grading. Standard Dirac identification, the continuum
limit, density/current, a physical electron and physical spin remain
downstream work.
