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
| `DIRAC-002` | **PASS** | A declared noncanonical two-field complex-C3 candidate action is nonnegative, and all six face products of its concrete prepared eight-corner background are proved. The action is a starting hypothesis, not a claimed unique consequence of a deeper law. |
| `DIRAC-003` | **PASS** | Independent arbitrary variations of the actual local action produce the native Ψ and χ force and momentum families. No all-time trajectory, decay assumption or continuum equation is used at this lattice-first gate. |
| `DIRAC-004` | **PASS** | A two-step discrete action is built from the frozen kinetic energy and potential. Lean proves its stationary point is uniquely the explicit centered C19 update, proves exact reversibility, and proves both reciprocal equations for every complex-C3 field variation. |
| `DIRAC-005` | **PASS** | For the period-two C19 unit cell, the unrestricted action-derived update preserves the tetrahedral Ψ/χ ray sector. Its exact normal cross block is zero. Lean reconstructs every sector state from one unique set of sixteen complex amplitudes; the two committed layers therefore have thirty-two. No four-component split is supplied or assumed. |
| `DIRAC-006` | **PASS** | The admitted sector's exact native bond energy reduces to the Ψ-χ product amplitude. An explicitly constructed Bloch-modulated native Ψ/χ state over a period-two ray background then produces the nonnegative local infinite-lattice C19 character symbol, with zero first moment and isotropic second moment 12 in the declared relative-weight normalization. No four-component split is supplied. |
| `DIRAC-007` | **PASS** | At every nonzero timestep, the complete pair of committed native Ψ/χ layers is exactly equivalent to the current native state plus the backward interval velocity already used by the action. Lean proves both full-state round trips and an explicit equivalence between the thirty-two sector amplitudes and the running-sector state without selecting four components. |
| `DIRAC-008` | **PASS** | Conjugating the unique action-stationary native update through the DIRAC-007 readout gives an exact reversible first-order evolution map on the complete running Ψ/χ state and all thirty-two sector coordinates. Its four explicit displacement/velocity formulas are proved from the same acceleration. It is nonlinear and supplies no Dirac matrix or four-component selector. |
| `DIRAC-009` | **NEXT** | Linearize the complete native map at the declared running background and derive any invariant four-component sector, mass grading and three transport generators without inserting them. |

These passes do **not** yet establish the Dirac equation, an electron,
physical spin one-half, particle formation, or a canonical LFM framework.
The derivation is conditional on an already-running prepared background; no
origin story is part of the present theorem.

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
`chi wedge Psi` across the declared C19 bonds. There is no independent
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
| 4 | `DIRAC-004` | Native time | **PASS** | Derive the centered update as the unique stationary point of the discrete two-step action; prove reversibility and both equations for arbitrary field variations. |
| 5 | `DIRAC-005` | Native unit-cell sector | **PASS** | Prove the action-derived update closes on the period-two, eight-site Ψ/χ ray sector; calculate the normal cross block and prove that amplitude extraction and reconstruction are exact inverses, yielding sixteen complex amplitudes per state slice and thirty-two across the two committed layers. The four-component selection problem remains downstream. |
| 6 | `DIRAC-006` | Native local transport symbol | **PASS** | Construct the repeated Bloch-modulated native Ψ/χ state over the period-two ray background, reduce its actual bond action and derive its exact local infinite-lattice C19 character coefficient, zero first moment and isotropic second moment. |
| 7 | `DIRAC-007` | Paired readout | **PASS** | Prove an exact two-way equivalence between the two committed native layers and current displacement plus backward interval velocity; construct the explicit sector-to-coordinate equivalence and preserve all thirty-two complex unit-cell coordinates in both directions. |
| 8 | `DIRAC-008` | Native first-order evolution map | **PASS** | Conjugate the unique action-stationary two-layer update through the lossless running-state readout; prove the explicit Ψ/χ displacement and interval-velocity formulas, reversibility, sector closure and exact evolution on all thirty-two coordinates. |
| 9 | `DIRAC-009` | Native Clifford algebra | **NEXT** | Linearize the complete map, derive rather than choose any invariant four-component sector, then derive mass grading, three transports, their squares and anticommutators. |
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
LeanDirac/Dirac008.lean                complete reversible native first-order map
LeanDirac/Dirac007.lean                lossless state-plus-velocity readout
LeanDirac/Dirac006.lean                native product-channel C19 transport symbol
LeanDirac/Dirac005.lean                invariant period-two unit-cell sector
LeanDirac/Dirac004DiscreteAction.lean  discrete action/update equivalence
LeanDirac/Dirac004.lean                force-coordinate/update algebra
LeanDirac/Dirac003Contract.lean         independent raw-action curve contract
LeanDirac/Dirac003.lean                local two-field action-variation proof
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
scripts/production_guard.py            dependency/hash/status guard
```

## Rebuild

Install Lean through `elan`, then run:

```bash
lake update
lake build
lake build LeanDiracAudit
python scripts/production_guard.py
python scripts/adversarial_guard_test.py
```

The axiom report may contain Lean's standard foundations
`propext`, `Classical.choice` and `Quot.sound`. The production guard executes
the audit and fails on any other dependency, or on `sorry`, `admit` or a custom
`axiom` in any production Lean module.

## Claim boundary

This repository records a noncanonical mathematical research candidate.
`DIRAC-003` is a local variational theorem: it identifies the two native force
and momentum families as derivatives of the same declared action. It does not
claim formation, background selection or a continuum trajectory.
The public cumulative certificate explicitly retains the DIRAC-001 framework,
DIRAC-002 candidate-action/geometry certificate and DIRAC-003 local variation
certificate rather than relying on imports alone. `DIRAC-004` constructs the
actual two-step discrete action, differentiates it
under arbitrary combined field variations, and proves that stationarity is
equivalent to the explicit centered update. It also proves exact reversibility
and both equation families for every field direction. `DIRAC-005` proves that
the period-two eight-site tetrahedral ray bundle is invariant under that
unrestricted update and that its exact normal force block vanishes. Lean also
proves that amplitude extraction and state reconstruction are exact inverses,
so the bundle has sixteen unique complex coordinates on each state slice:
eight Ψ amplitudes and eight χ amplitudes. The two-layer centered state has
thirty-two.
No four-component split is supplied by DIRAC-005. At the close of that gate,
the repeated-lattice transport symbol and native four-component selection were
both still open. `DIRAC-006` now closes the first item: it constructs an actual
Bloch-modulated native Ψ/χ state over a period-two ray background and proves its exact
local neighbor-transport symbol on lattice characters. It does not
claim that the product channel is a four-component spinor or that it closes
under the complete local time evolution. `DIRAC-007` now proves that the two
committed native layers can be replaced exactly by the current state and the
backward interval velocity, and rebuilt with no information loss. This
velocity belongs to the interval from the previous layer to the current layer;
it is not asserted to be a same-time continuum derivative. Lean constructs an
explicit equivalence between the running sector and its thirty-two complex
coordinate functions. It does not select four components or define their
evolution. `DIRAC-008` now conjugates the unique action-stationary two-layer
update through that lossless readout. Lean proves the complete nonlinear
first-order map, its exact inverse, the explicit Ψ/χ displacement and
interval-velocity formulas, preservation of the admitted sector, and the
equivalent reversible map on all thirty-two complex coordinates. No matrix,
projector or four-component split is supplied. Native four-component
selection and linearization, Clifford algebra, long-time stability, an
electron and physical spin remain downstream work.
