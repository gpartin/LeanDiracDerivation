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
| `DIRAC-002` | **PASS** | A nonnegative two-field complex-C3 action with geometry-derived exterior-area transport and a concrete prepared eight-corner background. |
| `DIRAC-003` | **PASS** | Exact state and velocity variations derive the reciprocal Psi and chi Euler-Lagrange force and momentum families on the concrete C19 bond map. |
| `DIRAC-004` | **NEXT** | Derive the action-compatible reversible native time update. |

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

The action contains ordinary positive kinetic energy, the local potential

```text
(rho_chi - chi0^2)^2 / 4
+ rho_chi rho_Psi / 2
+ rho_Psi^2 / 4
```

and a positive neighbor energy comparing the derived oriented area
`chi wedge Psi` across the declared C19 bonds. There is no independent
spinor, projector, link field, held track, Kato kinetic rule, Dirac matrix or
frame target.

The old scalar flat-octic potential is not part of this lowest-zoom action.
Recovering its exact higher-zoom scalar response remains open.

## Fifteen-stage derivation

| # | ID | Stage | Status | Completion criterion |
|---:|---|---|---|---|
| 1 | `DIRAC-001` | Authority boundary | **PASS** | One public root; no experimental or superseded dependency enters the production graph. |
| 2 | `DIRAC-002` | Native action | **PASS** | A positive two-field action, derived geometric transport, coefficient provenance and a cleared no-bolt-on review. |
| 3 | `DIRAC-003` | Native equations | **PASS** | Vary the frozen action and prove exactly two reciprocal Euler-Lagrange equation families. |
| 4 | `DIRAC-004` | Native time | **NEXT** | Derive and verify the action-compatible reversible second-order update. |
| 5 | `DIRAC-005` | Native sector | OPEN | Derive the invariant eight-site sector and retain every native copy. |
| 6 | `DIRAC-006` | Native symbol | OPEN | Derive finite-lattice transport coefficients from lattice characters. |
| 7 | `DIRAC-007` | Paired readout | OPEN | Build a lossless first-order readout from native displacement and velocity. |
| 8 | `DIRAC-008` | Native first-order operator | OPEN | Construct the first-order operator from action and cube data only. |
| 9 | `DIRAC-009` | Native Clifford algebra | OPEN | Derive mass grading, three transports, their square and anticommutators. |
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
LeanDirac/Dirac003.lean                DIRAC-003 action variation and equations
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
```

The axiom report may contain Lean's standard foundations
`propext`, `Classical.choice` and `Quot.sound`. The production guard executes
the audit and fails on any other dependency, or on `sorry`, `admit` or a custom
`axiom` in any production Lean module.

## Claim boundary

This repository records a noncanonical mathematical research candidate.
`DIRAC-003` derives the two weak Euler-Lagrange equation families from the
frozen action. It does not yet supply the native discrete time update, an
invariant spinor sector, a Dirac operator, an electron or physical spin. No
downstream stage inherits a pass from historical work outside this production
repository.
