# Formal proof chain

The sole publication root is
`LeanDirac.current_highest_gate_is_dirac011`. Each passing gate is a theorem
dependency of that root, not a status label standing in for proof.

| Gate | Status | Formal contribution | Next dependency |
|---|---|---|---|
| DIRAC-001 | PASS | Fixes two complex C3 fields, constants, C19 geometry, and the authority boundary. | A concrete action. |
| DIRAC-002 | PASS | Declares one positive local-plus-exterior-area two-field action and the prepared cube products. | Vary the same action. |
| DIRAC-003 | PASS | Derives reciprocal Psi and chi force/momentum families from arbitrary variations. | Build the native time map. |
| DIRAC-004 | PASS | Derives the centered update from the discrete action and proves reversibility. | Show prepared-sector compatibility. |
| DIRAC-005 | PASS | Proves the conditional prepared eight-site sector is closed by the native update. | Derive its lattice transport. |
| DIRAC-006 | PASS | Derives the exact scalar C19 transport symbol from the native bond action. | Preserve all state information. |
| DIRAC-007 | PASS | Proves an exact equivalence between two committed layers and state plus interval velocity. | Conjugate the update. |
| DIRAC-008 | PASS | Produces an exact reversible first-order map on the complete native state. | Factor the continuum spatial symbol. |
| DIRAC-009 | PASS | Derives cubical Clifford geometry as a square root of the C19 quadratic continuum symbol, retains both rank-four copies, and bounds the exact lattice remainder. | Identify the standard representation. |
| DIRAC-010 | PASS | Proves exact invertible maps from both native copies to standard Dirac alpha/beta operators. | Derive the native frequency and mass map. |
| DIRAC-011 | PASS | Builds both frequency graphs from the same phase space and embeddings, retains the exact C19 remainder, and uniquely selects the nonnegative rest coefficient from the native running frequency. | Take controlled limits. |
| DIRAC-012 | OPEN | Prove the uniform-chi low-momentum continuum Dirac equation with explicit errors. | Derive observables. |
| DIRAC-013 | OPEN | Derive density, current, and continuity from the same action and embedding. | Final audit. |
| DIRAC-014 | OPEN | Eliminate every critical/high circularity, placeholder, and axiom finding. | Clean reproduction. |
| DIRAC-015 | OPEN | Reproduce and package the complete derivation from a clean checkout. | Publication decision. |

## Current theorem boundary

DIRAC-011 proves a common frequency/rest-mass map, not the full free Dirac
equation. The exact finite-lattice remainder remains present, and DIRAC-012
must control it in the continuum evolution theorem. Both rank-four copies are
retained. No physical spin or electron claim is made.

## Axiom policy

The admitted publication theorem may depend only on `propext`,
`Classical.choice`, and `Quot.sound`. No `sorry`, custom axiom, assumed
native Dirac matrix, or status metadata closes a physical proof obligation.
