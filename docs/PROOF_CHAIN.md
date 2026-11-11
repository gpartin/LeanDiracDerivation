# Formal proof chain

The sole publication root is
`LeanDirac.current_highest_gate_is_dirac009`. Each passing gate is a theorem
dependency of that root, not a status label standing in for a proof.

| Gate | Status | Formal contribution | Next dependency |
|---|---|---|---|
| DIRAC-001 | PASS | Fixes the two complex C3 fields, constants, C19 geometry, and public authority boundary. | A concrete action on that state. |
| DIRAC-002 | PASS | Declares one positive two-field action and proves its prepared cube products. | Vary the same action. |
| DIRAC-003 | PASS | Derives reciprocal Psi and chi force/momentum families from arbitrary variations. | Build the native time map. |
| DIRAC-004 | PASS | Constructs the centered update from those forces and proves reversibility. | Show the prepared sector is compatible. |
| DIRAC-005 | PASS | Proves the prepared unit-cell relations and exact period-two cancellation. | Restore surrounding-cell information. |
| DIRAC-006 | PASS | Evaluates the native residual on a repeated lattice and derives nonzero signed incidence transport. | Preserve all state information. |
| DIRAC-007 | PASS | Proves an exact equivalence between two committed layers and state plus interval velocity. | Conjugate the update through that readout. |
| DIRAC-008 | PASS | Produces an exact reversible first-order map on the complete native running state. | Derive the native cube algebra. |
| DIRAC-009 | PASS | Connects the action response to three equal-square anticommuting cube directions and retains both rank-four copies. | Identify, rather than assume, standard Dirac structure. |
| DIRAC-010 | OPEN | Identify native grading and transport with a standard Dirac representation. | Rebuild frequency readouts. |
| DIRAC-011 | OPEN | Build the common positive/negative-frequency readouts from the same embedding. | Take controlled limits. |
| DIRAC-012 | OPEN | Prove the uniform-chi low-momentum continuum Dirac equation with explicit errors. | Derive observables. |
| DIRAC-013 | OPEN | Derive density, current, and continuity from the same action and embedding. | Final audit. |
| DIRAC-014 | OPEN | Eliminate every critical/high circularity, placeholder, and axiom finding. | Clean reproduction. |
| DIRAC-015 | OPEN | Reproduce and package the complete derivation from a clean checkout. | Publication decision. |

## Current theorem boundary

`DIRAC-009` proves a native cubical Clifford-like algebra from the declared
action response. It does not yet prove the free Dirac equation, select one of
the two rank-four copies, identify physical mass, establish physical spin
one-half, or establish an electron. These are intentionally downstream gates.

## Axiom policy

The production audit imports every Lean module and inspects the compiled
environment. The admitted publication theorem may depend only on Lean's
standard foundations:

- `propext`
- `Classical.choice`
- `Quot.sound`

No `sorry`, custom axiom, assumed Dirac matrix, or status metadata is accepted
as a physical proof obligation.
