# DIRAC-021 red-team review

## Verdict

`PASS_SCOPED_ACTION_DERIVED_REST_CLOCK`

DIRAC-021 adds no native dynamics. It follows the nonnegative rest coefficient
already selected from the native action through the zero-momentum Dirac
Hamiltonian and derives its two opposite continuum time phases.

## Findings and repairs

| Severity | Finding | Repair | Result |
|---|---|---|---|
| Critical | Writing `exp(±i 19 t)` directly could bake in the desired clock. | First proved from the existing Hamiltonian that explicit upper and lower rest states have energies `+chi0` and `-chi0`; then proved the phase derivatives and both componentwise Dirac time equations. The value `chi0=19` remains the action-selected DIRAC-011 result in the transitive certificate. | Closed. |
| High | The doubled frequency could be a definition rather than a consequence. | Defined relative branch phase as the quotient of the independently proved opposite phases and proved it equals the exponential with coefficient `2 chi0`. | Closed. |
| High | “Compton clock” could imply that an isolated global phase is directly observable. | Proved and independently checked that each pure branch has constant density. The theorem claims only phase evolution; a measurable doubled signal requires a specified cross-branch observable, left to DIRAC-022. | Closed as a claim boundary. |
| High | `1/19` could be mislabeled as the native-coordinate reduced Compton length. | Separated the inverse-mass scale `1/19` from the native-coordinate length `c_eff/m = sqrt(12)/19`, using the already-derived C19 continuum speed. | Closed. |
| High | The two signs could be confused with the two cubical flavor copies or declared electron/positron states. | The proof uses the positive and negative mass-grading eigenspaces inside one four-component representation. No particle identity, charge, flavor selection, or antiparticle interpretation is claimed. | Closed. |
| Medium | The number `19` could be presented as the measured electron mass. | Every value is explicitly in native lattice units. No SI bridge or measured mass is claimed. | Closed. |
| Medium | Numerical evidence could pass without testing the branch distinction. | Independent controls reject a wrong mass and a same-sign pair; they also verify that pure-branch density contains no clock signal. | Closed. |

## Full-chain review

- The publication certificate contains the unchanged DIRAC-001--020 chain.
- The mass entering DIRAC-021 is the unique nonnegative rest coefficient proved
  in DIRAC-011, not a newly fitted electron parameter.
- The rest Hamiltonian, phase derivatives, first-order equations, periods, and
  scale relations are formal Lean theorems.
- No native field, action term, force, stencil change, projector, clock
  register, fitted coefficient, or electromagnetic assumption was added.
- Audited theorems use only `propext`, `Classical.choice`, and `Quot.sound`.
- Open claims include an observable cross-branch velocity oscillation,
  zitterbewegung amplitude and position law, SI calibration, electromagnetic
  coupling, and nonlinear localized-electron dynamics.

