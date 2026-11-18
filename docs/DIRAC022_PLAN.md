# DIRAC-022 plan: derived zitterbewegung observable

## Goal

Use the Hamiltonian, mass, branch phases, and velocity operator already proved
in DIRAC-010 through DIRAC-021 to derive the free-Dirac zitterbewegung signal.
No action term, field, clock, oscillation rule, fitted coefficient, or prepared
frequency may be added.

## Formal construction

1. Specialize the already-derived velocity operator
   `∂H/∂p_j = c_eff alpha_j` to zero momentum.
2. Construct an explicitly normalized coherent superposition of one proved
   positive-energy rest state and one proved negative-energy rest state that
   the native `alpha_j` couples.
3. Evolve both components with the independently proved DIRAC-021 phases.
4. Derive the velocity expectation from the native inner product. Prove that
   it oscillates at angular frequency `2 chi0=38`, while either pure branch has
   zero rest-velocity expectation.
5. Integrate that velocity expectation to an explicit position-expectation
   oscillation and prove its native amplitude bound. Distinguish the
   oscillation amplitude from the full peak-to-peak excursion.
6. Relate the result to the general Heisenberg velocity identity for the same
   Hamiltonian, keeping the zero-momentum specialization and every required
   coherence assumption explicit.

## Acceptance

DIRAC-022 passes only if:

- the cumulative certificate imports DIRAC-021 transitively;
- the velocity observable is the Hamiltonian momentum derivative proved in
  DIRAC-013, not a newly supplied matrix;
- the `2 chi0` frequency follows from the cross term between the two proved
  energy branches;
- pure-branch, same-sign, wrong-mass, and incoherent-mixture controls do not
  produce the claimed oscillation;
- normalization, frequency, phase, velocity amplitude, position law, and
  amplitude bound are Lean theorems and independently reconstructed;
- the action, fields, C19 geometry, recurrence, mass coefficient, and prior
  publication theorems remain unchanged;
- the full DIRAC-001--022 chain passes the compiled axiom, adversarial,
  contract, final-chain, and publication audits.

## Failure outcomes

- `OPEN_OBSERVABLE_MAPPING`: the existing velocity operator cannot be pulled
  through the chosen rest-state embedding without a missing theorem.
- `OPEN_COHERENCE_CONSTRUCTION`: no normalized native cross-branch state with
  the required nonzero velocity cross term is constructed.
- `FAIL_FREQUENCY_OR_AMPLITUDE`: the derived expectation does not have the
  standard `2m` frequency or the stated native amplitude.
- `FAIL_BOLTON`: any repair requires an inserted oscillation, velocity,
  position, clock, or electron-specific native rule.

The gate remains a free continuum-sector observable theorem. It will not by
itself prove a localized nonlinear electron, trembling of a native lattice
object, electromagnetic response, or an SI length/frequency.
