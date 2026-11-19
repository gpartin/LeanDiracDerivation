# End-to-end red team: DIRAC-001 through DIRAC-030

## Verdict

`PASS_SCOPED_CONDITIONAL_FREE_DIRAC_CHAIN_THROUGH_GORDON`

The review was rerun from the two-field C19 action through DIRAC-030 after
all five new stages compiled independently. No unresolved critical or high
finding remains.

## Top-down dependency audit

1. DIRAC-001--004 declare the two complex three-component fields, positive
   C19 action, its variations and native reversible update.
2. DIRAC-005--009 derive the retained cubical sector and Clifford square.
3. DIRAC-010--015 map that sector to the free continuum Dirac operator with
   mass, time, transport, current and controlled remainders.
4. DIRAC-016--025 derive the two-copy translation structure, spin-half
   representation, transport, rest clock, zitterbewegung, phase harmony and
   velocity-clock identities.
5. DIRAC-026 derives boosts from the same transport generators.
6. DIRAC-027 combines those boosts and rotations into the spin-half
   Pauli--Lubanski invariant.
7. DIRAC-028 derives the infinitesimal Wigner/Thomas half factor.
8. DIRAC-029 derives the free charge-conjugate frequency branch.
9. DIRAC-030 derives the free Gordon current split.

Every new certificate contains the prior certificate. No later theorem is
imported upstream, and no stage edits the native action.

## Injection, tautology and circularity audit

- No theorem is closed by `sorry`, `admit`, a custom axiom, status metadata,
  or a proposition defined as `True`.
- `#print axioms` for every new root reports only `propext`,
  `Classical.choice` and `Quot.sound`.
- Boosts are composed from prior transport; relativistic spin from prior
  boosts/rotations; Thomas rotation from their noncommutation; conjugation
  from prior grading/transport plus native complex conjugation; Gordon spin
  current from prior gamma commutators.
- The target constants `3/4`, `1/2` and `1/(2m)` do not appear in action data
  or theorem premises. They are proved operator consequences.
- All two-copy results retain both copies. None is silently projected out.

## Framework-drift audit

DIRAC-026--030 add no field, C19 bond, weight, action term, coefficient,
update rule, projector, runtime spinor rule, external interaction or fitted
parameter. The locked two-field action remains byte-identical.

## Remaining boundaries

The cumulative result is conditional on the previously derived low-momentum
four-component sector. It is not an exact finite-spacing Lorentz theorem, a
localized nonlinear electron, an electromagnetic charge/magnetic-moment
theorem, an SI calibration, or a Standard Model derivation. Those claims are
not needed for the scoped free-electron results proved here.
