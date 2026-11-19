# DIRAC-026 red-team review

## Verdict

`PASS_SCOPED_FREE_LORENTZ_BOOSTS`

No critical or high finding remains. DIRAC-026 changes no native action,
field, coefficient, C19 weight, update rule, dispersion or current.

## Injection and circularity checks

- Each boost generator is constructed from the three transport generators
  already derived from the C19 Hamiltonian; no boost matrix is supplied as
  an independent axiom.
- Lean proves the boost-boost and rotation-boost commutators by expanding the
  inherited cube operators component by component.
- The rapidity map is then built by exponentiating a generator whose square
  was proved, and its inverse is proved from the hyperbolic identity.
- Boosted rest states are proved to solve the same inherited Dirac
  Hamiltonian at the transformed energy and momentum.
- The mass shell, density and current are calculated independently; none is
  defined to equal its target result.
- `#print axioms` reports only `propext`, `Classical.choice` and `Quot.sound`.

## Adversarial controls

Changing the boost sign breaks the boosted-state eigen equation. Removing a
transport direction breaks the Lorentz commutators. Replacing the hyperbolic
pair by circular sine/cosine breaks the inverse and mass-shell identities.

## Claim boundary

This is an exact theorem about the already-derived continuum Dirac readout.
It does not assert exact Lorentz symmetry of the finite-spacing lattice or a
localized nonlinear electron.
