# V3.2 Exact Finite-C19 Factorization Investigation

## Question

Can the original V3.2 global-spectrum finish line be recovered without
changing the V3.1 action, the eight-component cubical fiber, locality, or the
Dirac continuum limit?

## Machine-checked findings

`V32GlobalSpectrum.lean` now proves all of the following.

1. The unchanged action has an exact finite-C19 spectrum
   `mass^2 + c19TransportSymbol(momentum)` with real positive and negative
   branches and a nonzero-mass global gap.
2. The existing cubical Dirac root squares to the quadratic continuum symbol.
   It equals the exact finite-C19 Hessian exactly when
   `c19SpatialRemainder(momentum) = 0`.
3. At `(pi, 0, 0)`, the exact C19 transport symbol is `48`, while the
   remainder is `48 - 12*pi^2`, which is nonzero.
4. More generally, no finite set of real transport coefficients can factor
   the exact C19 symbol as a sum of squares if every coefficient is both odd
   under momentum reversal and periodic on the Brillouin torus. At `(pi,0,0)`
   every such coefficient must vanish, contradicting the exact value `48`.
5. The exact symbol does have a local bond decomposition:
   it is the sum of the squared norms of the eighteen weighted C19 bond
   differences `sqrt(weight) * (1 - character)`.

The main declarations are:

- `v32_pi_axis_transport_exact`
- `v32_no_global_exact_local_oriented_root`
- `v32_no_odd_periodic_sum_of_squares_factorization`
- `v32_exact_c19_symbol_is_local_bond_sum`

## Interpretation

The obstruction is not insufficient ingenuity in choosing three momentum
functions. It applies to any finite number of ordinary real, odd, periodic
Dirac-like coefficients. The exact C19 operator avoids the boundary zero by
using complex oriented bond differences. Those differences naturally map a
site amplitude into separate bond channels; they are not three operators on
the existing four- or eight-component Dirac fiber.

## Candidate routes

### A. Continuum root plus controlled remainder

Keep the V3.1 action and fields unchanged. State the Dirac theorem at the
continuum/root level and carry the exact finite-C19 discrepancy explicitly.

- Preserves the current theory and all proved V3.1 results.
- Local, geometry-derived, and already supported by the fourth-order
  remainder theorem.
- Does not claim an exact finite-spacing Dirac Hamiltonian.
- Suitable for V3.2 if its claim is explicitly narrowed.

### B. Bond-incidence first-order enlargement

Use the exact eighteen C19 bond differences as a site-to-bond incidence map
and its adjoint as the return map.

- Exact and local at finite lattice spacing.
- Geometry-derived rather than fitted.
- Requires additional bond-channel state or an enlarged graded fiber.
- The return-side spectrum, extra kernels/species, mass grading, spin sector,
  current, and relation to the original two-field action would all need new
  proofs.
- This is a new framework/operator research program, not an unchanged-action
  V3.2 proof.

### C. Spectral square root

Use `sqrt(c19TransportSymbol)` directly in momentum space.

- Exact spectrum with a small fiber.
- Nonlocal in position space and does not supply the required directional
  Dirac/current structure.
- Rejected as a V3.2 physical solution.

### D. Wilson-like even term

Add an even momentum-dependent term to remove Brillouin-boundary zeros.

- Can evade the odd-periodic no-go.
- Changes the finite action/dispersion and generally the mass sector.
- Rejected under the unchanged-action lock; it belongs to a separately
  versioned candidate framework if ever investigated.

## Recommendation

Use Route A for V3.2. It is the strongest claim supported by the unchanged
action and existing cubical fiber. Preserve Route B as a separate exploratory
branch for a possible exact finite-lattice incidence theory. Do not place
Routes B, C, or D into V3.2 or describe them as already derived from the V3.1
two-field action.

## Decision

The owner authorized Route A for V3.2. Route B remains a separate prospective
research program and is not part of this release.
