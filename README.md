# Lean LFM-to-Dirac Derivation V3 + local V4 research

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository contains one cumulative production proof line. V2 derives the
complete free linearized Dirac equation from the declared two-field C19 lattice
action. V3 continues from that theorem and derives the Lorentz/spin-one-half
representation carried by both exhaustive cubical copies.

The default import is unambiguous:

```lean
import LeanDirac

#check LeanDirac.V2.complete_first_principles_free_dirac
#check LeanDirac.V2.lorentz_spin_half_from_native_action
```

The checked-out tree does not contain the historical numbered V1 proof or its
downstream consequence stages. That development remains available in Git
history at tag `v1-legacy-final`.

## Starting point

At each lattice site the native fields are `Psi_i, chi_i : C^3`. The declared
geometry is C19: six face bonds of weight 2 and twelve edge bonds of weight 1.
The geometric normalization is `chi0 = 19`, identified with the nineteen
C19 cube modes (center, six faces, and twelve edges). The chi kinetic
normalization is declared as `B = 1197`; equivalently the existing
parameterization writes `kappa = 1/63` and `B = chi0 / kappa`. V2 does not
yet derive this remaining dimensionless normalization.

That declaration belongs to the full substrate dynamics, but it is not an
independently established prerequisite for the free-Dirac result. The
finite-lattice transverse theorem is uniform in `B`: at the proved vacuum its
linear transverse `Psi` sector has no linear `chi` motion, and its Dirac mass
and transport generator contain no numerical use of `1197`. The value of `B`
remains relevant to live-`chi` inertia, nonlinear response, and any future
localized moving-electron construction.

The proof starts from one real reciprocal action `L = T - U`, derives both
Euler--Lagrange equations, and derives the centered two-layer update. The full
action is stated in [docs/FRAMEWORK.md](docs/FRAMEWORK.md).

## Result

The cumulative theorem is
`LeanDirac.V2.complete_first_principles_free_dirac`. Its dependency chain
derives:

- reciprocal native field variation from the action;
- an exact native zero-energy vacuum and the full finite-lattice tangent;
- the closed four-complex-component phase space;
- the cubical Clifford root and an exhaustive direct-sum decomposition into
  exactly two rank-four copies;
- the standard alpha/beta form, nonnegative mass, and three transport terms;
- centered/Verlet time evolution with an explicit vanishing time defect;
- the position-space free Dirac equation with the explicit C19 remainder;
- the positive-frequency density/current bridge and its controlled remainder.

There is no external, prepared, period-two, maintained, or Dirac-shaped
background. The expansion point is one exact zero-energy vacuum solution
proved from the native action. Selecting a representative of the degenerate
vacuum family is not claimed to be a cosmological origin mechanism.

Within the declared cubical root, the two four-component sectors are a
formal prediction, not discarded doubling. Lean proves that both sectors
are invariant, disjoint, exhaustive, and carry the same free Dirac operator.
The interpretation that an electron measurement accesses one sector without
revealing which one was accessed is a physical hypothesis; it is not itself
a theorem in this repository.

The theorem is a free linear tangent result. It does not claim a nonlinear
localized electron or any later physical consequence, and it does not derive
or require the numerical value `B = 1197` within that transverse free sector.

## V3 Lorentz and spin extension

V3 continues from the proved free-Dirac boundary without
changing the action, equations, C19 stencil, vacuum, or readout.  It derives
the spatial rotation generators directly from the cubical gamma operators,
proves that their unique nonzero closure scale is `1/2`, and then proves the
spin-one-half spectrum, Casimir `3/4`, density-preserving finite rotations,
`2*pi` sign reversal, `4*pi` return, and the rotation/boost Lorentz algebra.
It also proves the finite rotation group law and generator derivative,
intertwines rotations and boosts through both exhaustive cubical copies,
proves covariance of the full continuum Dirac symbol while four-momentum and
the spinor transform together, and constructs the Pauli--Lubanski vector from
four-momentum and the derived Lorentz generators before obtaining its rest
Casimir `-3 m^2 / 4`.

The cumulative local theorem is
`LeanDirac.V2.lorentz_spin_half_from_native_action`.  The exact chain and
claim boundary are documented in
[docs/LORENTZ_SPIN_CHAIN.md](docs/LORENTZ_SPIN_CHAIN.md).

This is an infinitesimal continuum-sector Lorentz theorem. It does not claim
exact Lorentz invariance at finite lattice spacing, a localized nonlinear
electron, electromagnetic coupling, or a dynamical rule selecting one copy.

## V4 localized-electron research

V4 begins from the unchanged V3 action and does not alter any released V2/V3
theorem. Its first proved gate constructs an exact, freely updated, fixed-center
localized `Psi`--`chi` orbit on every periodic C19 box of side at least three:

```lean
#check LeanDirac.V4.exact_localized_rest_seed
```

The orbit includes the eight-site carrier, its live `chi` halo, both committed
time layers, and ordinary exterior vacuum. Both fields advance under the native
centered update; neither is held or reset. The proof is symbolic in `B` and
requires only `B > 1`, so it does not derive or require the numerical value
`1197` for this rest solution.

This is deliberately not yet called an electron theorem. Stability, the map
from this same localized carrier to the V3 four-component sectors, operational
spin one-half, translation, and same-carrier Dirac current remain open V4
gates. See [the V4 gate ledger](docs/V4_LOCALIZED_ELECTRON.md).

## Reproduce

Install Lean through `elan`, then run:

```bash
lake build LeanDirac LeanDiracAudit
python scripts/verify_release.py
```

The verifier compiles the complete dependency closure, checks the sealed
sources, rejects `sorry` and custom axioms, and admits only `propext`,
`Classical.choice`, and `Quot.sound`.

See [the proof chain](docs/PROOF_CHAIN.md), [claims](docs/CLAIMS.md), and
[reproducibility instructions](docs/REPRODUCIBILITY.md).
