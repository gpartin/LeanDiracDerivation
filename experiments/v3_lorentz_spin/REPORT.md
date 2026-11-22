# V3 Lorentz and spin-one-half release report

```text
V2_FIRST_PRINCIPLES_FREE_DIRAC             = PASS (INHERITED)
NATIVE_CUBICAL_ROTATION_GENERATORS         = PASS
UNIQUE_NONZERO_SU2_SCALE                   = PASS (1/2)
HERMITIAN_SPIN_GENERATORS                  = PASS
FINITE_ROTATION_GROUP_AND_DENSITY          = PASS
TWO_PI_SIGN_AND_FOUR_PI_RETURN             = PASS
LORENTZ_LIE_ALGEBRA                        = PASS
BOTH_EXHAUSTIVE_CUBICAL_COPIES             = PASS
FULL_CONTINUUM_DIRAC_SYMBOL_COVARIANCE      = PASS (INFINITESIMAL)
GENERAL_PAULI_LUBANSKI_ORTHOGONALITY        = PASS
REST_PAULI_LUBANSKI_CASIMIR                 = PASS (-3 m^2 / 4)
EXACT_FINITE_LATTICE_LORENTZ_INVARIANCE     = NOT CLAIMED
LOCALIZED_NONLINEAR_ELECTRON                = SEPARATE OPEN GATE
```

V3 changes no native field, action term, coefficient, C19 weight, equation,
vacuum, or four-component readout. It begins with the cumulative V2 theorem
and derives rotations and boosts from the cubical gamma operators already
obtained from the native action.

The release closes four weaknesses found by hostile review. It proves the
finite angle-group law and generator derivative rather than relying only on
2π/4π endpoints; it verifies both exhaustive rank-four copies; it transforms
the complete continuum Dirac symbol together with four-momentum rather than
checking gamma matrices alone; and it constructs the Pauli--Lubanski vector
at arbitrary momentum before deriving the rest-frame spin-half invariant.

The result is scoped to the free low-momentum continuum sector. It is not a
claim of exact finite-spacing boost symmetry, nonlinear localization,
electromagnetic interaction, copy selection, or cosmological origin.
