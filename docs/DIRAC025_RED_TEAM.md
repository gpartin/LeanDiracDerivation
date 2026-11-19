# DIRAC-025 red-team review

## Verdict

`PASS_SCOPED_VELOCITY_VARIANCE_CLOCK_IDENTITY`

No critical or high finding remains. DIRAC-025 changes no native action,
field, coefficient, C19 weight, update rule, dispersion, current, or clock.

## Injection and circularity checks

- The longitudinal velocity operator is the unit-momentum combination of the
  three Hamiltonian momentum derivatives already proved in DIRAC-013.
- Its square is proved from the inherited Clifford anticommutators and the
  independently proved normalization of the momentum direction.
- The mean is derived from the existing positive-frequency current theorem.
- The second moment is evaluated with the same state, density and inner
  product. The variance is their standard difference; the desired result is
  not present in its definition.
- The moving-clock ratio comes unchanged from DIRAC-023. Lean proves that it
  equals the normalized rms velocity fluctuation for arbitrary nonzero 3D
  momentum; the two sides are not defined from one another.
- No hidden path, sign switch, probability assignment, fitted coefficient,
  phase reset, projector or new observable register is introduced.

## Independent controls

An independent NumPy reconstruction tests five oblique momenta and both
positive-energy spin states. The worst error is below `5e-11`. It reproduces
the exact speed/clock pairs `3/5 -> 4/5`, `4/5 -> 3/5`, and
`sqrt(3)/2 -> 1/2`. An unnormalized direction, a transverse substitution and
the wrong rest mass all fail by finite margins.

## Claim boundary

The result is an operator-expectation identity in the previously derived
continuum sector. It is not evidence that a classical particle literally
travels back and forth at `c_eff`, and it does not establish an exact native
finite-spacing identity, SI calibration, or localized nonlinear electron.
