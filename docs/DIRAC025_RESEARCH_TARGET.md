# DIRAC-025 research target: light-speed velocity fluctuation and the clock factor

## Hidden gem found in the DIRAC-001--024 chain

The existing Clifford algebra makes the velocity operator in any selected
direction square to `c_eff^2`. Its possible sharp outcomes are therefore
`+c_eff` and `-c_eff`, even though DIRAC-013/019 prove that a positive-energy
massive packet has the subluminal mean group velocity.

For the velocity direction parallel to a nonzero momentum, the candidate
identity is

```text
velocity variance = c_eff^2 - |v_group|^2
normalized rms fluctuation = sqrt(1 - |v_group|^2/c_eff^2)
```

DIRAC-023 independently proved that the final square root is exactly the
moving phase-clock rate divided by the rest phase-clock rate. If the same
positive-frequency state and inner product close both calculations, Lean can
prove

```text
rms internal velocity fluctuation / c_eff
  = moving phase-clock rate / rest phase-clock rate.
```

This would connect three previously separate consequences of the same action:
the light-speed spectrum of the Dirac velocity operator, the subluminal
transport velocity, and relativistic clock slowing.

## Required safeguards

- Derive the momentum-direction velocity operator from the existing three
  Hamiltonian momentum derivatives; do not insert a new observable.
- Use the same normalized positive-frequency mode and density/current inner
  product already admitted in DIRAC-013/019.
- Prove the operator square, expectation, variance positivity and clock-ratio
  identity for arbitrary nonzero momentum.
- Retain rest and high-momentum limits as controls.
- Do not describe velocity-operator eigenvalues as a classical hidden path or
  claim that the electron literally alternates at light speed.
- Change no action, field, coefficient, C19 weight or update rule.

## Feasibility

High. The required operator-square, current expectation, dispersion and clock
factor are already proved upstream. The remaining work is to place them in
one normalized expectation/variance theorem without changing their shared
state or inner product.
