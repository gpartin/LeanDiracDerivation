# V4-002 five-RCA stop report

## Issue

The action that currently gives the strongest movement response does not keep
the exact localized rest carrier linearly stable. Four reciprocal timing and
phase repairs also fail the same necessary stability screen. V4-002 therefore
does not pass and the nonlinear 100-cycle acceptance run is not authorized.

## Why it matters

Every later V4 claim must use one action and one carrier. A transport result
from an action whose rest carrier has an exponentially growing native mode
cannot be combined with the earlier stationary result to claim one Dirac
electron. V4-003 through V4-010 therefore cannot advance on this dependency
chain.

## Symptoms

All screens used the complete side-five, two-committed-layer, co-rotating
Floquet map with all `Psi` and `chi` coordinates. A passing necessary screen
requires the largest multiplier modulus to be at most `1 + 1e-7`.

| RCA | Tested action delta | Largest modulus | Growth rate | Leading-mode `Psi` power | Result |
|---|---|---:|---:|---:|---|
| 1 | Movement incumbent: face Laplacian plus area oscillator at `6/19` | 1.0030599618422145 | 0.7821541600324176 | 97.01% | Fail |
| 2 | Add positive symplectic face loop at `1/19` | 1.0045886954254180 | 1.1720190614644388 | 99.97% | Fail; worse |
| 3 | Diagnostic negative symplectic face loop at `-1/19` | 1.0131395541511765 | 3.3418186258588300 | 96.57% | Fail; much worse |
| 4 | Add spacetime plaquette at `1/361` | 1.0030655176656964 | 0.7835721080380719 | 97.01% | Fail; slightly worse |
| 5 | Add spacetime plaquette at `1/19` | 1.0031684987566147 | 0.8098533484940180 | 97.08% | Fail; worse |

The unchanged movement incumbent is the least unstable member of this tested
family. Each numerical artifact records the exact tangent-cache SHA-256.

## Expected symptoms

A viable final action must first place every Floquet multiplier on the unit
circle within the declared tolerance, preserve the same eight-site carrier,
and retain containment and conservation. Only then may it enter the nonlinear
100-cycle perturbation, timestep, and box-refinement gate.

## Root cause

The remaining unstable direction is predominantly a `Psi` radial/phase mode,
not a missing bulk `chi` inertia mode. The tested symmetric face-loop and
spacetime-loop repairs act in the wrong response direction: both signs and
both natural geometric scales leave the mode unstable and generally increase
its growth. This is a scoped obstruction for this action family and this
localized branch. It is not a no-go theorem for every reciprocal C19 action.

## Corrective action

Stop coefficient tuning inside this failed timing-loop family. The next
research program should construct the localized nonlinear branch and its
action response together, beginning from native band-edge/delocalized
nonlinear modes, and should test an intersite nonlinear coupling family rather
than another onsite or scalar stiffness correction. This follows established
discrete-wave results: moving localized modes can be pinned by onsite
nonlinearity and recovered by suitable intersite response, while nonlinear
localized branches can be continued from band-edge modes in square and simple
cubic lattices.

Relevant primary references:

- [Moving localized modes in nonlinear lattices, Phys. Rev. B 47, 14228 (1993)](https://doi.org/10.1103/PhysRevB.47.14228)
- [Generalized discrete nonlinear Schrödinger model, Phys. Rev. E 54, 5788 (1996)](https://doi.org/10.1103/PhysRevE.54.5788)
- [Discrete breathers in square lattices from delocalized nonlinear vibrational modes, Phys. Rev. E 107, 034214 (2023)](https://doi.org/10.1103/PhysRevE.107.034214)
- [Delocalized nonlinear vibrational modes and discrete breathers in a simple cubic lattice, Phys. Rev. E 109, 014215 (2024)](https://doi.org/10.1103/PhysRevE.109.014215)

Any future action remains subject to the same real, local, reciprocal,
cubic-symmetric, no-Dirac-bolt-on rules. Per the user-defined stop condition,
this task stops here after five consecutive RCAs without a V4-002 closure.
