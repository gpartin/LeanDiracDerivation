# V4-006 — Current advance and the native two-clock interference

**OPEN — frozen V4-002–005 research action.** C19 geometry, complex Psi and
complex chi, six-face oscillator, dt=1/256, and fixed 17564-update
comparison interval. This is local midpoint initial-data research, not a
whole-cell orbit, a Dirac-current identification of the nonlinear carrier,
or a new action.

## Issue

The prior two-charge/current-matched midpoint preparation could deliver
the required initial flux, but Psi lost most of that rate within a few
native updates even while chi retained it.

## Why it matters

A complete handoff must maintain the current over the full crossing
interval and repeat in the other directions. Matching one or two local
intervals is an independently useful diagnostic, not that gate.

## Symptoms

The previous current-matched preparation gave Psi 89.70%, 77.88%, 64.86%
of the chart's initial midpoint rate on three unrestricted updates.
The next initial-data inverse problem matches both global phase charges,
both initial cut currents, and both first current advances of the same
target chart. The new exact source of the next layer is the native DEL,
not the prescribed path. An initial-data constraint is not used after
release.

| Free native update | Prior launch Psi | New matched-energy Psi | New matched-energy chi |
|---|---:|---:|---:|
| 1 | 89.70% | 100.02% | 100.04% |
| 2 | 77.88% | 98.28% | 100.51% |
| 3 | 64.86% | 94.90% | 101.44% |
| 4 | not run | 90.01% | 102.94% |
| 5 | not run | 83.78% | 105.12% |

The unconstrained-energy screen reaches the six current/charge constraints
to 9.64e-11, but uses a native interval energy 2.286% above the rest
reference, slightly above the prior launch. The seven-constraint repeat
matches the prior native energy to about 1.5e-12 relative to rest, with
constraint norm 2.43e-9. The worst exact DEL over the release is
9.98e-8 (first interval reused from the root), with later solves under
1.3e-11 and global charge defects below 2.33e-11. This is improvement
at matched native energy, not an established physical Dirac kinetic-energy
law: the prior 2.175% above rest energy remains unexplained.

The chosen midpoint is still only a trial state; it is not proved to arise
from a freely moving electron starting at rest or from quiet exterior.
The current targets are obtained by differencing the earlier chart's native
charge at the independent Dirac comparison interval. They are not a
derived full Dirac wavepacket current.

## Expected symptoms

On a genuine same-carrier trajectory, both charge currents sustain the
correct integrated flow, the state traverses a full integer-cell
translation without resets, and the measured four-component motion and
continuum corrections agree with the independent Dirac comparator.
The initial-energy comparison must be justified by native dynamics,
not imposed from the continuum by renaming a substrate energy.

## Root cause

### Exact current advance on the original reciprocal action

Write the full native discrete Euler–Lagrange equation on three layers as
G(q,r,s)=0, with midpoint q fixed. The next current is H(r,s(r)).
Differentiate G=0 for any admissible initial-layer change dr:

    G_r dr + G_s ds = 0
    dH = H_r dr + H_s ds.

Solve the adjoint equation y G_s = H_s; then

    dH = (H_r - y G_r) dr.

The derivative is proved in Lean, conditional on differentiability and an
actual root curve. Numerically both adjoints were solved on the full
native state, worst relative defect about 1.22e-6, using the rest tangent
as a preconditioner only. Each nonlinear line-search trial independently
solved the following native layer. No forces were fitted. The first
native current change was initially too negative; matching it makes the
first free step agree exactly with the chart, but later curvature
remains.

### What exactly is making the current turn back?

For either directly rotating face field F=Psi or W=(chi wedge Psi)/chi0,
the exact two-layer centered-clock coordinates are (a=exp(i theta))

    P_n = (F_(n+1) - conjugate(a) F_n)/(2i sin(theta))
    N_n = (a F_n - F_(n+1))/(2i sin(theta)).

This is an invertible readout of the same two committed layers, not a
third field, new spinor rule, or the two historical Clifford copies.
The signed direct face current between sites i and j is

    I_n = 2 (B/dt) sin²(theta) Im C_n,
    C_n = conjugate(a) (P_j† N_i - P_i† N_j).

**All same-clock-frequency contributions cancel exactly.** The exact
nonzero current in these two matched channels is the interference of the
two opposite clock coordinates. This identity is checked in Lean and
against the independent native face current; the observed maximum
relative discrepancy is about 1.1e-12.

For the actual clock defect R_(n+1)=F_n+F_(n+2)-2 cos(theta) F_(n+1),
the exact frequency-coordinate update is

    P_(n+1) = a P_n + R_(n+1)/(2i sin(theta))
    N_(n+1) = conjugate(a) N_n - R_(n+1)/(2i sin(theta)).

Therefore if the native forces provide no defect drive, the interference
rotates as C_(n+1)=exp(-2i theta) C_n. Its signed current has period
pi/theta = 42.319 updates at this dt, half the rest clock period. It
naturally returns to zero and reverses; there is no unforced constant
nonzero current on this oscillator. This is a theorem about the matched
face term, not a universal no-go for the full nonlinear two-field action.

For the actual driven case define D_n=C_(n+1)-exp(-2i theta) C_n and let
a2=exp(-2i theta), K=2(B/dt)sin²(theta). The exact second-difference law is

    I_(n+1)-2 Re(a2) I_n+I_(n-1)
       = K Im(D_n-conjugate(a2) D_(n-1)).

A locally constant nonzero I needs a drive of 4 sin²(theta) I. At the
first free advances in the improved launch, the direct Psi native drive
supplies only about 13–25% of that constant-rate requirement, while W
supplies 36–47%. The actual full current is the two matched contributions
plus independently measured other action terms. This explains the
observed turnback after matching the first current advance. It is not
evidence that chi has stopped orienting the receiver; chi total current
in the measured local release is increasing.

The same native frequency recursion gives a more direct requirement than
fitting successive current samples. For the negative clock coordinate
to co-rotate with the positive clock under a zero-envelope-rate control,

    N_(n+1) - a N_n = [4 sin²(theta) N_n - R_(n+1)]/(2i sin(theta)).

A slowly moving envelope permits a small nonzero right-hand side, but
the raw closure identity and the zero-rate control are exact. In the
current-advance preparation the measured native R has only about
29.7–29.9% of its required co-rotation component for Psi and W. The
full relative co-rotation defect is 0.72–0.73. This is a concrete
missing *native phase response*; it does not mean that the negative
clock coordinate is a separate field or fermion copy. The algebra is
verified in Lean.

### Which part of the action could supply the drive?

At a committed center q, let J_W be the real derivative of the native
wedge field, L6 the positive six-face Laplacian and let m be the exact
mixed finite-layer wedge product increment. With R the two-field
clock-defect vector, the matched face DEL is exactly

    G_face = -(B/dt)[A(q) R + J_W† L6 m],
    A(q) = L6 diag(1,B) + J_W† L6 J_W.

This follows by varying the same positive oscillator and applying the
finite-layer wedge identity. No new response term is postulated. At the
tested native roots, the equilibrated metric has rank 1500 of 1500, and
the inverse decomposition of R among the remaining native action
forces reconstructs R within 1.1e-11 relative error. At the first two
tested curvature evaluations, the existing direct prong communication
is the largest positive direct-Psi clock-drive source:

| contribution | first | second |
|---|---:|---:|
| required constant direct-Psi drive (approximate) | 110,600 | 109,000 |
| direct-prong term | +29,445 | +33,000 |
| base/core action | -10,135 | -9,176 |
| mixed wedge increment | -4,666 | -4,674 |
| remaining terms (sum) | +52 | +43 |
| total supplied | +14,695 | +19,193 |

These values are current rate second differences at the chosen
midpoint, not force coefficients. The direct-prong term is the strongest
existing candidate for the missing recurrence response.

## Corrective action

The current action's direct-prong kinetic metric and the matched face
oscillator share the same outer B scale. An explicit partial derivative
of the two *actual implicit steps* when varying only their relative
coefficient gives the following derivatives (at coefficient 1, fixed
initial pair).

- Increasing the direct-prong coefficient improves the negative Psi
  second current difference by about +17,973 per unit, but worsens its
  first advance by -113,909 per unit.
- It also changes chi's current and both native charges for the fixed
  pair. Native two-step controls at 0.9 and 1.1 of the coefficient
  independently reproduce the predicted trade-off (about 2.15% relative
  linear prediction discrepancy on the finite current changes). The direct-prong variation is zero on rest up to the numerical
  residual, but that is not enough to preserve the V4-002 stability
  or V4-005 transport gate.
- Hence no coefficient change is admitted, and a one-number fit to the
  desired current curvature is not a derivation.

The admissible next mathematical target is the native reciprocal
response that maintains the slowly varying interference drive, while
preserving both charge constraints, rest and the already frozen readout.
The fixed-midpoint preparation screen separately tests two successive
advances at the same energy; its outcome is recorded below. If it cannot
close, do not simply fit more time samples. Allow genuinely free carrier
shape/velocities or derive a reciprocal change from the identified
direct-prong and clock-metric interplay, then retest the full chain.

### Two-advance preparation screen

The exact native adjoint was propagated through a second implicit step,
at the same energy, to match one more current change. At the fixed
midpoint this was not an efficient repair: the residual norm fell only
from 0.0180750 to 0.0166688 after three accepted corrections, while
earlier current and energy agreement deteriorated. The last observed
second-advance mismatch was -1.502% of the initial Psi current and
+0.348% of the initial chi current. All tested full Newton increments
were rejected. The expensive fixed-shape screen was stopped and the
console evidence preserved; it produced no curvature-matched initial
pair. The converged one-advance launch remains the best retained result.
This does not exclude a freely solved carrier shape or a native action
repair.

Artifacts: current_advance_inverse{,_matched_energy}.json and NPZ
states, clock_interference_drive_{old,advance,matched_energy}.json,
native_clock_drive_attribution.json, prong_clock_drive_response.json,
and current_curvature_screen.json (stopped-run evidence).
Formal research modules: CurrentAdvanceSensitivity.lean and
FacePhaseMomentum.lean. Canonical V40 and the incumbent V4 action are
unchanged. V4-006 remains OPEN.
