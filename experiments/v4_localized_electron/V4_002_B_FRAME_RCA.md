# V4-002 complete-frame stability repair

## Issue

The exact rest orbit existed, but every physical two-layer packet launched by
the prior `6/19` complete-frame action reached a growing amplitude/phase
response. V4-002 therefore failed even though the first-step orientation
leakage was small.

## Why it matters

Every later V4 stage needs the same localized carrier to remain identifiable
without projection or runtime support. A growing launch would make any later
spin, current, or movement comparison meaningless.

## Symptoms

The prior action reached phase-space norm ratio `4.82008` and aligned error
`4.65365` over 100 internal cycles. The first departure was overwhelmingly an
in-plane `Psi` amplitude/phase response rather than a missing three-prong
orientation channel.

## Expected symptoms

Both cubical copies, both momentum signs, three axes, and a diagonal should
remain bounded for 100 internal cycles. The exact rest orbit, native phase
charge, discrete energy diagnostic, timestep refinement, and larger-box
response must remain intact.

## Root cause

The complete face-frame kinetic channel had been normalized by the fraction of
C19 directions that are faces (`6/19`). That is the transport fraction, but it
does not normalize the live material frame's inertia. The native `chi` field
already declares the independent kinetic normalization
`B = chi0/kappa = 1197`. With the smaller coefficient, the physical packet
could excite the inherited in-plane response much faster than the interface
frame could carry it.

This diagnosis was found from the full action tangent, not from fitting a
Dirac force. Direct differentiation also exposed that one inherited tangent
cache did not equal the declared action; all final evidence was therefore
rebuilt from the action itself.

## Corrective action

Use the already-declared `B=1197` as the common outer normalization of the
complete six-face frame channel. Keep the previously derived `6/chi0^2`
spacetime-face term and unit parallel-frame completion. No new field,
direction, corner label, copy selector, projector, Dirac matrix, or fitted
coefficient is introduced.

The exact-action cache agrees with direct automatic differentiation to
`2.82e-16`. Sixteen physical launches remain between `0.5` and `2` in phase
space over 100 internal cycles (worst ratio `1.23388`), and one-step leakage
falls by a factor `0.25002` when `dt` is halved. The unrestricted nonlinear
short run, full-state Noether continuity, discrete-energy diagnostic, and
side-5/7/9 stability checks pass. The `53.7%` accumulated 100-cycle Dirac
comparator discrepancy is not hidden; transport accuracy and the fixed
four-component readout remain V4-003/V4-005 obligations.
