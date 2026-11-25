# V4-002 physical-launch RCA

## Issue

The complete-frame action preserves the exactly prepared fixed-center orbit,
but the native movement preparation excites a growing `Psi` amplitude/phase
response. The first departure is not primarily a failure of `chi` to follow
or a loss of the three-component orientation frame.

## Why it matters

V4 requires one localized `Psi`--`chi` carrier to retain its identity while
carrying the Dirac density and current. An exact unperturbed clock is not a
stable electron if the movement required by that claim activates a native
direction that changes the clock amplitude and phase on every handoff.

## Symptoms

The exact `6/361` complete-frame audit covers both cubical copies, both signs,
three axes, a diagonal, and both committed layers.

- One-update Dirac-sector leakage is `9.4752485e-5`.
- The worst one-update configuration error is `1.8435816e-4`.
- In that first departure, `99.999908%` of residual power is in `Psi`; only
  `0.0000915%` is in `chi`.
- The maximum-error site is one of the eight occupied carrier corners.
- On those corners, `96.3650%` of the `Psi` residual is amplitude/phase along
  the existing `Psi` ray, `3.2830%` points toward the in-plane `chi` companion,
  and only `0.3519%` leaves the local `Psi`--`chi` plane.
- The residual is spatially distributed: `35.8723%` of its total power is on
  the eight occupied sites and `64.1277%` is in their surroundings, although
  the single largest site is a carrier corner.
- After 100 internal cycles, every tested launch has norm ratio at least
  `3.4920812`; the maximum is `4.8200822`, with maximum aligned shape error
  `4.6536528`.

## Expected symptoms

A passing action may produce a small controlled lattice remainder, but the
same frozen readout must keep the along-ray amplitude/phase response bounded
and convergent to the C19/centered-time Dirac comparator. The receiving shell
must build on the previous push instead of feeding an increasing along-ray
error back into the occupied corners.

## Root cause

The complete native frame is present. `chi cross Psi` communicates transverse
orientation, while `chi^dagger Psi` supplies the missing parallel relation and
clock coordinate. Those additions greatly reduce the old error and keep the
out-of-plane prongs tight.

What remains is a different degree of freedom: transport must decide the
signed amount and phase of `Psi` to place along the already-correct ray. The
current action launches that transfer almost correctly, but its native
phase-space connection does not uniquely route the along-ray response through
both C19 face and edge bonds. The bare action has negative generalized
curvature in this charge-neutral in-plane direction, while the existing
tight-frame/orientation terms are nearly blind to it. The small first-step
amplitude/phase mismatch therefore receives the wrong net restoring response
and grows on successive pushes.

Mechanically, the carrier has a well-formed three-prong rail but insufficient
torsional control along the rail. `chi` is not the initiating error: it follows
the malformed `Psi` handoff and later participates in the feedback cycle.

## Corrective action

Do not add another unsigned frame-disagreement penalty or simply increase the
existing stiffness. Derive and rank-test the smallest real, local, reciprocal,
rest-null, odd-in-native-current `Psi`--`chi` phase-space response that acts on
the measured along-`Psi` amplitude/phase residual. It must:

1. have zero first variation on the exact rest orbit;
2. have correct-sign response on the measured residual without fitting its
   coefficient to Dirac;
3. derive reciprocal `Psi` and `chi` equations from one action;
4. preserve both copies, C19/cubic symmetry, positive kinetic mass, and the
   already-correct transverse frame;
5. reduce the one-update residual, then the one-cycle error, before any new
   100-cycle run.

This diagnosis does not prove that no alternate preparation or reciprocal
action works. No action is changed or promoted by this RCA.

