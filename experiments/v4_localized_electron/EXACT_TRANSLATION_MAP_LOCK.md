# V4 exact translation map lock

Status: **LOCKED TARGET; V4-006 OPEN**
Date: 2026-11-24

## Non-negotiable movement target

The movement endpoint is the literal same-assignment translation of the joint
native `Psi`--`chi` carrier by one lattice cell:

1. every one of the eight occupied carrier sites moves to its corresponding
   face-neighbor site;
2. the translated carrier has exactly the same internal `Psi` and `chi`
   values, up to its freely solved native clock phases;
3. a surrounding precursor or tail is permitted only when it is part of the
   freely solved native state and translates with the carrier;
4. intermediate numerical layers may prepare the receiver, but the completed
   handoff must return the identical translated carrier;
5. no receiver preload, projection, reset, copied force, selected direction,
   corner rule, spinor rule, or Dirac force may execute the handoff.
6. the relative-periodic equality applies to both committed layers of the
   second-order state, not only to a single displayed configuration.

No alternative movement definition may replace this target without explicit
authorization from G. D. Partin.

## Native mechanism under test

The current physical hypothesis is a make-before-break handoff.  A weak
leading `Psi`--`chi` precursor communicates the complete three-prong live
frame `[chi, Psi, chi wedge Psi]` to neighboring sites.  The receiving field
is progressively oriented before the core arrives.  The action must derive
this communication reciprocally and locally.

The strongest tested representation is the regularized dual-frame map

`M_ij = F_i (F_j^dagger F_j + kappa I)^-1 F_j^dagger`,

with `F=[normalized chi, normalized Psi, normalized chi wedge Psi]`.  Unlike
three independent projections, this map accounts for nonorthogonal live
frames during a handoff.

## Current verdict

The compact canonical carrier does not execute the exact shift.  A canonical
first-shell precursor reduces the nonlinear recurrence defect to about
`0.675`--`0.706` in all axes and both signs, but does not close it.  The prior
chi live-frame action plus a solved precursor and clock also does not close.
Adding a matching Psi kinetic broadcast worsens the bounded solve and is
retired.

The dual-frame chi candidate with the derived `kappa=1/63` regularizer is the
strongest structurally correct trial.  Its bounded nonlinear profile retains
the exact core and uses a precursor about `0.59%` of the carrier norm.  The
remaining action residual is about `82%` chi, `63%` carrier-core, and `66%` in
the second complex component.  A 200-iteration implicit-next-layer audit
requires a correction about `17.96%` of the desired cell step and is still
not fully converged.  Therefore it is not admitted and V4-006 remains open.

Tiny dual-frame regularizers can make the relative action residual look small
by making the kinetic metric nearly singular and extremely stiff.  Those
results are diagnostics only and are not admissible improvements.

## Next admissible action requirement

Retain the exact endpoint map and the dual-frame representation.  Derive a
real reciprocal **spacetime bond connection** that transports the joint
`Psi`--`chi` frame between the sender at one committed layer and the receiver
at the next committed layer.  It must be activated by native signed bond
current, sum all C19 directions without choosing one, reduce exactly to the
rest action when bond current vanishes, and produce a positive/well-posed
kinetic sector.  Its first cheap gate is the exact translated-layer update,
measured after solving the implicit update—not an unpreconditioned action
residual.  Only a passing exact-map candidate advances to repeated handoffs.
