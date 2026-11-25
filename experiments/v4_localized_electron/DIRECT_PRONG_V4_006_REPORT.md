# Direct-prong V4-006 report

## Verdict

V4-006 remains **OPEN**.  The latest direct component-communication action
produces a real short-time Dirac response and can be made linearly stable,
but it has not translated the complete localized Psi--chi carrier by one
lattice cell or repeated that handoff.

## What passed

- All real and imaginary components of `Psi_x/y/z` and `chi_x/y/z` enter one
  local C19 best-matching translation velocity.
- Every component is advected into the identically labelled neighbor
  component; no internal-label mixer, target direction, Dirac matrix, carrier
  projector, or runtime track occurs in the action.
- The discrete C19-Dirac tangent comparator passes for both copies, both signs,
  the three axes, and a diagonal.  The worst DEL residual is
  `2.62268694e-4`.
- The nonlinear implicit action solve passes one update for all eight tested
  copy/direction cases, using continuation for three solver-stalled cases.
- A representative nonlinear packet remains under the declared short gate for
  four updates: final Dirac error `0.0157931`, leakage `0.000780025`.
- Completing the omitted positive velocity-regularization energy preserves
  the action form but does not by itself remove the long-time wake.
- Reducing the action to one common physical velocity and adding the previously
  derived rest-null even-transfer, density-balance, and cubic loop-curvature
  sector removes exponential growth.  The largest computed Floquet modulus is
  `1.0000000000000018`, and the exact rest DEL RMS is
  `3.611645973283916e-13`.

## What failed

- Giving the three prongs independent material velocities creates a growing
  first-shell mode.  At the `3/4` relative-prong weight, the leading multiplier
  is `1.011458828` per update and the ten-cycle error diverges.
- Adding the positive regularization energy reduces but does not remove that
  mode.
- Using both four-component copies at relative phases `1`, `i`, `-1`, and
  `-i` does not cancel the wake.
- One common velocity is stable, but after ten cycles its physical Psi-density
  transfer is `[0.0369049, 0.265617, -0.491475]` cells while the independently
  evolved discrete Dirac target is `[0.0385440, -0.0161222, 0.0180828]`.
  The forward component is close, but the transverse components are not.
- The joint Psi--chi density moves `[-0.176955, 0.0680770, -0.0138257]`, so
  chi does not co-translate with the Psi carrier.
- Balanced two-copy preparations do not repair that displacement.
- A positive penalty on disagreement between per-prong and common velocities
  leaves the ten-cycle mismatch essentially unchanged.
- No full one-cell same-pattern handoff, second handoff, 100-cycle nonlinear
  gate, box refinement, or Lean theorem has passed.

## First-departure and orientation-link refinement

The first native/Dirac departure is now resolved at update two.  It is not
initially a chi lag: `99.9791%` of its power is in Psi, concentrated almost
equally in the first two complex components, and the receiving shell contains
more error power than the occupied carrier.  Chi is only `0.0209%` of that
first error.  Chi separation is therefore a downstream consequence of an
incorrect incoming Psi orientation.

Three native oriented-area completions were tested:

1. A full material wave for `w=(chi cross Psi)/chi0` preserves rest and lowers
   the ten-cycle shape error from `1.4210` to `1.2609`.  At update 64 it lowers
   the total error from `0.22571` to `0.14696`, but it also slows the correct
   forward Psi response.
2. Completing that wave with its C19 spatial spring gives an exact geometric
   balance.  The cube eigenvalue is `27`; directed action counting doubles it
   to `54`, deriving the local frequency square `19^2-54=307` with relative
   residual `8.76e-17`.  This preserves rest but worsens moving-sector leakage
   and reverses the forward response.
3. Keeping only the convective orientation connection avoids extra inertia
   but leaves the ten-cycle mismatch essentially unchanged (`1.4191`).

This retires scalar stiffness, a second independent orientation wave, and a
bare convective cross-term as complete repairs.  The useful part of the full
area-wave response must be represented without adding its longitudinal
inertia.

## Complete live-frame communication test

The update-two error was decomposed in the instantaneous live `Psi`--`chi`
frame.  `48.5808%` of the `Psi` error is parallel to `chi` and `51.4192%` is
transverse.  This proves that neither the wedge channel nor a parallel-only
channel can be a complete repair.  The pair

`chi dagger Psi`, `chi cross Psi`

is the minimal native frame carrying both parts; this is a diagnostic identity,
not a prepared spinor or target rule.

Five reciprocal action realizations were then tested without fitting a Dirac
coefficient:

1. A full live-gradient penalty on relative prong velocities slightly improves
   the first error but overconstrains legitimate internal rotation.  Its
   ten-cycle shape error is `3.4064`, so it is rejected.
2. A smooth projective oriented-area kinetic is positive, rest-null, and
   translation-null, but is blind to the parallel half of the measured error.
   Its ten-cycle result is indistinguishable from the incumbent (`1.42101`).
3. A material wave for the parallel overlap alone lowers the headline
   ten-cycle shape error to `1.31284`, but worsens update-64 error and nearly
   removes the correct forward displacement.  It is not a movement repair.
4. Adding the missing positive C19 bond square for `chi dagger Psi`, with the
   same `1/(2 chi0^2)` normalization as the canonical area bond, lowers the
   shape error to `1.26279`.  It also raises exterior leakage to `0.10205` and
   leaves large transverse drift.
5. Completing both live-frame coordinates as material waves gives the best
   headline shape error in this family, `1.19937`, but the first threshold is
   crossed one update earlier, update-64 error rises to `0.31755`, and the
   final axial `Psi` displacement reverses to `-0.05303` rather than the Dirac
   target `+0.03854`.
6. Keeping only the signed convective connection for both frame coordinates
   avoids duplicate inertia, but the parallel coordinate is zero on the rest
   orbit and enters only beyond linear order.  The tangent result therefore
   remains `1.41914`, indistinguishable from the wedge-only connection and the
   incumbent.  It does not supply the missing linear receiver response.

Two additional velocity-source controls fail sharply.  Letting `Psi` alone
determine velocity leaves a near-null spatial direction and is linearly
unstable.  Letting density-activated `chi` fill that rank while excluding the
factor `B` remains unstable.  Thus the stable incumbent's full joint metric is
needed for bounded evolution, but its response must be altered without simply
adding redundant frame inertia or stiffness.

No tested complete-frame term closes an integer handoff.  Canonical V40 is
unchanged and no research action is admitted.

## RCA

**Issue:** The action transfers forward Psi amplitude but does not transport
the complete eight-site Psi--chi carrier as one object.

**Why it matters:** V4-006 requires the same localized carrier, mass, spinor
readout, density, and current to move together.  Fractional center-of-density
motion is not a completed lattice-cell handoff.

**Symptoms:** The common channel is Floquet-stable and its x-directed Psi
transfer nearly matches the discrete Dirac comparator, but the first error is
an in-plane Psi orientation/amplitude defect on the receiving shell.  Large
transverse Psi motion and oppositely directed joint Psi--chi motion follow.
Independent prong transport removes neither defect and destabilizes the shell.

**Expected symptoms:** At the Dirac-derived integer handoff time, the complete
same-assignment eight-site pattern should reappear one site away, with Psi and
chi centers agreeing, transverse current vanishing for an axial launch, and a
second identical handoff following without support.

**Root cause:** The tested action has two incompatible responses.  Its common
material channel supplies stable signed forward transport, while its relative
prong channel supplies extra directional freedom as independent kinetic
coordinates and destabilizes the shell.  Removing that channel restores
stability but leaves the incoming in-plane Psi orientation underdetermined.
Chi consequently follows a malformed Psi handoff rather than initiating the
failure.  The first defect is split nearly equally between the live parallel
and transverse `Psi`--`chi` coordinates.  Complete-frame waves and bonds act
on both parts, but they add redundant inertia/stiffness and redirect rather
than repair the already-correct forward current.  The missing object is a
signed, current-sensitive transfer of the complete frame, not an unsigned
penalty on frame disagreement.  A full induced-area wave supplies part of the
orientation response but also adds longitudinal inertia; its isolated
connection is insufficient.
This is an ablation-supported action-class diagnosis, not a universal
two-field no-go.

**Corrective action:** Use the complete-frame decomposition as the inverse
requirement and derive its *odd-in-current* C19 transfer response.  It must
communicate `chi dagger Psi` and `chi cross Psi` with a sign supplied by native
bond current, while vanishing at rest and leaving the longitudinal common
translation tangent unchanged.  Do not add another unsigned square: this
family has now been tested.  Rank-test the first variation against the two
measured live-frame error directions and the correct forward-current tangent
before another nonlinear run.  Require the unit-circle Floquet result, then
test exact site occupancies at `+/-1` and `+/-2`.

## Measurement clarification

A center of density can change by less than one cell even on a discrete
lattice because amplitudes on integer sites change continuously.  It is an
expectation/readout, not an intravoxel coordinate.  The active completion gate
nevertheless remains the user's stronger requirement: the same carrier must
be recovered at an integer-translated site and repeat the handoff.
