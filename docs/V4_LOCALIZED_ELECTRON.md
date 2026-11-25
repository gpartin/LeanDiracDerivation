# V4 localized-electron gate ledger

The executable release process and corrected same-action stage contracts are
defined in [V4_COHERENT_RELEASE_PLAN.md](V4_COHERENT_RELEASE_PLAN.md).  That
plan is controlling for V4 promotion.  This longer ledger retains the research
history and must not be read as permission to combine passes obtained under
different action hashes.

V4 keeps the V40 two-field C19 action and every V2/V3 theorem unchanged. Its
target is stronger: one freely evolving localized state must realize the rest
mass, four-component readout, spin, translation, density, and current already
derived for the free linear sector.

| Gate | Required result | Status |
|---|---|---|
| V4-001 | Exact unrestricted localized rest orbit on all periodic boxes `L >= 3` | **PASS; CLEAN-BRANCH REBUILD RECORDED IN `v4_001_revalidation.json`** |
| V4-002 | Nonlinear rest stability of that same orbit | **STOPPED AFTER 5 CORRECTED RCAS: COMPLETE-FRAME CLOCK REPAIR GREATLY IMPROVES BUT DOES NOT CLOSE LINEAR STABILITY** |
| V4-003 | Lossless localized four-component readout, including both committed layers | **NOT STARTED; BLOCKED BY V4-002** |
| V4-004 | Operational spin-one-half action on that localized carrier | **NOT STARTED; BLOCKED BY V4-002** |
| V4-005 | Integer cell-shift covariance and signed native current for the same readout | **NOT STARTED; BLOCKED BY V4-002; HISTORICAL CONTROLS RETAINED** |
| V4-006 | Exact free one-cell relative-periodic handoff of both committed layers | **NOT STARTED; BLOCKED BY V4-002; EXACT ENDPOINT CONTROL RETAINED** |
| V4-007 | Repeat the identical handoff into the next cell in both directions | **NOT STARTED; BLOCKED BY V4-002** |
| V4-008 | Coarse-grained agreement of that hopping state with Dirac mass, transport, density, and current | **NOT STARTED; BLOCKED BY V4-002; HISTORICAL TANGENT EVIDENCE IS NOT A PASS** |
| V4-009 | Robustness, all three axes, diagonal control, and both cubical copies | **NOT STARTED; BLOCKED BY V4-002** |
| V4-010 | Cumulative localized Dirac-electron theorem and clean release audit | **NOT STARTED; BLOCKED BY V4-002** |

### 2026-11-25 coherent-chain stop

The clean `feature/v4-coherent-chain` rebuild passed V4-001. An initial
V4-002 screen accidentally used the older movement incumbent and is retained
only as superseded provenance. The corrected screen uses the actual
complete-frame clock action: six-face frame transport at `6/19`, the matched
parallel material coordinate `chi^dagger Psi`, its native-clock potential,
and the derived spacetime-face response.

That correction is material. It lowers the full `Psi`--`chi`,
two-committed-layer growth rate from `0.7821541600` to `0.0610029580` while
preserving the exact rest solution. The largest multiplier nevertheless
remains `1.0002383212`, and approximately 97.74% of its leading-mode power is
in `Psi`. Spacetime calibration, matched-frame strength, clock-potential
strength, both symplectic signs, triangle completion, and transverse material
completion do not place the spectrum on the unit circle. The negative
symplectic response is the best repair at growth `0.0553607134`, but it still
fails. The required nonlinear 100-cycle gate was not run because its
necessary linear-stability gate failed.

This triggers the declared five-RCA stop rule. V4-003 through V4-010 are
dependency-blocked, not failed on their own merits. The scoped RCA and the
next research route are recorded in
`experiments/v4_localized_electron/V4_002_COMPLETE_FRAME_RCA_STOP.md`. No V4
release is promoted and `main` remains unchanged.

### 2026-11-25 gate correction

The stage contract now distinguishes a discrete one-cell handoff from a smooth
sub-cell slide. A finite relative-periodic orbit may exist even when the rest
orbit has no neutral infinitesimal translation mode. Consequently, absence of
that neutral mode and a nonzero Peierls--Nabarro barrier are diagnostics of
smooth sliding or pinning; they are **not** failures of the authorized minimum
one-cell movement hypothesis.

The V4-006 endpoint remains the literal same-assignment joint `Psi`--`chi`
translation locked in `EXACT_TRANSLATION_MAP_LOCK.md`. A passing orbit must
satisfy, for both committed layers,
`X(n+N) = exp(i phi) tau_cell X(n)` under unrestricted native evolution. It
does not have to occupy a meaningful fractional center between endpoints.

The earlier stability work is retained rather than silently discarded. The
reduced tight-frame research action has a saved unrestricted 100-cycle
zero-`chi`-charge rest result with maximum aligned shape residual
`5.64898e-14`, carrier fidelity at least `0.9999999999999997`, zero measured
`Psi` exterior fraction, and interval-energy drift `2.08238e-15`. That is a
real nonlinear **rest** stability pass for that declared action and branch. It
is not a charged-branch pass and it does not transfer automatically to the
later transport-action deltas. The common-velocity/wake candidate separately
places its linearized multipliers on the unit circle to about `2e-15`; that is
a tangent/Floquet result, not a nonlinear 100-cycle moving-carrier pass.

## V4-001 result

`LeanDirac.V4.exact_localized_rest_seed` proves, from the native reciprocal
action, an exact two-layer recurrence for the eight-site `Psi` carrier and its
live `chi` halo on every periodic C19 box of side at least three. It also proves
zero local imbalance and nontrivial phase advance in both fields. The theorem
uses no holding force, background reset, projection, or Dirac matrix.

The result is symbolic in `B` and holds for `B > 1` under the stated timestep
condition. It therefore does not select `B = 1197`.

## V4-002 full tangent result

The complete mass-normalized tangent and exact co-rotating two-layer Floquet
map now include every `Psi`/`chi` component, all carrier/halo/exterior sites,
and both committed layers. The unchanged action has a finite continuum-time
growth rate approaching approximately `1.037`. It is unchanged between
side-three and side-four boxes, while the one-step multiplier approaches one
at the expected rate as `dt` shrinks. This is a physical tangent instability,
not a timestep or boundary artifact.

The leading canonical mode is entirely on the eight carrier sites and lies in
their independent radial/phase span. It has negligible translation overlap.
V4-002 therefore fails for the canonical action, and V4-003 through V4-007
cannot be promoted as same-carrier electron results.

A generic bounded bond polynomial,
`F(s) = s - 0.8 s^2 / chi0^4 + 0.03 s^3 / chi0^8`, combined with research
`B = 229`, reduces the matched side-three rate to about `0.0755` and the
side-four rate to about `0.0422`. Both remain positive and converge under
timestep refinement. This is a diagnostic research candidate only: its
coefficients are not derived, it is box-sensitive, and it does not pass the
stability gate.

Two independent multi-site spatial repairs were then derived from real,
reciprocal, cubic-symmetric actions and tested against the complete tangent:

- positive two-hop C19 curvature of `W = chi wedge Psi`;
- positive elementary cube-plaquette curvature linking the two bond turns
  that share a corner.

Neither works. The first preserves the exact orbit but monotonically worsens
its growth. The second also worsens growth and deforms the exact orbit. An
inadmissible negative-sign probe produces only a 0.36% improvement before
worsening, so a bounded spontaneous-curvature completion is not supported by
this response. This closes the simple spatial-stiffness route. The next action
basis must alter relative `Psi`--`chi` timing/orientation response (for example
through an independently derived phase- or velocity-resolved reciprocal
coupling), not add another scalar bend penalty.

The first phase/velocity test adds the positive local kinetic square
`|chi x PsiDot - chiDot x Psi|^2`. Its canonical momenta and implicit,
self-adjoint discrete Euler--Lagrange update were derived rather than reusing
ordinary Verlet. The zero-coupling implementation reproduces the old analytic
growth rate to `9.43e-8`, and a small positive coupling admits a new exact rest
orbit. It nevertheless increases the leading growth by a factor of `4.08`;
negative diagnostic coefficients also worsen it and lose global kinetic
positivity. Symmetric extra relative inertia is therefore rejected. The next
admissible basis must provide an antisymmetric/gyroscopic phase response.

Five gyroscopic response cycles have now been completed. The local relative,
C19 bond-relative, common-orientation, and field Berry one-forms progressively
reduce the best full-spectrum growth from `0.0755557` to `0.0704348` (`6.78%`)
while preserving an exact freely re-solved carrier. The elementary-plaquette
one-form was tested with independent corner phases and retired because both
signs worsened growth. The surviving result is not admitted: all coefficients
are continuation diagnostics without native provenance, and positive growth
remains. It does demonstrate that antisymmetric phase-space steering affects
the correct missing channels, whereas scalar stiffness and symmetric inertia
do not.

## Fixed-center tangent classification

The exact symbolic controls have also been rerun. The isolated localized orbit
does not commute with unit translation (the canonical rational witness is
`43309086609 / 17378239864`), so its tangent is a defect operator rather than a
Bloch family. The exactly closed compact carrier block has momentum-derivative
ranks `(0,0,0)`. Enlarging to the natural four-dimensional eigenvalue-27 corner
space does not fix this: its axis response ranks are `(2,2,2)`, it has a common
transport-null direction, its cross anticommutators are nonzero, and its first
response requires exterior channels.

This retires only a fixed-center compact Bloch interpretation. It does not
exclude a noncompact joint field/scattering bundle or a family of translated
relative-periodic orbits. Those are the remaining admissible routes.

## Current movement RCA

**Issue:** No freely moving branch of the exact compact carrier has been
proved.

**Why it matters:** A localized Dirac electron requires the same state to
carry identity and charge across cube boundaries while preserving its internal
four-component structure.

**Symptoms:** Existing native trials move only a small fraction of a site and
complete no verified carrier handoff. Directly copying the rest seed into the
next cell creates conflicting shared halos rather than a translated solution.

**Expected symptoms:** A passing branch has signed bond current, preserves the
carrier under a common internal-frame alignment, and completes repeated
translations in both directions while matching the independently evolved
discrete Dirac comparator.

**Root cause:** The compact exact rest preparation has no proved narrow signed
momentum direction. Its exterior `chi` geometry cancels the coupling required
for the fixed-center orbit, while adjacent copied preparations overlap through
their halos. This identifies a preparation/operator obstruction; it is not a
universal no-go for the action.

**Corrective action:** First derive the full localized tangent and Floquet
operator and test whether it contains a translation mode. If it does, continue
that mode into a jointly solved multi-cell `Psi`--`chi` relative-periodic
orbit. Change the action only if an exact response-rank calculation proves the
required native translation channel absent.

## Literal edge-snap update

The movement comparator has been corrected. A minimum translation does not
hold the four shared-face values fixed. Every one of the eight carrier sites
moves to its same-assignment C19 face neighbor, so the shared face is replaced
like a conveyor. This endpoint map is exact in both signs of all three axes;
the former reflected assignment instead moved four sites zero spacings and
four sites two spacings and is retained only as a superseded PN control.

The canonical action does not execute the exact shift. The static carrier has
essentially zero native chi acceleration, while the translated two-layer
recurrence requires a large nonzero chi second difference. A response-rank
audit at the locked `B=1197` separates the missing action class:

- all 30 verified velocity-linear gyroscopic one-forms have the wrong
  direction-reversal parity and explain only `1.40%` of the squared target;
- eleven symmetric kinetic channels have the correct parity and span
  `95.85%` without positivity restrictions;
- arbitrary nonnegative additions of those squares point the wrong way;
- the generic C19 factorization
  `(I-D19/24)^2=(A19/24)^2` reduces the fixed-phase residual to `0.09891`.

The last result is a response oracle, not an admitted action. Its exact form is
semidefinite and deletes the inertia of the `D19=24` modes. Adding native
`kappa=1/63` makes it invertible but leaves `10.2%`--`11.0%` residual. The next
research gate is a positive, invertible, field-dependent C19 kinetic square
whose connection force closes that remainder, followed by a freely re-solved
translation phase and period. V4-002 and movement remain **FAIL**.

That field-dependent screen now has a partial survivor. The material
substrate velocity

`Vchi = chiDot + (1+54/19^2) rhoPsiDot chi / [2(rhoChi+kappa 19^2)]`

inside the positive C19-average kinetic square reduces the literal handoff
residual by `18.53%` without changing the exact rest orbit or worsening its
full implicit two-layer spectrum. The coefficient comes from the carrier's
exact native radial balance and is not fitted to a Dirac velocity.

The physical endpoint remains a one-edge translation, but current evidence
rejects identifying it with one numerical integration update. Relative-
periodic searches over two and four updates improve substantially but do not
close while also conserving the two native phase charges. The saved
four-layer RCA assigns `96.09%` of the remaining defect to `chi`, principally
its third component at the leading shell. This is the next action-response
target. The candidate remains research-only; no localized movement, spin,
or same-carrier Dirac gate has passed.

## Conditional edge-channel correction

The literal whole-carrier shift remains a useful massless/chiral and rigid
soliton control, but it is not the massive Dirac movement target. The derived
cubical algebra gives, on every axis, the projectors
`P+ = (I+gamma)/2` and `P- = (I-gamma)/2`. A nearest-neighbor Dirac step sends
these two channels toward opposite edges and mixes them locally through the
mass grading `beta`. Their unequal weights create the observed net current.

This distinction was checked directly from the Lean cubical definitions. Each
of the two rank-four flavor copies contains two forward and two backward edge
channels. The flavor involution commutes with every `gamma` and with `beta`, so
transport preserves both copies separately; one copy is not the leading edge
of the other. For a representative `m=1`, `p=0.6` positive-energy state, the
two edge-channel weights are exactly `(1+p/E)/2 = 0.7572478777` and
`(1-p/E)/2 = 0.2427521223`. The conditional-edge/mass step converges to the
Dirac generator under refinement, whereas a rigid common shift retains an
order-one generator mismatch. A common rigid shift is recovered only in the
massless one-chirality control.

This also retires two false repair paths. Halving the timestep in the best
density-tensor orbit makes the properly normalized physical residual nearly
twice as large, so the finite mismatch is not temporal under-resolution. A
generic positive material derivative built from the native `Psi` bond current
does not improve the six-direction handoff response, so ordinary scalar
advection of `chi` is not the missing mechanism.

V4-005 must now use a frozen localized cubical readout and test whether the
unrestricted native update produces the signed `+/-gamma` bond currents and
local mass mixing. Those projectors remain measurements only. No conditional
shift, copy selector, or Dirac matrix may be inserted in the action. V4-005,
V4-006, and V4-007 therefore remain **OPEN**.

The conditional shift and the rigid shift have the same centered second
difference at zero mass phase because `P+ + P- = I` and `gamma^2 = I`.
Consequently the rigid recurrence remains a necessary Klein--Gordon-square
action control, but it is not sufficient evidence for signed Dirac current.
The complete variational response audit confirms that this correction does
not change the action-class diagnosis: gyroscopic terms have the wrong parity,
while a nontrivial symmetric `chi` kinetic metric is required.

The current partial survivor is the nonlinear positive metric

`Tchi=(B/2)[kappa|chiDot|^2 +`
`|A19(Pchi Vchi+s(a)(I-Pchi)Vchi)/24|^2]`,

where `Pchi=chi chi^dagger/(rhoChi+kappa chi0^2)`,
`a=rhoPsiDot/[2 chi0(rhoPsi+kappa chi0^2)]`, and
`s(a)=(1+g a^2)^(-1/2)`. It is real, reciprocal, phase covariant and cubic
covariant, and contains no Dirac matrix, selected copy, corner label, or
preferred direction. Since `a=0` on the exact rest orbit and the first metric
change is quadratic in `a`, its complete rest quadratic action is unchanged.

The six-face normalization `g=6` improves the frozen handoff response by
`41.42%`; the C19 normalization `g=19` improves it by `38.45%` and currently
gives the better freely optimized four-layer residual, `0.00239137`, with both
native phase-charge defects near `1e-5` or below. The orbit does not close.
The remaining residual is `87.95% chi` and is still dominated by component
three at a carrier corner. No coefficient or action is admitted, V4-002 still
fails spectral stability, and V4-005 through V4-007 remain **OPEN**.

## Direct `chi` edge-response trial

The native edge-snap hypothesis now has one concrete action-level success.
The current action moves only the composite area `chi wedge Psi` between C19
neighbors. Testing ordinary positive field bonds directly on the saved
four-update residual shows that `Psi` stiffness has the wrong sign, while
both face and edge `chi` stiffness have the required sign. Their positive
four-column fit explains `13.52%` of the squared defect.

The first non-fitted candidate uses the C19 low-momentum normalization:

`Delta U=(B/6) sum_(i,n in C19) w_n |chi_(i+n)-chi_i|^2/(2*24)`.

It contains no prepared ray, corner label, flavor copy, preferred axis, shift
operator, or Dirac matrix. With the existing g19 nonlinear kinetic metric it
reduces the freely optimized four-update equation residual from `0.00239137`
to `0.00223564`, a `6.51%` improvement. This is evidence that the action is
missing an independent neighboring-`chi` response, not proof that the carrier
snaps exactly.

The candidate remains **UNADMITTED**. The unchanged rest carrier is not an
exact orbit after adding this term, and the charge defects worsen. The next
gate is a free continuation of the complete localized rest carrier and halo
under this action, followed by the complete two-layer spectrum. Only then may
the movement orbit be re-solved using its own exact endpoints. V4-002 and all
movement gates remain **FAIL/OPEN**.

The static-bond continuation is now retired as the minimal repair. Its rest
forcing has a `44.72%` Fredholm obstruction, dominated by the same third
`chi` component seen in the moving residual. Retuning the ordinary local and
bond potentials does not cancel that obstruction.

The replacement partial incumbent is a positive transfer-activated `chi`
bond. Its activation scalar is the native local `Psi` density rate, so it is
zero on the exact rest orbit and nonzero only during a handoff. The side-three
rest orbit remains exact and its complete two-layer growth is not worsened.
Its first free four-layer solve lowers the movement residual from `0.00239137`
to `0.00232952` (`2.59%`). This is a reproducible improvement, not closure:
the remaining residual is `89.86% chi`, and both native charge defects remain
nonzero. V4-002 and V4-005 through V4-007 remain **FAIL/OPEN**.

## Localized conditional-shift lift audit

A positive transfer-driven neighbor connection has now been tested. The
native coefficient `1/(2 chi0)=1/38` acts as a `chi`-frame reflection: it
pushes the radial substrate curvature forward while turning the transverse
curvature oppositely. It lowers the freely optimized handoff residual to
`0.00231328` and does not worsen the full rest spectrum. This is retained as a
partial action response only; the rest spectrum still grows and the orbit
does not close.

The more consequential result is a measurement-boundary correction. The
Lean cubical Clifford coordinates are an internal eight-coordinate fiber,
not yet a proved set of eight physical lattice sites. The current physical-
corner lift of the conditional shift disagrees with a literal translated
carrier by `84%`--`100%` in `Psi`. Both four-component flavor copies sum back
to the same internal shift and do not remove this mismatch. Each copy already
has two forward and two backward channels, hence zero one-axis walk index;
the copies are not a leader/follower pair.

V4-005 is therefore blocked upstream on a concrete mathematical bridge, not
on another stiffness coefficient: derive one localized embedding from full
`Psi`, live `chi`, both time layers, and the surrounding shell, then prove
that it is closed and intertwines translation consistently across overlapping
cubes. Until this is done, the rigid and conditional endpoint searches are
diagnostics rather than physical electron movement tests. No canonical action
or public theorem has changed.

## Overlapping-cube phase-space lift

The upstream bridge now has a surviving construction. Each cell's internal
cubical amplitudes are synthesized onto its physical vertices using the
localized carrier's native `Psi` ray and `chi` companion. Contributions from
overlapping cells add. This is a readout map only; it does not modify the
action or install a conditional shift.

For either four-component flavor copy, `Psi` alone and `chi` alone each have
complex rank two. The joint `Psi`--`chi` map has complex rank four at all
`4923` sampled lattice momenta, with minimum sampled singular value
`0.0825054`. Its physical-translation intertwiner defect is `6.73e-15`.
Thus a four-component localized readout is supported only as a joint native
two-field object in this candidate.

The two flavor copies map to the same spatial rank-four image to numerical
precision; their image-projector difference is below `7.69e-15`, and their
combined image still has rank four. This is evidence against treating them as
independent leader/trailer particles in the spatial lift, but time-layer and
nonlinear equivalence remain unproved.

The complete implicit two-layer update supplies the missing first-order
information. After the previous-layer graph is solved from the native action,
the full invariance defect is `1.3440e-4`, `3.3637e-5`, and `8.4114e-6` at
`dt=1/128`, `1/256`, and `1/512`. The observed orders `1.99847` and `1.99962`
support continuum tangent closure with a second-order discrete remainder.
This supersedes the earlier statement that V4-005 is blocked on the existence
of a localized embedding; the spatial and two-layer candidate now survives.

V4-005 is still **OPEN**, because the same map has not yet supplied all three
native transport generators, the pulled-back density/current, or nonlinear
repeated translation. V4-002 rest stability also remains **FAIL**. The next
gate is term-by-term compression of the native update and Noether current
through this frozen embedding, not another action-coefficient sweep.

## Exact carrier-contrast motion channel

The physical center-motion audit identified one omitted native component.
Outside the carrier, ordinary `chi` points along its third component; on the
eight occupied sites the carrier replaces it with the in-plane companion.
Consequently, translation must carry the entire
`(localized carrier)-(chi0 e3 vacuum)` profile. A companion-only lift cannot
do this regardless of how its amplitudes are tuned.

The exact contrast profile changes the copy verdict. Either four-component
copy alone leaves a worst translation-tangent defect of `0.666635`. Keeping
both copies closes all three axes below `1.40e-15`, using only the existing
complex `Psi`, complex `chi`, and nearest cube centers. This is a kinematic
span result, not a programmed shift or a new field.

The native two-layer graph selected from the rest orbit has residuals
`1.4073e-2`, `3.6158e-3`, and `9.0754e-4` at `128`, `256`, and `512` updates
per interval. The observed orders `1.9605` and `1.9943` support the expected
second-order continuum closure. The current next gate is the action-derived
phase density and C19 bond current on this same two-copy contrast, followed by
large-box and nonlinear handoff tests. The action remains unchanged and
V4-002/V4-005 remain **FAIL/OPEN** until those gates pass.

## Fixed-charge transport obstruction

The current gate distinguishes a translated outline from physical carrier
motion. An unrestricted first-shell previous layer can reproduce a prescribed
one-edge profile displacement with a linearized native residual of
`8.04e-16`, but the action-derived phase charge does not move with it.

Odd neighboring-center phase ramps provide two robust signed current
directions. The third fixed-charge singular value is numerical zero. This is
not caused by omitting one copy, the normal component of complex `chi`, the
remaining native `C^3` frame, or the common conserved charge: an exhaustive
`96`-channel audit remains rank two. The copies reverse some signed responses
but are not a leader/trailer pair.

The exact rest geometry explains the loss. The carrier is globally coplanar
in field space, so every `W=chi wedge Psi` points along one common normal.
Neighboring `W` overlaps are real across x/y faces and imaginary across z
faces. Consequently, the present bond action has no linear phase curvature
for the third edge direction.

Position-only parallel-overlap and square-face action completions, local and
bond overlap symplectic terms, and a positive C19 area-gradient kinetic term
have all failed the same rank test. The next gate is to derive a freely solved
non-coplanar carrier/rest branch or a reciprocal native frame connection that
restores three bond-response curvatures without encoding a spinor shift.
V4-002 remains **FAIL**, V4-005 remains **OPEN**, and no research action is
admitted.

## C19 Bargmann-loop connection gate

A C19 face--edge Bargmann loop was tested as the smallest native mechanism
that can turn receiving `chi` before local `Psi` arrives.  Its positive phase
potential does provide that quiet-receiver force, but torques the exact rest
orbit and remains rank two.  A reversal-invariant joint kinetic connection
preserves rest to numerical precision and raises the fixed-charge response to
rank three, but `99.45%`--`99.97%` of its exact tangent image lies outside the
tested eight-component two-copy tangent and it has nonzero cross-copy blocks.

This is not same-carrier Dirac transport.  Projection would be a forbidden
bolt-on.  V4-005 through V4-007 remain open, and the production Lean facade
still contains only the exact fixed-center V4-001 theorem.  Full evidence and
the six-field RCA are in
`experiments/v4_localized_electron/C19_BARGMANN_CONNECTION_REPORT.md`.

## Tetrahedral corner geometry does not by itself close transport

The signed eight-ray preparation has an exact basis-free orientation map:
`P(z)=z z^dagger`. Its four distinct projectors form a three-dimensional
tetrahedral tight frame. All face and edge pairs admitted by C19 have the same
orientation product `-1/3`; the excluded body-diagonal pairs have product
`+1`. This explains why the corner algebra can encode three isotropic axes
even though the current `chi wedge Psi` lift transports charge in only two.

A positive native kinetic action was screened using the joint projector
`Q=(Psi Psi^dagger+chi chi^dagger)/chi0^2` and its C19 commutator connection.
It is real, cubic, reciprocal under full variation, contains no preferred
axis or spinor operator, and is exactly inactive on the rest orbit. Despite
those properties, its fixed-charge transport response remains rank two at
strengths `1`, `19`, and `361`. The third singular response is numerical
noise, so coefficient tuning cannot repair this basis.

The tetrahedral map is therefore retained as a geometric diagnostic only. It
must not be installed as an extra field or conditional shift. The next gate is
an unrestricted non-coplanar rest solve; failing that, the next action class is
an oriented C19 common-neighbor/triangle connection with a derived positive
Hamiltonian and full reciprocal `Psi`/`chi` equations.

## C27 orientation ablation and native gyro completion

Separate field stencils do not repair the motion gate. A body-diagonal `chi`
term remains rank two, while a body-diagonal `Psi` term is incomplete and
destroys the old rest orbit. The joint orientation field
`W=chi wedge Psi` is different: adding its eight body-diagonal Dirichlet bonds
raises the fixed-charge transport response to rank three in all axes. This is
the first tested action response with that property, but its positive
potential alone has rest DEL RMS `0.0196089` and is therefore rejected.

A full local/body, common/relative, velocity-linear orientation basis gives a
reciprocal rest completion. Its force matrix has rank two and the exact
rest-preserving native normalization reduces to

```text
a = 2 dt / sin(theta)  ->  2/chi0,
c = (a+B/4, a-B/4, -a/2, 0).
```

These are respectively the local common-area, local relative-area,
body-common-area, and body-relative-area one-form coefficients. The rule uses
the native rest clock, `B`, and the body/local incidence ratio; it does not use
a Dirac velocity or stability fit. The resulting side-five response has rest
DEL RMS `1.897e-12`, ranks `[3,3,3]`, and weakest relative singular values
`{0.04539,0.04521,0.09051}`.

The full side-three spectrum is unresolved only at the degenerate unit
multiplier: no modulus exceeds `1+1e-7`, the apparent excess stays about
`5e-8` under four timestep refinements, and its eigenvalue condition is
`5.68e7`. The exact generalized-unit quotient leaves a bounded residual
through 1000 cycles, but native observable transients still reach about `55`
under unnormalized velocity controls. At the justified spectral tolerance the
raw remainder is `90%`--`99%` exterior `chi` orientation. A direct positive
`chi` C19 spring was tested and rejected because `87.3%` of its rest torque is
outside the reciprocal W-gyro span. The candidate is therefore **not
admitted**. The next step is the exact charge/frequency family, physically
normalized perturbation gate, and unrestricted nonlinear implicit-action
evolution. C27 remains an ablation; canonical C19 and all public claims remain
unchanged.

The subsequent cube-incidence symplectic calculation supplies the previously
missing linear response. The exact rest-force nullspace derives
`A_area+,C19 - 4 A_tensor+,local + 4 A_tensor-,local - 4 A_area+,local`.
At unit overall coefficient its uncompressed 3000-dimensional phase-space
spectrum has positive kinetic mass and zero multipliers above `1+1e-7`; the
baseline had 31. The same declared result survives the fixed tangent pencil
at `dt`, `dt/2`, and `dt/4`. No coefficient is fitted and no axis, corner,
copy or Dirac matrix is present. This is a linear research-candidate pass,
not V4-006: the required implicit update, nonlinear stability, motion, and
native/Dirac current agreement remain open. Canonical V40 is unchanged.

## Movement-target correction

The literal whole-carrier snap and the massive Dirac transport target have
now been separated. A massive cubical Dirac step sends the `+gamma` and
`-gamma` internal channels in opposite edge directions inside each
four-component copy, then mixes them locally through the mass grading. It is
not a scalar translation of the complete eight-site carrier. The independent
conditional-walk comparator converges to the cubical Dirac step, while the
rigid scalar shift retains an order-one generator mismatch.

The reduced two-tight-frame research action does not yet execute this
conditional motion: its optimized recurrence residual remains approximately
one in all six signed axes. The surviving overlapping-cube construction gives
a full-rank, translation-equivariant spatial lift into joint Psi/chi fields,
but the same lift is not yet invariant. Its current-action cross block falls
from about `71%` at one center to `8.66%`/`8.43%` for the two copies when the
full nearest 27-center envelope is retained. This identifies missing halo as
most, but not all, of the prior mismatch. The next V4-006 calculation is the
two-layer invariant-graph/current completion of the remaining residual. No
action was changed or promoted by this target correction.

The all-center invariant-graph calculation sharpens that result. The final
spatial leakage is a rank-32 out-of-plane Psi response. A same-rank native
dressing of the original four-component envelope closes the configuration
acceleration tangent below `7e-15`, with a well-conditioned map back to the
undressed carrier. The actual two-layer update still leaves a `0.2164%`,
rank-32 phase response. Its enlarged hull closes near `2e-11`, but the phase
dressing has not yet been reduced to the same carrier coordinates. V4-006 is
therefore still open at the time-layer/current bridge, not at the spatial
configuration bridge.

The complete two-layer spectrum changes that last diagnosis. Once the
rank-32 phase response is retained, the update hull closes below `2e-13`, but
it has 184 growing multipliers whose physical growth rate converges to
`1.036012` as the timestep is refined. The fastest mode overlaps the target
carrier by `0.999996`, lies in the in-plane Psi amplitude/phase channel, and
is neutral under both native U(1) charges. Bare V40 supplies generalized
stiffness `-1.080742` in that direction; the two tight-frame terms are
numerically blind to it. Consequently the exact fixed-center orbit is not
perturbatively stable. Before the current can be pulled back as a physical
Dirac transport claim, an action-derived in-plane stabilization and the odd
transport response must be closed together. Canonical V40 is unchanged.

The first joint channel satisfying both requirements in part has now been
derived. On an unoriented C19 bond, the Psi-density difference supplies a
state-dependent bond orientation, while the full pair
`Z=chi tensor Psi` supplies the joint phase and component orientation omitted
by `chi wedge Psi`. Multiplying that density contrast by the antisymmetric
pair one-form gives a reciprocal, cubic symplectic connection with no chosen
axis, corner label, copy, track, or Dirac matrix. Its discrete
Euler--Lagrange residual on the exact rest orbit is below `4e-17`, and its
linear response is nonzero in all three translation axes.

At the native coefficient scale `B*chi0=22743`, the common-pair connection
reduces the maximum growth rate from `1.0360122` to `0.0710213`, a `93.14%`
reduction; the neighboring declared scales are worse. This identifies a
natural transport mechanism but does not close it: 184 growing multipliers
remain, the relative-pair channel cannot simply be added with the same sign,
and nonlinear two-cell motion/current has not passed. The term is not
admitted or canonical. V4-006 remains open pending classification and
action-derived closure of the residual spectrum.

That residual is now classified. The first limiting mode is exactly the
eight-corner `001` Psi phase character, alternating across one pair of cube
faces. A fine continuation of the connection reaches a nonzero plateau and
then switches to a distinct mode: `93.37%` chi and `93.15%` the third C3
component on the carrier/interface. This occurs because the wedge area is
zero in quiet `Psi=0` substrate and cannot communicate chi orientation there.

The six native scalar orientations of the same reciprocal pair connection
span five response directions, although their sufficiency for zero growth is
still open. Reusing the positive unit chi-interface bond
terms reduces the growing-mode count from 184 to 30; a derived positive area
phase square gives the best complete screen at growth `0.0631092` with 29
modes. The two required mechanisms—joint Psi--chi handoff and chi interface
orientation—are therefore identified, but their natural joint degeneracy is
not. No action is admitted. The next gate is the constrained Hamiltonian
bifurcation solve on those 29 modes, followed by full-spectrum refinement.

The declared position-only bifurcation screen did not close. The six scalar
pair connections have rank five but retain all `29`
growing modes; their best continuous inverse fit changes growth only from
`0.0631092` to `0.0625757`. The full spectrum confirms the reduced result.
The residual is a carrier-localized common Psi--chi `001` cube phase with
exact C19 eigenvalue `8`, and the scalar connections have zero stiffness on
it. A seventh positive cube-phase square reduces the best growth to
`0.0601106` and the count to `27`; this continuous seven-response optimizer
did not raise its convergence flag, so it is only diagnostic. The geometric
factor `12/8` gives
`0.0608864` and the same count. Thus no coefficient is admitted. The next
valid action calculation is a complete cube-incidence symplectic basis, not
another scalar stiffness retune. V4-006 remains open and canonical V40 is
unchanged.

The subsequent cube-incidence symplectic completion closes that linear rest
instability and passes an exact implicit rest update plus a preliminary
one-cycle nonlinear perturbation run. It does not yet pass movement.
Correcting the transport cut showed that the apparent launch current was
internal carrier circulation. Across the actual quiet boundary, all tested
global phase launches leave receiving-site `Psi` exactly zero for every
completed update. The native activity product switches the stabilizing
connection off at the quiet endpoint, while the canonical wedge geometry
makes both receiver-field forces vanish there. A symmetric mean-activated
replacement torques the rest orbit and has no rest-preserving combination in
the tested basis, so it was rejected. The remaining mathematical test is the
component-selective four-spinor launch with a freely solved exterior tail;
common-phase compact support is not the massive Dirac transport target.
V4-006 remains open.

The proper overlapping-cube four-component tail has now been tested under
the stable action. Its complete all-center configuration cross-block is
`0.925131%`. Exact configuration closure is possible only through an almost
singular pullback, and the two-layer update still leaves a `0.551294%`
residual. A better-conditioned graph lowers the two-layer mismatch to
`0.0163095%` but does not close its configuration equation. This identifies
the remaining object as a phase-space readout involving both committed
layers and the action-generated out-of-plane `Psi` response. Static halo
retuning is retired; V4-006 remains open pending that phase-space graph,
compressed transport generators and current.

The direct same-image two-layer graph has now been falsified. The bounded
previous-layer map has only `16` resolved control directions for the
`1000`-real-dimensional all-center carrier image. It leaves `99.9967%` of
the original normal leakage and fails graph self-reproduction by `99.2110%`.
Relaxing the singular cutoff can make the one-step residual look smaller,
but only with prior-layer operator norms between `4.59e7` and `2.22e9` and
reproduction defects between `1.00e10` and `1.85e13`; those are numerical
amplification, not native motion.

The action ablation identifies the responsible limitation. The bare and
cube-incidence connection blocks provide no resolved previous-layer control
of this leakage. The density-oriented reciprocal connection supplies all
`16` resolved directions, but they are almost orthogonal to the required
normal response. Thus preparation on the unchanged spatial image cannot
close V4-006. A broader phase-space embedding with a genuine spatial tail
remains mathematically possible; if it is not well-conditioned, the research
action needs a new rest-preserving reciprocal kinetic/gyroscopic connection
whose native image overlaps the missing normal channel. No action was
changed or promoted by this diagnosis.

## Corrected extended-packet motion gate

The compact-hop obstruction is not the continuum Dirac motion verdict. The
eight-corner algebra is now treated as the internal fiber of a self-contained
joint `Psi`--`chi` envelope whose overlapping cube contributions extend ahead
of and behind its center. No receiving cube is preloaded and no external
background is maintained; the nonzero tail belongs to the state being tested.

In the carrier co-rotating frame, the one-step native mismatch against the
independent C19 Dirac comparator is `0.362501%` at `dt`, `0.0968164%` at
`dt/2`, and `0.0246562%` at `dt/4`. The observed orders are `1.90466` and
`1.97330`. Leakage outside the native four-component spatial lift falls from
`6.75e-5` to `1.69e-5` to `4.23e-6` over the same refinement. Both copies,
both signs of all three axes, and a diagonal control are included.

This survives the corrected continuum tangent gate and overturns the use of
strict compact-support handoff as the primary motion test. V4-006 is still
open: the next gates are box/width refinement, native Noether-current versus
Dirac-current equality, and unrestricted nonlinear propagation of the same
extended packet. No action was changed or promoted.

## Extended-packet completion and red-team result

The motion interpretation survives three further checks. At `dt/4`, four
packet widths from `0.95` through `1.6` and carrier wave numbers `0.2` and
`0.35` all remain below `3.38e-4` state error. At `dt/8`, the shared
positive-frequency readout has maximum density error `5.46e-5` and maximum
C19 Dirac-current error `0.003089`, including both copies, both signs, all
three axes and a diagonal. An unrestricted nonlinear four-update run at
`1e-4` of the rest-state norm converges in every tested copy/sign case; its
worst native-state error against the independent Dirac lift is `0.02544` and
its worst leakage from the packet lift is `0.0006643`.

The raw nonlinear global-phase charge is **not** silently relabeled as the
Dirac probability. Its provisional contraction has a `0.6067%` conservation
defect and a `1.412%` center-shift mismatch. The applicable quantity is the
positive-frequency phase-space charge proved in V2; a full nonlinear local
Noether-current identity for this localized orbit remains open.

The red team therefore does **not** close V4-006. The corrected movement
mechanism is strongly supported at tangent order and survives a short
nonlinear run, but the research action is still hard-coded to side five, so
box refinement has not been performed. Nor has the packet completed two full
cell translations, 100 moving cycles, or the moving-carrier Lean proof.
Canonical V40 remains unchanged and the research action is not promoted.

The subsequent arbitrary-box and endurance audit changes the diagnosis. The
research action has now been ported to generic periodic box size with exact
side-five identity, and its packet residual converges on sides five, seven,
and nine. A 100-cycle sparse tangent run nevertheless leaves the intended
carrier sector; ten-cycle `dt/8` and `dt/16` controls agree, ruling out a
timestep artifact. The one-cycle defect is almost entirely in-plane `Psi` on
the carrier sites. Neither native first-step history nor coefficient changes
within the two existing connection terms close it. V4-006 therefore remains
open at an action-sector-closure obstruction, not at a box-size or launch
preparation problem. No action or canonical framework has been changed.

A response-rank audit of every directly reusable reciprocal invariant in the
research snapshot also fails to span the missing channel: seven terms provide
rank six and leave `0.28103` relative one-cycle error under an inadmissible
best fit. The next work is therefore a new action derivation, not another
retune or endurance run.

The simple moving-substrate hypothesis has also been falsified. Coordinate
translation and field-phase velocities are not nonlinear solutions of the
current rest orbit, and tangent background motion superposes independently
without changing the packet defect. Only a separately solved finite-amplitude
running `Psi`--`chi` branch could change this conclusion; no such background
is assumed or admitted.

The complete local `Psi`--`chi` frame has now been exhausted at this gate.
Besides the oriented area `W = chi wedge Psi`, it includes the parallel
overlap `s = chi^dagger Psi`. Two exact rest-preserving reciprocal action
directions expose that missing overlap response. Their union with every prior
action direction improves the one-cycle inverse residual from `0.587366` to
`0.191699`, but does not close the `0.05` gate, and the inverse coefficients
have no native provenance. Nothing was installed.

This narrows the next derivation to a signed spatial-temporal connection. It
must derive its orientation from `(s,W)`, vanish on the exact rest orbit, and
produce the odd-in-momentum response required by component-selective Dirac
transport while preserving both copies, positive kinetic mass, and the native
current. Symmetric overlap stiffness and the previously tested Bargmann
triangle are insufficient. V4-006 remains open.

## Odd C19 scalar--vector connection result

The lowest complete-frame first-order operator has now been derived. With
`tau=(chi dot Psi)/chi0` and `W=(chi cross Psi)/chi0`, cubic C19 geometry gives
`D(tau,W)=(-div19 W, grad19 tau+curl19 W)`, and direct evaluation verifies
`D^2=-sum_a G_a^2` to roundoff. This is an exact native spatial square-root
identity, not an inserted Dirac matrix.

The activated-relative real one-form built from this operator does not leave
the old rest state fixed, but it has a Fredholm-compatible jointly deformed
`Psi`--`chi` rest branch with quadratic nonlinear residual convergence. Its
linearized endpoint response is new: the total tested action-response rank
rises from nine to ten. It does not close motion, however. The full inverse
residual changes only from `0.1792865` to `0.1788594`, and no fitted
coefficient is admissible.

The next mandatory check is whether the Hodge field's native four components
`(tau,W)` are the same phase-space spinor as the historical corner readout.
The action remains frozen until that measurement bridge is proved. V4-006 is
open; no canonical or research action has been promoted.

The first bridge audit is complete. The density-activated field is unsuitable
for translation because its linear image has rank 32 and vanishes away from
the occupied corners. The unactivated field has rank 500 at configuration
level; adding the previous committed layer restores the full 1000 real
directions for both historical copies. Thus the second-order system already
contains the missing information as velocity. It does not yet form an
invariant Hodge sector, and a single positive-frequency choice keeps only one
rank-500 energy grade. The next derivation must recover the native beta
grading that combines the two grades into the four-component Dirac readout.

The unactivated Hodge action branch was tested as well. It improves the total
inverse residual from `0.1792865` to `0.1783271` but does not close V4-006.
No coefficient or action term was installed.

The beta derivation is now settled. The reduced `(tau,W)` scalar/vector chart
is one chiral Clifford representation and its beta anticommutant has dimension
zero. A configuration projector therefore cannot supply Dirac beta. Retaining
all eight cube cochain degrees instead reproduces the cubical incidence
generators exactly: corner parity anticommutes with all three axes, and the
native flavor involution yields two closed rank-four copies, all with zero
algebraic defect. The missing grade was cube information discarded by the
reduction, not a missing native field.

The existing all-center Fourier comparator is not accepted as the next motion
gate. It has a `2.42%`--`4.78%` error at zero momentum because it superposes
translated carrier frames while evaluating the tangent at one fixed localized
rest carrier. The next valid test is a finite joint `Psi`--`chi`
relative-periodic orbit between exact integer-translated endpoints, measured
with the complete cochain beta and both copies. V4-006 remains open and the
action remains unchanged.

The first finite-orbit screen rejects a one-update whole-cell jump. Letting the
same carrier cross the cell gradually is more than 800 times better and reaches
the translated endpoint to roundoff, but still leaves a nonzero DEL residual.
The residual is dominated by the canonical kinetic response of `chi`, not by
corner parity or copy leakage. The next gate therefore allows the intermediate
`chi` mold and its surrounding tail to deform in an unrestricted
relative-periodic boundary-value solve. This is an orbit/preparation test;
the action remains frozen until that solve determines whether a new reciprocal
term is actually necessary.

## V4-006 process correction

This historical correction is superseded in one respect by the 2026-11-25
gate correction at the top of this ledger. The extended same-carrier Dirac
packet remains a valuable continuum control, but it no longer replaces the
literal integer-cell endpoint as V4-006. What remains retired is a prescribed
rigid **interpolation** and receiver preload, not the exact translated
endpoint. The active test is an unrestricted finite relative-periodic orbit
of both committed layers. The long-time packet-sector defect remains evidence
about continuum transport, not a prerequisite that can reject a discrete hop.

## C19 triangle transport continuum gate

The current information-transfer research found that a covariant two-site
live-frame interaction remains rank two. The first rank-three response is an
oriented C19 face--edge triangle action exchanging `W=chi wedge Psi` together
with common and relative `Psi`--`chi` area velocities. Its apparent body-
diagonal form has an exact C19 triangle factorization; it is not a fundamental
C27 stencil. The candidate preserves the rest orbit and has all-axis linear
transport rank, but is not admitted because its long-time nonlinear/exterior-
`chi` behavior remains unresolved.

Its continuum equation-map gate passes. For both existing four-component
copies, the unchanged cubical Clifford matrices obey
`[2 alpha.s19(k)]^2=4|s19(k)|^2 I4`. Independently, the triangle action has
`lambda(k)=8[1-cos(kx)cos(ky)cos(kz)]`
`=4|s19(k)|^2+O(|k|^4)`. Consequently its massive dispersion and group
velocity converge to the same discrete-C19 Dirac values; the relative error
is second order in momentum. The action coefficient also converges to
`2/chi0` with second-order timestep error. No alpha, beta, Dirac matrix, or
target velocity occurs in the action.

This is not V4-006. Full live-field invariance of the same four-component
sector, equality of the action-derived Noether current, and the 100-cycle
nonlinear carrier gate remain open. Canonical V40 is unchanged.

The next same-state audit sharpened this conclusion. A coordinate mismatch in
the earlier component-ramp helper was corrected: ordinary action coordinates
had been unpacked with a mass-normalized diagnostic routine, suppressing the
`chi` perturbation by `1/sqrt(B)`. With consistent coordinates, all finite
tests retain rank three, and individual-copy and combined-copy sector leakage
both tend to zero with the timestep. The quadratic Noether current also
converges to the cubical Dirac current.

The first-order root still fails. The symmetric triangle/body symbol is even
in momentum and has zero derivative at the origin. The Dirac spatial operator
is odd, with nonzero `2 alpha_a` derivatives. Thus this candidate contains the
correct square, dispersion, and quadratic current but cannot by itself select
the signed first-order transport law. V4-006 remains open. Its next action
candidate must be a genuinely oriented, reciprocal, rest-preserving C19
face--edge one-form derived from the live fields; coefficient tuning inside
the symmetric family is retired.

The lowest pairwise oriented axial basis cancels identically under the full
cubic/translation sum. The previously tested genuine Bargmann loop is
nontrivial but sends more than 99% of its response outside the frozen carrier
and mixes the two copies. The remaining gate is therefore a higher native
loop or a derived first-shell carrier extension, not a retuning of either
tested triangle family.

The first-shell carrier extension has now been tested and rejected for this
Bargmann candidate. Its response power is entirely on the original eight
sites, with zero measured C19-shell power. A combined-copy translated basis
fits only by adding opposite-copy/internal directions; the single-copy
residual is `0.730628`. The remaining obstruction is internal signed routing
and copy closure, not missing spatial support. V4-006 remains open and no
action has been promoted.

A final fixed-basis control rules out the simpler explanation that the two
copies were merely labeled incorrectly. Each axis separately produces eight
independent responses, but the three axes jointly produce 24. The only matrix
that commutes with every directional response metric is the identity, so no
single rotation of the existing eight labels supplies a common four-plus-four
grading. The Bargmann candidate is therefore retired as a V4-006 repair. The
next action candidate must derive a common low-rank oriented carrier rather
than fitting these extra responses into a larger readout.

Movement research is now locked back to the literal same-assignment cell
translation in `EXACT_TRANSLATION_MAP_LOCK.md`. A solved first-shell precursor
partially prepares the receiver, confirming the make-before-break mechanism,
but neither the canonical action nor the earlier live-frame kinetic action
closes the update. An exact regularized dual-frame transporter repairs the
nonorthogonal-frame mixing missed by the earlier projection formula. At the
derived `kappa=1/63` it still leaves a core-chi timing defect and an implicit
next-layer correction of at least `17.96%` of a cell step. Near-singular
regularizers are rejected. V4-006 remains open; the next action derivation is
a signed-current-activated reciprocal spacetime bond connection, with the
exact translation map retained as the sole movement gate.

The subsequent direct-prong calculation preserves internal labels across
every bond. Positive same-component springs and current-directed averaging do
not close the handoff. A density-flow material action locates the correct
signed spatial direction but loses the phase/orientation data needed to set
the receiver exactly. The current research incumbent therefore derives one
local velocity from all real and imaginary `Psi_x/y/z` changes after removing
the local common phase, and uses it to advect matching `Psi` and `chi`
components. This reduces the optimized x-positive exact-map residual to
`0.06987` while preserving rest, but leaves z anisotropy and a near-singular
implicit update. It is a research action screen only: no action is promoted,
the canonical action remains unchanged, and V4-006 remains open.

The full direct-prong follow-up now identifies the boundary more sharply.
Independent per-prong transport is unstable, whereas one common velocity for
all Psi/chi components becomes Floquet-stable when combined with the prior
rest-null receiving-shell sector.  That stable candidate nearly matches the
forward discrete-Dirac Psi displacement but produces transverse Psi drift and
does not carry chi with the same center.  Both-copy cancellation and a
relative/common velocity-coherence metric fail to repair it.  V4-006 remains
open: no integer same-pattern handoff or repeated translation has passed.
See `experiments/v4_localized_electron/DIRECT_PRONG_V4_006_REPORT.md`.

The next live-frame audit resolves the first error into two native pieces:
`48.58%` is parallel to `chi` and `51.42%` is transverse.  The existing
`chi cross Psi` area channel therefore omits half of the information needed by
the receiver; its complementary invariant is `chi dagger Psi`.  Adding the
corresponding positive C19 bond or completing both invariants as material waves
improves selected shape metrics, but redirects the correct forward current and
does not close a cell handoff.  `Psi`-only velocity selection is rank-deficient
and unstable, while allowing `chi` to complete that rank without its native
inertia is also unstable.  No term is admitted.  The open inverse requirement
is now a rest-null, odd-in-native-current transfer of the *complete* live frame,
not another unsigned stiffness or inertia term.

The authorized action-repair pass tested that inverse requirement explicitly.
Exact rest-null cube-symplectic and curl--overlap terms improve packet shape
and norm together, but increase complementary-sector leakage. The best
combined case changes the ten-cycle triplet `(shape, leakage, norm drift)`
from `(1.42102, 0.05069, 0.44868)` to
`(1.28501, 0.07810, 0.21729)`. A native C19 triangle-current term also fails
the localized carrier despite its correct quadratic continuum-current limit.
These are rejected research deltas, not a canonical action change. V4-006
remains open; the next mathematical gate is a rest-null rank-four carrier
routing invariant with a vanishing cross block.

The next action calculation derives that missing rank-four routing directly
from the live fields. Normalized native `Psi` edge phases give the three
Clifford generators and two rank-four copies. The relative normal-axis
`Psi`/`chi` link also derives the native beta grading, rather than importing
it. At the carrier/exterior boundary the scalar `chi` overlap is exactly zero,
which explains why the earlier live-link terms could not prepare a receiver.

The new research interaction uses the smooth normalized squared `chi` wedge
to gate componentwise neighbor kinetic communication for both complex C3
fields. Its face-pair control preserves rest and produces the exact translated
four-corner `Psi` face. Both copies, both motion signs, an axis and a diagonal
then pass four unrestricted nonlinear small-amplitude updates with less than
`0.997%` discrete-Dirac error and `0.0273%` sector leakage. The fully weighted
C19 form is strictly positive with kinetic-metric bounds `[18/19,20/19]` and
passes its first nonlinear packet control, but it produces a broader edge
precursor and has not yet reconstructed the full-amplitude carrier.

The movement clock was also corrected. With `dt=1/256`, one cell per numerical
update would require speed 256. The discrete massive-Dirac group speed at the
declared `k=0.35` packet is `0.0169503`, requiring about `15103` updates per
cell; even the fastest axis momentum in the grid control requires about
`9735`. The next gate must therefore integrate the full C19 precursor over the
Dirac-derived crossing interval. The research action is not admitted,
canonical V40 is unchanged, and V4-006 remains open.

## Six-face transport correction

The full-C19 precursor interpretation is superseded for the added matter
current, not for the native field stencil. Both governing equations retain
C19. The research transport term uses the six face bonds only. A coefficient
`1/19` per undirected face bond gives the derived aggregate coordination
`6/19`, matching the proposed matter-channel ratio without fitting a bond to a
Dirac velocity.

The strongest version is the smooth positive joint `Psi`--`chi` face
Laplacian recorded in `live_face_transport_verdict.json`. It preserves rest,
is globally positive, reconstructs the exact first `Psi` receiver face, raises
the first `chi` target overlap to `0.964`, and passes all eight four-update
Dirac controls. It does not complete a cell translation. Compact-envelope
error exceeds one percent at update five, while the uniform zero-momentum
control remains below one percent through update nine. An arbitrarily stiff
version retains a finite within-sector error, so the next missing object is a
lossless compact-gradient/current bridge, not more strength in this same
term. Canonical V40 is unchanged and V4-006 remains open.

## Corrected frequency-sector and six-face endurance verdict (2026-11-24)

The rest carrier proved in `V2.RestOrbit` advances with **positive** complex
phase. The previous moving-packet controls nevertheless selected the positive-
energy Dirac eigenspace, whose laboratory phase advances with the opposite
sign. The matching comparator is the negative-frequency/negative-energy
sector with the corresponding rest clock removed. This is a readout and
preparation correction only: no native force, action term, state projection,
or runtime support changed.

With that correction, the unadmitted six-face `Psi` plus oriented-area
oscillator at scale `6/19` passes all eight unrestricted four-update controls:
both cubical copies, both signs, one axis and the diagonal. Worst native-versus-
centered-Dirac error is `0.234926%` and worst four-component leakage is
`0.0105723%`. One selected nine-update case remains below one percent at
`0.986520%`, with `0.0355710%` leakage. The odd finite-amplitude response
reproduces the same four-update result, so the improvement is not an even
nonlinear cancellation. The action-derived quadratic Noether charge moves in
the same longitudinal direction as the corrected Dirac packet; the earlier
opposite-direction diagnosis is retired.

This does **not** close V4-006. At 32 unrestricted updates the selected case
reaches `6.27058%` error, although leakage remains only `0.189386%`. The exact
linear tangent reaches a worst one-cycle phase-aligned error `1.35980` and
norm drift `0.818635`, with cross-block leakage `0.0199901`. Thus the first
remaining departure is predominantly inside the retained four-component
sector, not loss of the copy or failure to reach a receiving face.

The coefficient audit also separates the geometric statement from a fit.
`1/19` per undirected face produces six incident variations, but gives
`1.75764%` error at nine updates. Using `6/19` per bond improves the short run
but is six times the per-bond normalization. An exact affine scale scan shows
that merely increasing this same invariant cannot provide exact closure: its
one-cycle shape error approaches a nonzero floor of about `1.49%` even near a
million-fold diagnostic strength. No such fitted strength is admitted.

The existing reflection-covariant C19 triangle term is worse (`2.24581%` at
nine updates), and the simple signed face-frame companion fails the random-
state reflection-covariance gate. Changing the Dirac readout speed to `6/19`
or `2` also worsens the one-cycle comparison. The remaining mathematical
requirement is therefore a distinct reflection-covariant, oriented native
response that corrects the within-sector phase/transport block while
annihilating the exact rest orbit. Canonical V40 is unchanged, no research
action is admitted, and V4-006 remains **OPEN**.

## 2026-11-25 complete-frame timing verdict

The timing study found one material omission and repaired it without changing
canonical V40. The wedge field `chi cross Psi` does not contain the parallel
complex relation `chi^dagger Psi / chi0`. Completing that missing frame
component with its reciprocal material kinetic term and matched rest-clock
potential reduces the original one-cycle shape error by roughly a factor of
fourteen and preserves the rest orbit to about `2.5e-13` relative DEL error.

The best all-eight result has shape error `0.0975517670`, leakage
`0.0035972665`, and norm drift `0.0332510190`. Leakage and norm pass; the
`0.05` shape gate does not. Spacetime-face, symplectic, triangle, Hodge,
mass-readout, transverse-material, and both-copy quadrature controls do not
close the remaining in-sector block. Broader packets approach the continuum
gate but exceed the side-five box and are not a localized pass.

The repaired exact rest orbit also has no neutral infinitesimal translation
mode. Its response is `84.17%` outside that smooth-translation subspace, and
the seven-direction rank audit leaves `90.2699%` of that residual. This
retires the tested vocabulary as a neutral-slide repair, but does not obstruct
a finite integer-cell relative-periodic orbit.

V4-006 remains **OPEN**. Its primary next calculation is the finite-hop
boundary-value problem on the stable action; the arbitrary-momentum Bloch
symbol remains a separate continuum comparator. Full timing evidence and RCA
are in
`experiments/v4_localized_electron/TIMING_REPAIR_REPORT.md`.
