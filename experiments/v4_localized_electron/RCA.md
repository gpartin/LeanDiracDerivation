# V4-002 root-cause analysis

**Issue:** The exact localized V40 rest orbit is linearly unstable under the
complete native two-layer tangent evolution. The best bounded research action
tested so far greatly reduces but does not remove the growth.

**Why it matters:** V4 requires the same localized carrier to survive before
its four-component readout, spin, translation, density, and current can be
identified with one physical electron. An exact orbit is not sufficient when
arbitrarily small perturbations grow at a finite physical rate.

**Symptoms:** On the unchanged action (`B = 1197`), the largest Floquet
multiplier gives growth rates `1.041396`, `1.038175`, and `1.037373` at
`dt = 1/128`, `1/256`, and `1/512`. The result is identical on side-three and
side-four periodic boxes. The leading mode is completely supported on the
eight carrier sites and is an independent corner amplitude/phase shear. It is
not a translation mode, exterior radiation, boundary reflection, or timestep
artifact.

**Expected symptoms:** A passing rest orbit has no finite positive growth rate
as the timestep is refined, preserves reciprocity and energy structure, and
remains stable when the box is enlarged.

**Root cause:** The canonical local density response and the single quadratic
bond-area response do not provide enough independent action-Hessian response
to stabilize all internal corner directions. A bounded generic bond polynomial
can suppress the original amplitude/phase mode, but the dominant residual then
becomes a transverse third-component `chi` mode. Local density,
density-memory, parallel-overlap, and onsite area-polynomial invariants either
have zero response in that direction or move the instability into a different
corner mode. Positive two-hop C19 curvature and positive elementary-plaquette
curvature were then tested. Both worsen the spectrum. Even the inadmissible
negative plaquette probe improves growth by only 0.36% before becoming worse
and deforms the exact orbit. The deficiency is therefore not a shortage of
scalar spatial stiffness. The action lacks a response that controls the
relative timing and orientation exchange of `Psi` and `chi` while corners hand
the pattern to one another.

**Corrective action:** Derive a phase-/velocity-resolved reciprocal `Psi`--`chi`
action basis whose Hessian couples relative field timing to shared-corner
orientation. Rank-test it on the unresolved transverse `chi` mode and the
original in-plane shear before any coefficient sweep. Require locality,
C19/cubic symmetry, bounded Hamiltonian energy, an action-derived reversible
update, the exact rest orbit or a freely re-solved deformation, and unchanged
leading low-momentum C19 propagation. Do not promote the scalar curvature
terms, the diagnostic bond polynomial, or provisional `B = 229`.

The first velocity-resolved basis has now been tested. The positive local
square `|chi x PsiDot - chiDot x Psi|^2` produces a nonzero response and admits
an exact re-solved rest orbit, but increases growth by a factor of `4.08` at
the first tested positive strength. Negative diagnostic strengths also worsen
growth and are not globally positive. This narrows the corrective action to an
antisymmetric/gyroscopic phase response derived from a real action. More
inertia in the relative channel is not the missing mechanism.

## Gyroscopic repair cycles 1--5

**Issue:** The symmetric kinetic repair failed, while the exact carrier still
has a positive internal growth rate.

**Why it matters:** A phase-steering action must remove every unstable corner
mode, not merely exchange which mode is dominant.

**Symptoms:** The lowest local relative gyroscopic invariant lowers growth
from `0.0755557` to `0.0741703`. Adding its independent C19 bond form lowers it
to `0.0731762`. The elementary-plaquette form worsens growth even after the
eight ray amplitudes and phases are freely re-solved. The complementary common
orientation Berry invariant then lowers growth to `0.0705874`; independent
`chi` and `Psi` Berry controls reach `0.0704348`. Every improved branch ends at
a mode crossing between transverse `chi_3` and in-plane `chi_1,chi_2`
phase/radial corner modes. The best result is a `6.78%` reduction, not
stability.

**Expected symptoms:** A passing action drives the converged physical growth
rate to zero without destroying the exact/deformed eight-ray carrier, energy
boundedness, reciprocity, or the native translation tangent.

**Root cause:** The original potential supplies only position curvature, and
the diagonal kinetic law supplies no phase-space steering. Antisymmetric
one-forms supply the missing response and improve the correct metric, proving
that the diagnosis is directionally correct. The repeated mode crossings show
that constant local/bond Berry coefficients still lack enough amplitude-
resolved response rank to control transverse and in-plane phase/radial modes
simultaneously.

**Corrective action:** Enumerate the complete amplitude-resolved local Berry
basis before adding more spatial range. Rank-test density-weighted relative
and common one-forms on both crossing modes. If that basis is deficient, move
to the lowest-degree C19 common-neighbor triangle one-forms. Coefficients found
by continuation remain diagnostic; derive final values only from positivity,
spectral degeneracy, or a native bifurcation condition.

## Cube-geometry isolation

**Issue:** The amplitude-resolved gyroscopic search continued to improve the
full localized spectrum, but the resulting collection of one-forms was not a
reduced action and its coefficients lacked native provenance.

**Why it matters:** A fitted collection of stabilizers cannot establish that
the electron mechanism emerges from the cubic substrate. The improvement must
be attributable to a smaller reciprocal action whose coefficients follow from
the native geometry or a proved bifurcation condition.

**Symptoms:** The converged diagnostic scaffold lowers the leading growth from
`0.0755557` to `0.0454024`. Adding the reciprocal normalized-area common and
relative one-forms with the cube values `-2/chi0^2` and
`-1/(3 chi0^2)` lowers it further to `0.0412335`, while the rest residual stays
below `8e-12`. Either normalized term alone gives only `0.0485406` or
`0.0451095`; the gain is a joint response. The dominant residual is repeatedly
supported at C19 symbol `27`, not at the spatially uniform symbol. Replacing
all continued coefficients at once by nearby simple ratios instead worsens
growth to `0.0645913`. Removing any of the old gyroscopic, field-Berry,
density-sensitive, or overlap groups also worsens the spectrum; the current
candidate therefore remains scaffolded.

**Expected symptoms:** A reduced cube-derived action should reproduce the
joint common/relative improvement with one declared geometric construction,
remove rather than require the older fitted groups, and drive every unstable
C19 eigenspace to zero growth.

**Root cause:** The useful response is not a scalar onsite stiffness. It is an
amplitude-sensitive exchange between the common and relative rotations of the
oriented area `chi x Psi`. Its leading unresolved eigenmode is selected by the
face/edge C19 geometry (symbol `27`). The local normalized-area terms see the
right internal variable but do not themselves resolve that spatial cube mode;
the older terms are compensating for the missing bond-resolved pullback.

**Corrective action:** Construct the C19 bond pullback of the normalized
oriented area, using all six face and twelve edge bonds with their canonical
weights. Derive both reciprocal field equations and the canonical momenta from
that single real one-form, verify its variation independently, and rank-test
it first on the symbol-`27` unstable subspace. Admit it only if it replaces the
older fitted groups and preserves the exact rest orbit, energy structure, and
low-momentum propagation. Do not promote the current coefficient scaffold.

## C19 normalized-orientation action test

**Issue:** The leading residual carried C19 symbol `27`, suggesting that the
missing response might be the bond transport of the normalized oriented area
instead of another local term.

**Why it matters:** A genuine cube-geometry repair must survive timestep
refinement. Improvement at one integration step is not evidence for a stable
native action.

**Symptoms:** The real reciprocal one-form
`Im[Z^dag D19 Zdot]/sum(w)`, with
`Z=chi0(chi x Psi)/sqrt(chi0^4+|chi x Psi|^2)`, passed an independent first-
variation check. At `dt=1/256`, coefficient `1/(6 chi0)` lowered growth from
`0.0412335` to `0.0393285` and moved the leading mode from symbol `27` to
symbol `0`. The apparent gain did not converge: at `dt=1/512` and `1/1024`
the rates were `0.0960` and `0.1001`, while the zero-coefficient `1/1024`
control was `0.0419538`. Jacobian-step refinement left those rates unchanged.

**Expected symptoms:** The same fixed action coefficient should give a
consistent non-increasing growth rate under `dt`, `dt/2`, and `dt/4`, with no
discrete resonance.

**Root cause:** The bond Berry one-form changes the correct C19 eigenspace but
does so through the discrete symplectic map. Near `dt=1/256` it creates a mode
crossing that temporarily hides the symbol-`27` instability. It does not add
the positive reciprocal position curvature required by the continuum action.

**Corrective action:** Retire the bond one-form as a stabilization candidate.
Keep it only as evidence that the unresolved mode is geometrically
addressable. Test the paired positive normalized-area wave action, whose C19
kinetic and restoring terms share the same native dispersion.

## Positive normalized-area wave follow-up

**Issue:** The bond one-form lacked a paired restoring energy.

**Why it matters:** The replacement must have a bounded Hamiltonian sector and
must improve the continuum-oriented spectrum rather than one selected
timestep.

**Symptoms:** The paired action
`eta[<Zdot,D19 Zdot>-<Z,D19^2 Z>]/(2 sum(w))` passed independent variation
checks. For the geometric trial `eta=1/(12 chi0^2)`, the growth rates at
`dt=1/256`, `1/512`, and `1/1024` are `0.0407625`, `0.0414440`, and
`0.0415247`. The matched zero-eta controls are `0.0412335`, `0.0418991`, and
`0.0419538`, respectively. The improvement therefore converges, but is only
about one percent and leaves a finite symbol-`27` instability. Increasing eta
past roughly `2.5e-4` exposes a faster mode and eventually prevents the rigid
rest ansatz from solving.

**Expected symptoms:** A complete repair must bring the largest continuum
growth rate to zero and then permit removal of the fitted gyroscopic scaffold.

**Root cause:** The cube-wave term supplies the missing positive spatial
response but not the independent local phase/orientation response. The
remaining spectrum is a competition between a symbol-`27` transverse mode and
a spatially uniform transverse `chi` mode. One positive scalar coefficient
cannot remove both.

**Corrective action:** Retain the positive wave term as the first convergent
cube-geometry contribution, but do not admit the action. Derive the local
connection from the same normalized geometry and solve the two-mode
degeneracy condition analytically. Then repeat group ablations; success
requires that the unified geometric action replace the older fitted groups.

## Search-method failure and directional-potential retirement

**Issue:** Four hours of coefficient continuation and repeated orbit solves
did not close the rest-stability gate.

**Why it matters:** Repeatedly tuning coefficients inside an action class does
not determine whether that class can generate the missing response. It can
consume arbitrarily many runs while leaving the structural obstruction
unchanged.

**Symptoms:** The continuously extracted co-rotating generator retains growth
`0.03270684`, proving that the residual is in the action rather than the time
integrator. The local overlap potential has exactly zero response on this
mode. The positive cubic directional-area potential
`sum w |n_hat dot (W_i-W_j)|^2/(2 chi0^2)` creates a faster mode: at coefficient
`+1e-8` the leading continuous growth is `0.03654477`. The opposite sign gives
only `0.03270672` and is inadmissible because it makes this energy contribution
unbounded below.

**Expected symptoms:** An admissible basis must have a stabilizing projection
on every unstable eigenspace before an expensive nonlinear rest solve is run.
It must also retain response on all three translation tangents.

**Root cause:** The research loop was ordered incorrectly. It solved and tuned
full orbits before calculating the action-to-mode response map. The positive
directional bond energy also has the wrong curvature sign on the unresolved
mode; no positive coefficient can repair it.

**Corrective action:** Retire the local overlap and directional-area potential
bases. Freeze coefficient sweeps. Build the projected response matrix of each
complete admissible action basis on the unstable eigenspaces and the three
translation tangents. Solve the finite-dimensional rank/sign problem first.
Only a basis with the required response rank and admissible energy sign earns
one unrestricted deformed-rest solve. This replaces blind continuation with
an inverse variational calculation.

## Full-spectrum Hamiltonian correction

**Issue:** The first projected inverse calculation targeted only the twelve
largest unstable eigenvalues and therefore did not constrain the neutral and
smaller-growth Hamiltonian modes.

**Why it matters:** In a gyroscopic Hamiltonian system, opposite-Krein-sign
neutral modes can collide and leave the imaginary axis. Reducing the current
largest eigenvalue can expose or create a different unstable quartet, so a
leading-mode objective cannot establish stability.

**Symptoms:** The signed one-form response matrix has full rank `7` on the
seven distinct leading growth branches and predicts an exact first-order
cancellation. A damped inverse step of `1e-4` gives only the microscopic
improvement `0.0327068435 -> 0.0327049154`; steps `5e-4`, `1e-3`, and `2e-2`
produce growth `0.0368614`, `0.0327729`, and `0.334330` through mode crossings
and rest-branch motion. The exact Hamiltonian reconstruction satisfies
`||G^T Omega + Omega G||/(||G|| ||Omega||)=5.40e-14` and
`||-Omega G-S||/||S||=2.63e-16`. It reveals 23 eigenvalues with growth above
`1e-2`, 55 above `1e-4`, and 86 above `1e-6`, not one residual mode.

**Expected symptoms:** A valid stabilization has no finite positive real part
anywhere in the symmetry-reduced spectrum and preserves only the phase and
translation zero modes. It must not stabilize one Krein collision by creating
another.

**Root cause:** The research objective was the spectral abscissa of the
currently leading mode rather than the quadratic Hamiltonian and symplectic
structure of the complete relative equilibrium. Signed Berry terms move
frequencies and exchange which collision is dominant but do not provide a
global stability certificate.

**Corrective action:** Replace leading-eigenvalue tuning with a complete
Hamiltonian--Krein calculation. Identify the physical symmetry nullspace,
classify every neutral mode by energy signature, and measure which native
joint-field invariant sees each colliding subspace. Then search one complete
positive joint `Psi`--`chi` metric/potential basis whose augmented quadratic
form removes every opposite-sign collision while retaining the three
translation tangents. Only after that matrix condition passes should the
nonlinear rest orbit and long-time gates run.

## Complete joint-field observability and positive-metric tests

**Issue:** The wedge coordinate `chi x Psi` used by the current bond action is
not a uniformly sensitive coordinate for the broad unstable spectrum.

**Why it matters:** An action cannot stabilize modes that its native
coordinates do not observe, but observability alone does not guarantee that a
positive energy has the required curvature sign.

**Symptoms:** For all 28 positive-growth eigenvectors above `1e-4`, the scalar
overlap and density coordinates are blind to four modes. The wedge response
ranges from `0.00063` to `0.804`. The full cross-covariance `chi Psi^dagger`
sees every mode uniformly, with normalized response `0.702--0.987`. A positive
one-hop C19 bond energy of its normalized form improves maximum growth from
`0.03270684` to `0.03268831` at strength `0.001` and reduces the counts above
`1e-2` and `1e-4` from `11/28` to `9/26`. Larger strengths trigger new Krein
collisions. The complete normalized joint-projector bond energy gives the same
qualitative result (`0.03268673`, `10/26` at `0.001`) and also worsens beyond
the first collision. The cross-covariance kinetic metric always worsens the
leading rate. Its restoring energy is the only helpful component.

**Expected symptoms:** A sufficient native coordinate/action pair must see all
material modes and make the entire symmetry-reduced Hamiltonian spectrum
non-growing, not merely remove two unstable eigenvalues.

**Root cause:** The current carrier is not close to a constrained energy
minimum of the research action. The Hamiltonian audit contains 114 neutral
negative-signature modes in addition to the unstable quartets, far more than
the two field-phase constraints can remove. Small positive bond metrics can
shift individual collisions but cannot change that high-index saddle into a
stable relative equilibrium.

**Corrective action:** Retire incremental stabilization of the scaffolded rest
orbit. Return to the declared native action class and solve the carrier as a
constrained variational branch at fixed native Noether charges, using the
eight-site pattern only as initialization and an identity measurement. Apply
the energy--momentum/GSS index test before Floquet simulation. If no branch
with the required negative index and carrier overlap exists, derive a new
generic local joint-field potential/action family and test its constrained
Hessian directly. Do not add another stabilizer to the current scaffold.

## Binding-first reset

**Issue:** The research attempted to stabilize the prepared eight-site shape
before checking whether the local action energetically selects any finite
localized `Psi` density.

**Why it matters:** A generic orientation or bond correction cannot turn an
unbound high-index saddle into a particle. The local binding condition must be
settled before spending more runs on spinor timing or transport.

**Symptoms:** Eliminating the local `chi` magnitude from the canonical
potential gives, for `p=rhoPsi` and `c=rhoChi`,
`c*=chi0^2-p` and `Ueff(p)=chi0^2 p/2` throughout the interior branch. The
effective local energy is exactly linear: it contains no finite-density
minimum and no attractive self-interaction. The canonical rigid charge branch
has constrained Hessian index 117 at its least unstable sampled point and
continuous growth `1.0371`; changing its charge ratio only worsens it. The
scaffolded research orbit remains a high-index saddle after all continuous
symmetries are removed (140 negative directions).

**Expected symptoms:** A viable localized carrier must arise as a constrained
energy critical point with the correct small index after symmetry removal,
before its Floquet spectrum, motion, or Dirac readout can pass.

**Root cause:** Canonical coefficients place the local two-field potential
exactly on the threshold between repulsive and attractive effective `Psi`
self-interaction. Positive bond energy then favors spreading. The prior search
was repairing orientation and timing on a state that the action does not bind.

**Corrective action:** Test the smallest generic local family
`Ulocal=(c-chi0^2)^2/4 + g c p/2 + lambda p^2/4`. It is bounded for
`lambda>0`, contains no prepared geometry, and has a strict local binding
window when `0<lambda<g^2`. Use the constrained energy index as the cheap gate.
The first rigid-cube scan confirms that opening this window changes the growth
substantially (`1.0371` down to `0.4164` in the best sampled row) but does not
stabilize the rigid ansatz; its negative index remains large. Therefore do not
tune this rigid branch further. Solve the unrestricted localized constrained
profile, permitting the carrier and first shell to deform while measuring
retention of the same eight-site identity. If that deformed branch still has
the wrong index, retire this whole quartic local family before testing any new
action term.

## C19 cube phase obstruction and inverse requirement

**Issue:** The retained eight-site area pattern is an exact stationary phase
configuration, but the positive quadratic C19 area-bond action does not make
it a stable phase configuration.

**Why it matters:** An exact relative orbit can still be a saddle. No amount
of coefficient continuation in unrelated local, inertia, or halo terms can
repair phase directions on which the native bond energy itself has negative
curvature.

**Symptoms:** The exact phase-only Lyapunov--Schmidt gate in
`cube_phase_ls.json` gives the target six negative directions, one additional
zero direction, and no positive direction after the common phase is removed.
All `4^7 = 16384` quarter-turn assignments were checked after fixing the
global phase. Of 158 stationary assignments, the only strict minimum of the
positive quadratic C19 bond energy is the uniform phase. The target has
normalized bond energy `48` and Hessian eigenvalues
`{-8,-8,-2,-2,-2,-2,0}`.

**Expected symptoms:** A viable native interaction must leave the retained
pattern torque-free and give nonnegative curvature in all seven non-global
phase directions while retaining ordinary aligned-field wave stiffness.

**Root cause:** The current `|W_j-W_i|^2`, `W=chi wedge Psi`, interaction
places the required quarter-turn/half-turn area pattern at a high-index C19
phase saddle. Earlier action searches treated the resulting shears as
independent timing and stiffness defects instead of auditing the phase
critical point first.

**Corrective action:** Use `phase_action_inverse.py` as the mandatory cheap
gate. It proves that all face/edge bond potentials built only from first and
second phase harmonics are infeasible under the simultaneous stationarity,
aligned-wave-stiffness, and positive-target-curvature conditions. Allowing
harmonics through four gives the exact minimum-L1 oracle
`(1-cos(4 Delta theta))/16` on both bond classes. Do **not** install this
oracle: it directly selects quarter turns. Derive the same fourth-order
response by eliminating or dynamically solving a generic complex-chi/cube
mediator, then repeat the exact-rest gate before any spectrum or simulation.

## Rejected nonstationary quartic probe

**Issue:** A preliminary positive quartic face/edge combination was selected
from target curvature alone and failed the exact-rest test.

**Why it matters:** Positive Hessian curvature is irrelevant if the candidate
exerts a nonzero phase torque and destroys the orbit it is meant to stabilize.

**Symptoms:** At the geometric normalization `eta=2 chi0=38`, the old carrier
has relative Euler--Lagrange residual `0.0300` for `lambda=1` and `0.0418` for
`lambda=0.25`. The per-corner force contains alternating imaginary radial
coefficients. The exact phase-gradient audit gives alternating torque
`(-3.5,+3.5,...)` before normalization.

**Expected symptoms:** An admissible replacement must have phase torque below
the algebraic tolerance before any freely deformed rest solve is attempted.

**Root cause:** The first inverse problem constrained only the target Hessian
and aligned stiffness; it omitted the zeroth-order stationarity equations.

**Corrective action:** Reject this candidate and retain its output as a
negative control. Every later inverse solve must impose phase torque, aligned
stiffness, target curvature, vacuum behavior, and reciprocal-action
constraints simultaneously.

## Superseded shared-face-continuity control

**Status correction:** This control is retained as provenance but is
superseded by `edge_snap_verlet_gate.json`. Requiring the four shared sites to
retain their old values was not a requirement of a literal nearest-neighbor
shift.

**Issue:** Earlier movement trials translated the eight-site carrier into the
same local corner assignment in the adjacent cube.

**Why it matters:** Adjacent 2x2x2 carriers share four sites. If their native
fields disagree on that shared face, no amount of extra forward impulse can
produce a smooth identity-preserving handoff.

**Symptoms:** The same-copy source and destination differ on the shared face
by relative error `0.9574` after the best complex scalar alignment for both
`Psi` and `chi`; even a full common internal unitary leaves error `0.8420`.
Reflecting the local corner assignment in the direction of motion makes all
four shared `Psi` and `chi` values agree exactly in all six signed coordinate
directions. The source and destination then occupy 12 distinct sites, and a
cosine/sine `Psi` transfer with a quarter-phase destination preserves charge
to `2.3e-16` while moving its center continuously by one cell. A second
reflection restores the original copy.

**Expected symptoms:** A valid native translation branch should pass through
this 12-site shared-face family, exchange reflected copies once per cell, and
return to the original copy after two cells while satisfying the same action
without projections or receiver preload.

**Root cause:** The old movement preparation connected geometrically
incompatible endpoints. A reflected local corner assignment is the compatible
receiving pattern. This is a kinematic result, not yet a freely evolving
trajectory or a proof of Clifford-flavor exchange. The existing cubical
algebra has signed translation generators that preserve each flavor sector;
raw reflection exchanges the flavor grading on only one of the three chosen
bit axes.

**Corrective action:** Replace the same-copy translation control with the
reflected-assignment 12-site path in `copy_alternating_handoff.json`. The unchanged
canonical action has a seeded midpoint potential barrier of about `1.694e4`
at unit carrier amplitude in every tested direction, so geometry alone does
not close motion. Next minimize live `chi` freely along this fixed-charge path
to separate preparation cost from the irreducible Peierls--Nabarro barrier.

## Corrected complex-Hessian handoff audit

**Issue:** The first Fredholm test of the receiving `chi` response used nested
automatic differentiation through complex `abs(z)^2`. PyTorch retained the
first derivative but omitted some second-order complex curvature, so the
reported `78.21%` obstruction and the old carved-substrate tangent were not
authoritative.

**Why it matters:** The action decision depends on the Hessian range. A false
nullspace can make an adequate native response look impossible or make a
spurious repair appear necessary.

**Symptoms:** Rewriting every complex square explicitly as
`real(z)^2+imag(z)^2` changes the canonical full-box chi Hessian by `0.6644%`
in norm, raises its rank from `143` to `159`, and changes the prescribed-path
cokernel fraction from `0.78206` to `0.49834`. The forcing itself agrees to
`4.45e-9` relative error, confirming that the defect was second order. The
corrected carved-substrate full tangent also shows that positive strength
worsens the canonical maximum growth (`1.03494` to `1.03674` at `eta=0.25`),
so it remains rejected.

**Expected symptoms:** A valid receiver mechanism must put the complete mixed
Psi-to-chi forcing in the range of the corrected chi Hessian while leaving the
exact rest orbit and its internal corner phases stable.

**Root cause:** The canonical action has no carrier/vacuum interface
orientation curvature when the core `chi` directions are orthogonal to the
vacuum `chi`. The generic positive reciprocal probe
`sum w (rhoPsi_i-rhoPsi_j)^2 |chi_i^dagger chi_j|^2/(2 chi0^6)` supplies exactly
that curvature. Its value and first derivative vanish on the exact carrier,
and it reduces the corrected Fredholm obstruction to `1.56e-13` without the
carved term. This is a structural channel result, not a fitted coefficient.
The term does not respond to density-preserving internal corner phase shear:
core-core density contrast is zero and core-vacuum overlap is zero. Its full
rest tangent therefore worsens maximum growth monotonically over the tested
positive strengths even while reducing the number of unstable branches.

**Corrective action:** Retain the interface invariant as a transport-response
basis element, not as an admitted action or a complete repair. Do not tune its
coefficient yet. Derive a separate generic complex-chi/cube mechanism that
generates the required fourth-harmonic corner curvature, combine the two only
at the action level, and require the combined exact-rest spectrum to improve
before attempting nonlinear movement. All future complex Hessians must use
explicit real/imaginary squares and must lock `B=1197` after importing
historical modules; one audit caught an import-time mutation to `B=229` before
the result was recorded.

## Literal one-edge snap recurrence

**Issue:** The earlier handoff test treated continuity of the values on the
geometrically shared face as mandatory. That is the wrong condition for a
nearest-neighbor lattice shift, where the old shared-face values advance and
are replaced by values arriving from the trailing face.

**Why it matters:** The reflected endpoint leaves four sites fixed and maps
four sites across two face spacings. It can diagnose a PN interpolation but it
cannot test the user's hypothesis that every spinor-supporting corner advances
one edge per substrate tick.

**Symptoms:** `edge_snap_verlet_gate.json` proves that the same-assignment
translation is an exact local permutation: all eight `Psi` and `chi` carrier
sites move one face-neighbor spacing in both signs of all three axes with zero
endpoint defect. The reflected control instead has hop distances
`{0,0,0,0,2,2,2,2}`. For a traveling two-layer ansatz
`q_(n+1)=exp(i phi) S q_n`, `q_(n-1)=exp(-i phi) S^-1 q_n`, the best unchanged
canonical Verlet match still has combined relative residual `0.99976` to
`0.99981`, `Psi` residual `0.51758`, and `chi` residual exactly `1.0`. The
required chi second-difference norm is `110.84` to `124.05`, while the native
chi acceleration norm is `1.08e-15`. The best Psi tick phases fall into the
geometric set `0`, `+/-pi/2`, and `pi` by axis.

**Expected symptoms:** A native edge-snap branch has one common positive
`dt^2` for both fields and makes the two-layer recurrence residual vanish in
all six directions while retaining the exact corner pattern and conserved
charge. The integrator step need not equal this solved substrate period.

**Root cause:** At the exact static-chi carrier, the position-only canonical
action gives essentially zero chi acceleration. The same spatial state
therefore receives the same force whether its previous layer encodes rest or
translation. It cannot produce a rigid momentum-dependent conveyor shift of
chi. The interface invariant found by the Fredholm audit has zero first
derivative on the rest carrier, so it supplies receiver curvature but cannot
by itself launch the exact shift either.

**Corrective action:** Replace shared-face equality with the literal
same-assignment shift as the movement comparator. The next admissible action
basis must provide a reciprocal motion-dependent response, most naturally a
C19 bond-resolved antisymmetric/symplectic Psi--chi coupling, or else yield a
freely deformed traveling branch whose tick endpoints converge to this shift.
Derive its momenta, Hamiltonian and reversible update first; never install the
shift permutation, Dirac projectors, or direction labels as native rules.

## Positive native two-site phase-basis no-go

**Issue:** It remained possible that the required corner curvature was hidden
in an ordinary positive two-field bond square rather than a higher-order cube
mediator.

**Why it matters:** If that smaller action class worked, introducing a direct
fourth harmonic or a multi-site invariant would be unnecessary complexity.

**Symptoms:** `native_phase_basis.py` exhausts face and edge versions of the
positive squares `|Psi_j-Psi_i|^2`, `|chi_j-chi_i|^2`,
`|Psi_i^dagger Psi_j +/- chi_i^dagger chi_j|^2`, and
`|chi_i^dagger Psi_j +/- conjugate(Psi_i^dagger chi_j)|^2`, with nonnegative
coefficients and both global phase symmetries retained. Exact stationarity has
rank six. Its only normalized positive stationary ray is the `cross_plus`
family with the canonical C19 face:edge ratio `2:1`. That ray has the same
negative phase signature as the canonical area bond: its nonzero curvatures
are `{-32/9,-32/9,-8/9,-8/9,-8/9,-8/9}`. Adding it changes the canonical
negative curvatures from `{-16,-16,-4,-4,-4,-4}` to still more negative
values. The exact cutting-plane feasibility problem is therefore infeasible.

**Expected symptoms:** A successful smaller basis would supply a nonzero
positive stationary combination whose projected Hessian raises all six
negative corner-phase directions while leaving the two global phase modes
neutral.

**Root cause:** Every positive stationary quadratic two-field bond square in
this symmetry class reinforces the same first-harmonic phase ordering already
responsible for the saddle. The native C19 `2:1` geometry appears naturally,
but it does not change the harmonic response rank.

**Corrective action:** Retire this complete positive two-site quadratic phase
basis. Preserve the interface invariant separately because its successful
receiver-orientation Hessian is not a phase stabilizer. Move the corner gate
to genuinely multi-site cube/plaquette or dynamically eliminated complex-chi
actions capable of generating the independently derived fourth-order phase
response without inserting the target pattern.

## Edge-snap action response parity

**Issue:** The literal one-edge comparator identified a missing
motion-dependent response, but it did not determine whether that response is
gyroscopic (odd in velocity) or kinetic/geodesic (even in velocity). The old
gyroscopic probes also inherited `B=229` and a noncanonical diagnostic
potential, so they could not answer the V4 question directly.

**Why it matters:** Choosing the wrong velocity-action class leads to
coefficient sweeps that can never generate the required handoff. A candidate
must work with one cubic-symmetric coefficient set in both signs of all three
axes while leaving the same rest orbit intact.

**Symptoms:** `edge_snap_action_rank.py` locks `B=1197`, restores the canonical
position force, and tests the complete 30-column verified velocity-linear
one-form basis. The required second-order snap response is even under direction
reversal (`odd/even = 6.31e-7`), whereas the gyroscopic basis is odd
(`odd/even = 8.07e5`). Its unrestricted span explains only `1.3968%` of the
squared target norm; enforcing the exact-rest nullspace changes this only to
`1.3914%`. In contrast, eleven symmetric kinetic channels have the correct
parity and their unrestricted span explains `95.8521%`. Every individual
positive-square addition points the wrong way, so the nonnegative fit chooses
zero. The strongest single diagnostic is a negative C19 chi-gradient kinetic
coefficient `-0.96576`, which reduces the residual to `0.21302` but exceeds
the `-3/4` positivity boundary of the first-gradient metric.

**Expected symptoms:** A valid action has an even-in-velocity neighbor response,
one coefficient set for both directions, an exact or freely re-solved rest
orbit, and a positive invertible kinetic metric. Its discrete Euler--Lagrange
residual must then vanish after re-solving the phase and translation period.

**Root cause:** The handoff is a geodesic/kinetic-curvature problem, not a
magnetic/gyroscopic one. Reversing a traveling orbit reverses a velocity-linear
force, but it does not reverse the second spatial difference of the translated
carrier. The diagonal local chi inertia therefore supplies the wrong native
response class. The factorized cubic metric
`I-2 D19/24+(D19/24)^2=(I-D19/24)^2=(A19/24)^2` has the correct parity and
reduces the six-direction fixed-phase residual to `0.09891`; it follows only
from the C19 weighted degree `24`. Its unregularized symbol vanishes at
`D19=24`, however, so it makes part of the update undetermined rather than
proving a deterministic snap. Adding the existing `kappa=1/63` restores
invertibility but leaves residuals `0.10162` to `0.11004`.

**Corrective action:** Retire the velocity-linear gyroscopic family as the
primary edge-snap mechanism. Retain it only for later phase/stability effects.
Use the factorized C19 metric as a response oracle, not an admitted action.
Derive the smallest positive invertible *field-dependent* C19 kinetic square
whose connection force supplies the remaining receiver response, derive its
reciprocal implicit update, and re-solve rather than fit the movement phase and
period. Do not equate a singular low residual with a freely moving solution.

## Density-mold connection and relative-periodic snap

**Issue:** The positive C19-average kinetic metric reduced the literal snap
defect but left no field-dependent mechanism for the substrate to open ahead
of departing `Psi` and heal behind it.

**Why it matters:** A valid edge-to-edge carrier must move `Psi` and its live
`chi` mold together. Improving only the endpoint geometry or only the resting
spectrum does not establish a freely moving electron.

**Symptoms:** The positive reciprocal connection
`Vchi=chiDot+s rhoPsiDot chi/[2(rhoChi+kappa chi0^2)]`, inside
`B[kappa|chiDot|^2+|A19 Vchi/24|^2]/2`, reduces the six-direction fixed-snap
residual from `0.1068715` to `0.0870728` at the geometry-derived
`s=1+54/19^2`, an `18.53%` improvement. Its complete side-three implicit
two-layer spectrum is well posed and preserves the incumbent growth
`0.75166816348` to numerical precision. A two-update orbit search reduces its
residual by `86.4%` but does not close. A four-update unconstrained solve
reaches `8.86e-4` by inflating the intermediate state. Requiring the two
native discrete phase charges prevents that shortcut; the best matched run
has `0.008261` action residual, `0.9827%` maximum `Psi` charge defect, and
`0.2762%` maximum `chi` charge defect while its density center advances
monotonically toward the neighboring cube.

**Expected symptoms:** A passing relative-periodic branch makes every
discrete Euler--Lagrange residual and both discrete Noether-charge defects
vanish, returns to the identical eight-site carrier translated one edge, and
survives unrestricted evolution.

**Root cause:** The remaining residual is `96.09%` in `chi`; at the endpoint
layer `92.39%` of the `chi` residual lies in component three. The largest
single residual is at leading-shell site `(3,2,1)`. This is the substrate
normal/open-and-heal response, not a missing internal `Psi` frame rotation.
Replacing the local density rate by its C19 average, its Dirichlet difference,
or their symmetric average worsens the exact-snap response, so the local
density signal is the correct member of that scalar spatial family but is not
sufficient to close the orbit.

**Corrective action:** Preserve the local density-mold connection as the
current partial-improvement candidate but do not admit it. Project the saved
four-layer residual against the remaining positive reciprocal multi-site
orientation/plaquette action bases. Require nonzero response specifically in
the leading-shell third-`chi` channel, retain the exact rest orbit and charge
symmetries, and rerun the same four-update boundary-value gate. Do not force a
one-integrator-step hop or fit an action coefficient to Dirac velocity.

## Timestep refinement and bond-current material derivative

**Issue:** The smaller raw residual of the eight-update handoff and the
intuitive idea that `Psi` phase current should advect the `chi` mold both
appeared to offer a route to an exact edge recurrence.

**Why it matters:** Either effect could have converted the four-update failure
into a numerical-resolution problem or supplied the missing receiver response
without introducing an oriented spinor rule.

**Symptoms:** At fixed translation time `0.4`, the stored discrete-action
residual falls from `0.0033876663` at `dt=0.1` to `0.0016899276` at `dt=0.05`.
After restoring the required `dt^2` equation normalization, however, the
physical residual grows from `0.3387666` to `0.6759710`, a ratio of `1.99539`.
The refined residual is `91.75%` in `chi`; its largest entry is component three
at site `(2,1,2)` in the middle of the handoff. A positive C19 material
derivative constructed from
`jPsi=sum w n Im(Psi_i^dagger Psi_(i+n))/12` was tested with scalar advection
and both vector/covector strain conventions. Every variant is equal to or
worse than the density-tensor incumbent at the six-direction cheap gate.

**Expected symptoms:** A convergent branch would make the physical equation
residual decrease under timestep refinement. A useful material-current action
would lower the matched six-direction action residual before any expensive
orbit solve.

**Root cause:** The apparent refinement was the automatic `dt` scaling of the
discrete action, not improved satisfaction of the equations. Ordinary
advection follows a scalar phase-current velocity, while the remaining
response is an internal signed incidence channel. It cannot distinguish the
four forward/backward cubical components required by massive Dirac transport.

**Corrective action:** Retire scalar bond-current advection at the cheap gate
and do not run a sixteen-update continuation. Rebuild the movement comparator
from the already-derived cubical incidence channels before proposing another
action. Any later action candidate must generate that signed response from
native `Psi`--`chi` geometry rather than installing a shift or Dirac matrix.

## Rigid carrier shift versus cubical Dirac transport

**Issue:** The literal edge gate required the entire massive eight-corner
carrier to undergo the same one-site shift. That is a valid kinematic control,
but it is not the microscopic movement law of the Dirac equation.

**Why it matters:** Using the wrong target can reject a correct native action
or drive action fitting toward a scalar conveyor that a massive spinor should
not obey.

**Symptoms:** The exact cubical generators split every axis into rank-four
`+gamma` and rank-four `-gamma` edge channels. Inside each of the two flavor
copies, the split is rank two forward and rank two backward. The flavor
operator commutes with all three transport generators and with `beta`, so the
conditional shift has exactly zero copy leakage. For `m=1`, `p=0.6`, a
positive-energy state has channel weights `0.7572478777` and `0.2427521223`,
exactly `(1 +/- p/E)/2`. A symmetric conditional-edge/mass step converges to
the cubical Dirac evolution as `dt` is refined, while the rigid-shift generator
retains an approximately `0.566` mismatch. The rigid shift becomes exact only
for the massless single-chirality control.

**Expected symptoms:** The native localized readout must send its two signed
channels toward opposite neighboring edges, mix them locally through the mass
grading, and reproduce their net current. It need not translate every native
component rigidly in one update.

**Root cause:** The old test conflated motion of the carrier's measured center
with microscopic transport of its four spinor components. It also treated the
two flavor copies as a possible leader/follower pair, although the proven
algebra preserves each copy independently.

**Corrective action:** Preserve the rigid shift as a massless/chiral and
soliton control, not the massive-electron acceptance target. Freeze one
cubical readout at the localized rest carrier; measure the `+/-gamma` native
bond currents and `beta` mixing under the unrestricted two-field update; then
compare their recombination with the independent discrete Dirac evolution.
Do not add the conditional shift, flavor selection, or cubical projectors to
the action.

## Conditional-channel action rank and nonlinear chi-frame metric

**Issue:** Replacing the obsolete rigid microscopic comparator by the derived
forward/backward cubical channels could have invalidated the earlier action
response classification. The remaining movement defect also stayed dominated
by the third `chi` component, suggesting that the scalar kinetic metric was
transporting longitudinal and transverse substrate motion incorrectly.

**Why it matters:** If the old parity result were an artifact, the previously
retired gyroscopic family would need to be reopened. If a positive native
frame split could affect the residual while preserving rest, it would identify
a genuine action repair rather than a stronger launch.

**Symptoms:** `conditional_edge_action_rank.json` repeats the rank audit on
the zero-phase conditional channels. The target discrete Euler--Lagrange
response remains direction-even, while all thirty gyroscopic columns remain
direction-odd. They explain only `1.6721%` of the squared response. The
symmetric kinetic family explains `95.8711%` without sign restrictions, but
the nonnegative fit gives zero improvement. A positive frame metric

`Tchi=(B/2)[kappa|chiDot|^2+|A19(Pchi Vchi+s(I-Pchi)Vchi)/24|^2]`

reduces the fixed handoff residual from `0.08635783` to `0.02733465` at a
diagnostic constant `s=0.13555`, but that constant split raises the full
two-layer rest growth from `0.751654` to `1.099400` at `s=1/sqrt(54)`.

**Expected symptoms:** A valid repair must preserve the stabilizing rest
quadratic action while changing the finite moving response, close the
four-layer discrete equations, and preserve both native phase charges.

**Root cause:** The same transverse `chi` inertia that helps the fixed-center
linear spectrum obstructs finite density transfer. A constant frame split
cannot satisfy both gates. This is a nonlinear kinetic-metric problem, not a
missing direction-odd kick and not a timestep-resolution error.

**Corrective action:** Replace the constant split by the smooth native
transfer factor

`s(a)=(1+g a^2)^(-1/2)`,
`a=rhoPsiDot/[2 chi0 (rhoPsi+kappa chi0^2)]`.

It equals one through quadratic order at rest and softens only during actual
density exchange. The cubical face count `g=6` improves the cheap response by
`41.42%` and lies within `0.0084%` of the unrestricted diagnostic optimum.
The C19 count `g=19` improves it by `38.45%`. Both reproduce the incumbent
rest spectrum to `2.5e-14` in spectral radius. In the charge-constrained
four-layer solve, `g=19` lowers the action residual from the tensor
incumbent's `0.00338767` to `0.00239137` while reducing normalized charge
defects to `1.03e-5` and `1.94e-6`; `g=6` closes charges but saturates at
`0.00298683`. Retain the nonlinear family as a partial survivor, but do not
admit either coefficient until its exact normalization and the remaining
transverse-`chi` response are derived. The largest residual for `g=19` is
still `chi` component three at core site `(1,2,1)` on layer two; aggregate
residual power is `87.95% chi`.

## Literal edge transport through direct native field bonds

**Issue:** The current action transports the composite area
`W=chi wedge Psi`, but has no independent positive C19 Dirichlet channel for
either native field. The edge-snap hypothesis therefore had not been tested
as a direct reciprocal field-to-field bond on the actual optimized moving
residual.

**Why it matters:** A direct bond is the smallest action mechanism by which a
native field can respond only to neighboring lattice sites. If its positive
force points the wrong way, literal nearest-neighbor snapping is not repaired
by adding more ordinary stiffness. If it points the right way, it identifies
a genuine action knob without inserting a corner, copy, or Dirac label.

**Symptoms:** Positive direct `Psi` face and edge bonds correlate negatively
with the required four-layer response (`-0.1774` and `-0.1908`). Positive
direct `chi` bonds correlate positively (`0.3621` and `0.3608`) and together
explain `13.52%` of the squared saved defect in the unconstrained response
test. The exact geometry-derived trial

`Delta U=(B/6) sum_(i,n in C19) w_n |chi_(i+n)-chi_i|^2/(2*24)`

reduces the unchanged-trajectory discrete action residual by `5.41%`. After
the three intermediate layers are freely re-solved, the residual improves
from the incumbent `0.00239137` to `0.00223564` (`6.51%`), and the combined
movement-plus-charge objective improves by `5.63%`. The path still does not
close; normalized charge defects rise to `3.04e-4` for `Psi` and `5.20e-5`
for `chi`. The remaining equation residual is `86.64% chi`, with `54.74%` of
total power in the local `chi`-ray channel and `21.75%` in the `chi`
orientation-normal channel. Its largest entry remains layer two, site
`(1,2,1)`, `chi` component three.

**Expected symptoms:** An admitted edge mechanism must retain the same
localized rest carrier after a free rest re-solve, close every handoff layer
and both native charges, and then pass the full spectrum and repeated
translation gates.

**Root cause:** The direct `chi` bond supplies part of the missing receiving-
substrate response, so the old action is measurably underconnected in `chi`.
It does not supply the complete radial/orientation reconstruction. It also
acts nontrivially on the current rest texture: `65.58%` of its C19 force is
locally nonparallel to `chi`, so the old endpoint is not an exact rest orbit
of the trial action.

**Corrective action:** Continue the `B/6` action from zero coupling and freely
re-solve the complete rest state, phases, and halo. Retain it only if the
eight-site identity survives and the rest spectrum is not worsened. If that
gate passes, repeat the movement solve with the deformed endpoint and derive
the remaining radial/orientation response from the new residual. The current
result is a first improvement, not an admitted action or movement pass.

## Rest solvability of the direct `chi` bond

**Issue:** The movement-positive static `chi` bond had to be continued onto a
rest orbit of its own action before it could be admitted.

**Why it matters:** A movement correction that destroys the exact localized
rest state cannot be the common action behind resting and moving electrons.

**Symptoms:** Free LBFGS continuation retains `99.70%` carrier overlap but
stalls at normalized DEL RMS `3.75e-7`. The exact gauge-fixed Jacobian has
numerical rank `149/324`, not `322/324`; the direct-bond forcing has a `44.72%`
relative Fredholm obstruction. The obstruction is `83.21% chi`, and `92.14%`
of its `chi` power is component three. Ordinary local-potential, direct-`Psi`,
area-bond, and parallel-overlap partners remove only `0.74%` of its squared
rest obstruction and require nonphysical huge coefficients.

**Expected symptoms:** A smooth action continuation must place the added rest
force in the range of the baseline rest Jacobian while retaining the
eight-site carrier.

**Root cause:** A static direct `chi` spring supplies useful receiver force
but also excites a large degenerate rest cokernel. This is a structural
solvability failure, not a poor Newton step. It does not exclude a disconnected
finite-amplitude branch, but it retires smooth static-bond continuation as the
minimal repair.

**Corrective action:** Activate the `chi` neighbor stiffness with the native
`Psi` density-transfer rate so it vanishes exactly at rest and acts only during
a handoff. Derive it as a positive reciprocal kinetic action rather than
subtracting its rest force.

## Transfer-activated `chi` edge stiffness

**Issue:** The static bond's useful transport response had to be retained
without its rest-orbit obstruction.

**Why it matters:** This directly tests whether the substrate can become
edge-stiff only while the carrier transfers, which is the native version of a
literal nearest-neighbor snap without installing a shift rule.

**Symptoms:** The positive action

`Delta T=(B/(2*24)) sum_C19 w [(a_i^2+a_j^2)/2]|chi_j-chi_i|^2`,

`a=rhoPsiDot/[2 chi0 (rhoPsi+kappa chi0^2)]`,

has rest action `2.33e-26` and rest DEL-column norm `3.09e-11`. At the
undirected-bond coefficient `1/2`, its full two-layer rest growth changes from
`0.7516539941` to `0.7516538963`, so it does not worsen the incumbent. A
100-iteration unrestricted four-layer solve lowers the movement DEL residual
from `0.00239137` to `0.00232952` (`2.59%`) and the combined charge-aware
objective by about `1.67%`. The path still fails to close; charge defects are
`2.51e-4` and `1.98e-4`. Its residual is `89.86% chi`, with `54.74%` in the
`chi` ray and `25.80%` in the orientation-normal channel. The largest defect
remains layer two, site `(1,2,1)`, component three.

**Expected symptoms:** A passing action drives the complete four-layer DEL
and both phase-charge defects to numerical zero, then repeats the handoff
without runtime support.

**Root cause:** Transfer activation solves the static-rest incompatibility and
adds a real edge response, but a scalar positive bond does not reconstruct
the full radial and orientation state of the receiving `chi`. Splitting the
same construction into radial-magnitude and orientation-area bonds explains
only `0.34%` of the remaining squared defect; the positive orientation term
has the wrong sign. The existing complete multi-site/plaquette vocabulary can
span `27.27%` only with unrestricted, largely non-proven coefficients; its
positive symmetric kinetic subspace spans `1.27%`.

**Corrective action:** Retain the transfer-activated bond as the new partial
incumbent. Do not add elementary radial/orientation stiffness. Derive the
next candidate from a positive field-dependent multi-neighbor kinetic metric
whose finite-transfer connection carries independent radial and orientation
response, and require native coefficient provenance before another orbit
solve.

## Transfer-driven neighbor connection and physical-site lift

**Issue:** A positive C19 neighbor connection improved the free handoff, but
the conditional Dirac comparator used to define an edge snap had not been
proved to act on the physical corner sites of the localized carrier.

**Why it matters:** Action tuning against an internal fiber shift cannot prove
physical movement unless one embedding maps that fiber into the full native
`Psi`--`chi` phase space and intertwines translation on overlapping cubes.

**Symptoms:** Placing `chi0*a*(chi-A19 chi/24)` inside the positive incumbent
kinetic square selects a transverse coefficient `-1/38` within `0.32%` of the
independent linear-response optimum. Combining radial `+1/38` and transverse
`-1/38` as one native frame reflection lowers the freely re-solved DEL
residual from `0.00232952` to `0.00231328` and the combined charge-aware score
to `0.00232382`. Its side-three rest growth falls slightly, from
`0.7516538963` to `0.7516537336`; the carrier remains unstable and the orbit
does not close.

The lift audit then applies the existing internal conditional shift to the
literal physical corners. Relative to a physical one-edge translation, the
`Psi` defect is `84.34%` on x and `100%` on y/z; the shared-face defects are
`59.64%` and `70.71%`. Adding both flavor copies reconstructs exactly the same
conditional fiber shift (defect below `1.1e-16`) and does not repair the
physical-site mismatch. Each four-component copy independently has two
forward and two backward channels and zero one-axis walk index.

**Expected symptoms:** A valid movement target must be constructed from a
localized embedding `E` with `E^dagger E=I`, native sector closure, and an
overlapping-cube translation intertwiner. The same `E` must carry mass,
transport, density, and current.

**Root cause:** `LeanDirac/V2/NativeCubicalCompression.lean` explicitly states
that its eight coordinates form an internal cubical fiber and are not asserted
to be eight separate physical sites. The movement experiments provisionally
identified those coordinates with the eight localized corners. That missing
measurement/embedding bridge is upstream of the remaining action residual.
The two copies are balanced flavor sectors, not a leading and trailing pair.

**Corrective action:** Freeze further coefficient tuning. Derive the localized
phase-space embedding from native `Psi`, live `chi`, both committed layers,
and the surrounding shell; prove its overlap and translation consistency.
Only then rebuild the conditional edge endpoint and continue the action
test--RCA--repair loop. The small frame-reflection improvement remains
preserved as research evidence, not an admitted action.

## Overlapping-cube lift and native two-layer closure

**Issue:** The cubical Clifford fiber needed an honest lift into the localized
native fields before an edge snap could be treated as physical transport.

**Why it matters:** Applying an internal conditional shift directly to eight
physical corners produced an `84%`--`100%` mismatch. Without a common
embedding, action tuning used the wrong movement target.

**Symptoms:** A translation-equivariant overlapping-cube synthesis was built
from the carrier's eight native `Psi` rays and eight `chi` companions. `Psi`
alone and `chi` alone each have complex rank two on either flavor copy; the
joint native field has complex rank four at all `4923` sampled momenta. Its
smallest sampled singular value is `0.0825054`, and its translation
intertwiner defect is `6.73e-15`. The plus and minus flavor images coincide
to projector defects below `7.69e-15`; together they still have rank four,
not eight, under this spatial lift.

A configuration-only force test has about `30.38%` cross-block leakage, almost
entirely in the local `Psi`-ray channel. Solving the previous-layer graph from
the complete implicit native update reduces the full invariance defect to
`1.3440e-4`, `3.3637e-5`, and `8.4114e-6` at `dt=1/128`, `1/256`, and `1/512`.
The observed orders are `1.99847` and `1.99962`. Thus the full two-layer
defect, unlike the configuration-only diagnostic, converges quadratically to
zero. The native-selected `chi/Psi` graph scale approaches about `-0.84` for
the plus copy and about `-0.852` for the minus copy; no Dirac target entered
that selection.

**Expected symptoms:** A viable localized readout must have rank four in the
actual two native fields, commute with lattice translation, and have a native
two-layer leakage that vanishes under timestep refinement before its mass,
signed transport, density, and current are compared with Dirac.

**Root cause:** The earlier large lift failure came from identifying internal
fiber coordinates with literal sites and omitting the previous committed
layer. The four components are carried jointly by `Psi` and `chi`; their
first-order evolution is encoded in the relation between the two time layers.
The present evidence supports continuum tangent closure but not exact
finite-step closure or nonlinear carrier translation.

**Corrective action:** Retain this single overlapping-cube `E` as the frozen
measurement candidate. Derive its effective mass and the three signed
transport generators from the native update, then pull back the native
Noether density/current through the same map. Only a nonconvergent residual in
those tests becomes an action-repair target. Do not resume coefficient tuning
against the obsolete literal-corner shift.

## Two-copy carrier-contrast edge snap

**Issue:** The first overlapping-cube lift carried only the in-plane `chi`
companion and therefore could not represent physical center motion of the
localized carrier.

**Why it matters:** The candidate replaces ordinary vacuum
`chi=chi0 e3` by an in-plane companion on its eight occupied sites. A physical
handoff must restore the vacuum-normal component behind the carrier and remove
it ahead. Omitting that native direction makes every action look unable to
translate a state that the readout itself cannot represent.

**Symptoms:** Allowing independent in-plane `Psi` and `chi` amplitudes does
not change the nearest-center translation defect: it remains `71.9739%`, even
for the full eight-corner amplitude space. Adding only the existing third
component of complex `chi` reduces all three translation-tangent defects below
`1.01e-15`. The more reduced, coefficient-free profile using the exact
localized carrier minus ordinary `chi0 e3` vacuum gives the decisive copy
test: either four-component copy alone leaves a `66.6635%` worst defect, while
both copies together span all three physical translations to `1.40e-15`.

On the side-three periodic control, the joint envelope captures translation
to `1.62e-15`. Its leakage equation has rank `72` in a rank-`112` retained
space, leaving `40` previous-layer directions undetermined; `66.4%`--`71.3%`
of the translation coordinates lie in that nullspace. Selecting those
directions with the carrier's native rest-layer rotation gives full two-layer
residuals `1.4073e-2`, `3.6158e-3`, and `9.0754e-4` at `dt=1/128`, `1/256`,
and `1/512`. The observed orders are `1.9605` and `1.9943`; the retained
spectral radius simultaneously approaches one (`1.02121`, `1.00376`,
`1.00091`).

**Expected symptoms:** A valid snap readout must represent the same carrier's
translation in both signs of all axes, determine its current/previous-layer
relation from native dynamics, and leave only the declared centered-time
remainder before any nonlinear handoff is attempted.

**Root cause:** The failed literal-corner and companion-only lifts discarded
the vacuum-normal part of live `chi`, which is the part that erases and
rebuilds the substrate mold. They also inspected position without the
previous committed layer that selects direction. Once the exact carrier
contrast is used, the two algebraic copies are complementary for physical
translation even though they coincided under the incomplete in-plane spatial
lift. The remaining finite-step residual scales as the centered-time
discretization rather than plateauing as a missing spatial force.

**Corrective action:** Freeze the exact two-copy carrier-minus-vacuum contrast
as the movement embedding. Derive the full action-dependent phase density and
C19 bond current through this same graph, then repeat the response with a
matrix-free side-five box and solve a free nonlinear relative-periodic
handoff. Do not alter the action unless a nonconvergent current or large-box
cross block identifies a missing native response. Rest spectral stability
and repeated finite translation remain open.

## Fixed-charge edge transport has only two native directions

**Issue:** The exact carrier-contrast lift spans three configuration-space
translations, but it was not known whether the unchanged action transports
the conserved phase charge with that shape in all three directions.

**Why it matters:** Redrawing the eight-site profile in a neighboring cube is
not electron motion unless the same native charge, four-component identity,
and substrate state cross the boundary. A free Dirac sector requires three
independent signed transport generators.

**Symptoms:** Solving the unrestricted previous layer makes a prescribed
one-edge profile displacement satisfy the linearized discrete
Euler--Lagrange equation to `8.04e-16`; more than `99.99999999%` of the launch
power is inside the first shell. The Noether audit nevertheless shows that
the shape displacement and phase-charge center disagree. Local common phases
and local relative-copy phases do not repair it. Odd neighboring-center phase
ramps generate clean signed x/y responses, with the two copies reversing the
response, but the exact fixed-charge center matrix has singular values near
`{9.7e-6,6.9e-6,1e-15}` and rank two.

The omission is not a restricted launch artifact. Adding independent
real/imaginary normal `chi` motion leaves the third singular value at noise.
Exhausting all `96` real/imaginary `Psi`-ray, `Psi`-companion, `Psi`-normal,
`chi`-ray, `chi`-companion, and `chi`-normal ramps in both copies still gives
rank two on every spatial ramp axis. The missing physical output is z to
about `1e-10` angular accuracy. The common conserved charge is also rank two;
the apparent rank-three `chi` center is ill-conditioned because its baseline
charge is essentially zero.

Several admissible action classes were screened against the same exact test.
The positive parallel-overlap bond energy, local and reciprocal-bond overlap
symplectic forms, and positive square-face factorization energy all preserve
the exact rest orbit but do not raise the rank. The plaquette result remains
rank two even at geometric screen strengths `chi0^2` and `chi0^4`. A positive
C19 area-gradient kinetic square changes the momentum strongly but also
remains rank two and destroys the old rest orbit at unit strength.

**Expected symptoms:** A passing carrier/action pair must have three robust
fixed-charge singular directions under the same two-layer native update, with
the weakest singular value stable under timestep and box refinement. The
corresponding nonlinear state must then cross two edges in both signs of all
three axes without a prepared receiver.

**Root cause:** The prepared carrier is globally coplanar in internal
`C^3`: every `Psi` and `chi` ray lies in the same complex two-plane, so every
`W=chi wedge Psi` is parallel to one field-space normal. At rest, neighboring
`W` overlaps are purely real across x/y faces but purely imaginary across z
faces. The ordinary bond action therefore has linear phase curvature in two
directions while the z bond sits at a quarter-turn with zero linear curvature.
This collapses the physically meaningful Noether-center tangent to rank two.
The abstract cubical Clifford algebra still has three generators; the loss
occurs in its present coplanar native lift.

**Corrective action:** Stop scalar coefficient sweeps and do not install the
forbidden direct fourth-harmonic phase oracle. Derive the smallest
non-coplanar `C^3` carrier deformation or reciprocal native frame connection
for which all three `W` bond-response curvatures are nonzero, while preserving
the eight-site corner identity and solving the unrestricted rest equations.
Then repeat the exact fixed-charge rank test before any nonlinear movement
run. If no such freely solved branch exists, the next admissible class is a
field-dependent multi-neighbor kinetic connection whose full momentum map,
Hamiltonian, and rest spectrum are derived together.

## Tetrahedral orientation and joint-frame connection

**Issue:** The prepared signed corner rays contain a three-dimensional
orientation geometry that the coplanar `W=chi wedge Psi` variable does not
retain. It was therefore necessary to determine whether a basis-free native
frame connection could recover the missing fixed-charge direction without a
Dirac matrix or programmed shift.

**Why it matters:** A successful connection would explain literal edge
transport using only the two native fields and C19 relationships. A failed
complete connection basis prevents further blind coefficient tuning.

**Symptoms:** `carrier_hopf_geometry_probe.json` evaluates each normalized
corner ray through its rank-one Hermitian projector `P=z z^dagger`, using the
basis-free orientation product `2 Tr(P_i P_j)-1`. The eight signed rays are a
two-to-one lift of four orientations. They span three real dimensions, sum to
zero, and have frame tensor `(8/3)I`. Every one of the 12 cube-face pairs and
12 cube-edge pairs has product `-1/3` to `2.78e-16`; every one of the four
excluded body-diagonal pairs has product `+1`. Thus the prepared rays already
carry exact tetrahedral geometry matched to C19.

The positive midpoint kinetic candidate uses the joint native tensor
`Q=(Psi Psi^dagger+chi chi^dagger)/chi0^2` and the anti-Hermitian connection
`A_i=(1/24) sum_C19 w[Q_i,Q_j]`, replacing the local velocities by
`PsiDot+g A Psi` and `chiDot+g A chi`. It contains no corner, axis, copy,
Pauli/Dirac matrix, or target motion. The connection norm is below
`1.06e-33` on the exact rest intervals, its interval action delta is zero,
and the complete rest DEL RMS remains `1.895e-12`.

Nevertheless, the fixed-charge response has rank two for all three phase-ramp
axes at screen strengths `g=1`, `19`, and `361`. The third singular value is
only `3.75e-10`--`2.12e-9` relative to the first and does not become robust
under either geometry scale increase.

**Expected symptoms:** A valid native frame completion must preserve the rest
orbit while giving three robust fixed-charge singular directions whose weakest
value remains nonzero under coefficient and timestep refinement.

**Root cause:** The tetrahedral geometry is present in the rays, but the tested
connection is built from a C19 sum of joint commutators. That sum vanishes by
the exact tight-frame balance at rest, and its linear variation acts within
the same two-dimensional charge-current image as the incumbent action. Scaling
cannot change its response rank. The failure is therefore structural for this
isotropic one-site commutator connection, not a weak coefficient.

**Corrective action:** Retire the scalar-strength joint-commutator connection.
Do not install the Hopf/Bloch coordinates as a new field or force. Next test
whether the unrestricted rest equations admit a non-coplanar `C^3` deformation
that preserves the eight-site identity. If they do not, enumerate the
lowest-degree reciprocal C19 common-neighbor/triangle kinetic connections
whose oriented loop response is nonzero despite the tight-frame cancellation,
derive their full momenta and Hamiltonians, and apply the same rank gate before
any nonlinear handoff run.

## Body-diagonal orientation closes rank, but marginal modes remain

**Issue:** Separate C27 extensions of `Psi` or `chi` did not repair the native
three-direction transport map. A joint body-diagonal action on
`W=chi wedge Psi` did close the missing response, but its positive potential
alone torqued the exact localized rest orbit. The completed action now
preserves rest and rank-three transport, while its 100-cycle quotient-stability
gate remains open.

**Why it matters:** This is the first action class in the program that gives
the same localized two-field carrier three robust fixed-charge transport
directions. It cannot be admitted unless the exact rest orbit, full spectrum,
nonlinear perturbation gate, and repeated movement all survive together.

**Symptoms:** A direct body-diagonal `chi` energy remains rank two in every
axis. A direct `Psi` energy is rank three in only one axis and breaks rest.
The joint positive Dirichlet energy of normalized `W` is rank three in all
axes, with relative third singular values `{0.2735,0.2570,0.4406}`, but raises
the rest DEL RMS from about `1.9e-12` to `0.0196089`. A full-box Fredholm solve
removes only `87.6%` of that first-order torque; the remaining obstruction is
equally split between `Psi` and `chi`, components zero and one, and all eight
carrier sites.

The complete reduced local/body common/relative `W`-linear gyroscopic basis
has force rank two. Its minimum-norm coefficients cancel the body potential's
rest torque to relative residual `8.34e-14`, restore the DEL RMS to
`1.895e-12`, and preserve rank `[3,3,3]`. The four coefficients reduce exactly
to two sums and two rest-force null directions. Native inertia and cube
incidence select the reduced discrete coefficients
`(a+B/4,a-B/4,-a/2,0)`, where `a=2 dt/sin(theta)` and `a -> 2/chi0`.
The resulting action has rest DEL RMS `1.897e-12` and relative third transport
singular values `{0.04539,0.04521,0.09051}`.

The side-three full two-layer eigenproblem has no modulus above `1+1e-7`.
The apparent maximum excess, about `5e-8`, is constant under rates
`128,256,512,1024`; its leading eigenvalue has condition number `5.68e7` and
right residual `4.86e-15`, identifying unresolved numerical splitting at the
highly degenerate unit eigenvalue rather than a resolved finite Floquet
exponent. Direct 100-cycle linear propagation nevertheless has secular shape
residual up to `6443` under generic velocity controls after only phase and
translation alignment. This is over `4.7e8` better than the incumbent's
worst shape growth. The exact generalized unit-multiplier quotient reduces
that residual to `33`--`61`; it remains bounded rather than growing
exponentially (`49` at one cycle, `35` at 100, and `37` at 1000 for the
spectrally justified tolerance). At that tolerance `90%`--`99%` of the raw
remainder is exterior `chi` orientation. Native density/orientation
observables are also bounded but have transient amplification up to about
`55`, so this is not yet the declared perturbation pass.

**Expected symptoms:** A passing action must retain the exact rest residual
and rank-three transport while carrier-shape perturbations remain bounded for
100 cycles after every independently derived collective phase, charge,
frequency, and translation tangent is removed. The unrestricted nonlinear
implicit update must then preserve the same pattern and complete repeated
translations in both signs of all axes.

**Root cause:** C19 omits the four body-diagonal orientation identifications
that close the third response of this coplanar carrier. Adding them as a
potential supplies the missing geometry but creates an in-plane reciprocal
rest torque. The common/relative gyro completion cancels that torque, and the
otherwise-undetermined null direction couples naturally to the native `chi`
inertia `B`. The remaining linear growth is concentrated at a degenerate unit
multiplier and contains phase/frequency and translation generalized modes;
the present quotient includes phase and position tangents but not the full
charge/frequency family. Exterior vacuum sites also have an exact local
orientation degeneracy when `Psi=0`: rotating `chi` at fixed magnitude changes
neither density nor `W`. A direct positive `chi` C19 spring would lift that
degeneracy, but its carrier rest torque lies `87.3%` outside the complete
tested W-gyro response span. Installing it would therefore break the electron
rather than repair this candidate.

**Corrective action:** Freeze the native-normalized action only as the current
research incumbent. Derive the nearby charge/frequency rest family and its
physical perturbation normalization, then repeat the 100-cycle observable
quotient and derive the implicit reversible nonlinear update. Do not add the
failed direct `chi` C19 spring. If a noncollective carrier residual survives,
decompose its first field/site/component and extend only the rest-force-null
gyroscopic response that overlaps it. If the quotient closes, solve the freely
moving relative-periodic branch and compare its native bond current with the
frozen four-component Dirac readout. C27 remains an ablation and no canonical
promotion follows.

## Nonlinear clock test: raw growth is principally phase, not shape

**Issue:** The C27/body-gyro candidate shows very large raw phase-space
amplification under a chi-clock perturbation, and a rigid finite-frequency
continuation is obstructed beyond its first-order tangent. It was unclear
whether this was physical carrier destruction or neutral clock shear measured
against the wrong fixed phase.

**Why it matters:** Retiring the first exact-rest, rank-three transport
candidate on a global-phase diagnostic would repeat the measurement error that
the Dirac program was designed to avoid. Conversely, bounded density with an
unbounded phase-quotient shape defect would still fail the electron gate.

**Symptoms:** The full Fredholm tangent requires linked clocks with
`dthetaChi/dthetaPsi=-0.9986217652` and closes to relative range residual
`6.36e-10`. Finite continuation is quadratic at first, but a rigid
single-frequency family develops a residual near `5.32e-8`; about `82%` is
chi, `78%` of that lies outside the eight carrier sites, and `79%` is
transverse orientation. Direct chi-bond gyros, cross-bond phase gyros,
collective C19 covariance terms, and an exact-rest vacuum-mold alignment term
remove less than one percent of this residual and are rejected for this RCA.

The unrestricted implicit discrete Euler--Lagrange update was then run with no
held sites or projection. A 1% clock perturbation leaves the local solver's
nearby branch after roughly `0.4`--`0.5` cycle. A matched 0.1% perturbation
completes one full cycle in both signs. Its raw phase-space distance amplifies
by about `170`, but after removing only the two exact global Psi/chi phase
symmetries the maximum relative shape residual is `0.00153151`. Exterior Psi
density remains below `4.5e-10` RMS and carrier density changes remain below
one percent over that cycle.

**Expected symptoms:** A passing perturbation must retain small phase-quotient
shape, density, area-orientation, energy, and Noether-current defects for at
least 100 cycles, with convergence under timestep and box refinement. Raw
phase separation from a clock-perturbed reference is not itself a failure.

**Root cause:** The previously reported `170x` amplification is dominated by
the symmetry-related Psi phase/frequency Jordan channel, not immediate loss of
the eight-site carrier. The remaining measurable problem is a much smaller
but nonzero shape drift of about `0.153%` after one cycle. The current evidence
does not yet distinguish a bounded quasiperiodic deformation from secular
shape drift; the rigid relative-equilibrium ansatz is too restrictive to do
so.

**Corrective action:** Stop broad coefficient and stencil scans. Keep the
C27/body-gyro action as an unadmitted transport oracle. Extend the nonlinear
run with the frozen phase-quotient measurement and native energy/Noether
diagnostics at matched smaller amplitude, timestep, and box. If the quotient
shape remains bounded, proceed to the moving relative-periodic branch. If it
grows linearly after phase removal, derive the missing semisimple
phase/frequency response from the exact quotient residual; do not tune direct
chi stiffness or the rejected one-neighbor bases.

## Bounded completion audit: triangle transport closes, stable motion does not

**Issue:** The joint body-orientation oracle closes the missing third transport
response, but it has not produced a stable freely translating localized
carrier. The user-requested bounded search has now reached ten consecutive
mathematically distinct completion bases with zero improvement at their
declared response or full-spectrum gates.

**Why it matters:** Rank-three infinitesimal transport is not electron motion.
The V4 goal requires the same exact rest carrier to survive perturbations,
translate twice in both signs of all axes, and carry the same native/Dirac
current. Continuing to scan coefficients inside a response-deficient class
would create more artifacts without advancing that gate.

**Symptoms:** The apparent C27 body term and all four of its gyro one-forms
factor exactly, in value and first variation, through averages of two-link C19
face--edge triangles. Errors on exact-rest and random full fields are below
`9e-17`. The resulting C19 action retains rest DEL RMS `2.34e-12` and linear
fixed-charge transport rank `[3,3,3]`. Under unrestricted nonlinear evolution,
a `0.1%` clock perturbation reaches three cycles with phase-aligned shape
residual `0.00246368`; a `1%` perturbation leaves the locally solved branch
within about half a cycle. No free translated orbit or Dirac-current match was
found.

The exact C19 chi bond audit gives normalized rest overlaps
`{0,1/3,1}`. A positive parallel/orthogonal bond potential fails to preserve
the additional `1/3` carrier bonds. The geometry-complete nonnegative
polynomial `s(3s-p)^2(p-s)/chi0^12` does preserve rest, but creates four
resolved unstable multipliers and raises the growth rate to `0.007259`.
A C19 neighbor chi Berry term at native clock magnitude `B chi0` raises growth
to `1.31`--`1.48`. The remaining local chi Berry, C19-gradient chi Berry, and
positive C19-gradient kinetic bases also preserve rest but raise growth to
`1.21`--`1.38`. None improves the numerical-floor incumbent spectrum.

The ten zero-improvement bases, with machine-readable evidence, are recorded
in `c19_triangle_candidate_bounded_verdict.json`: direct chi bond gyros,
cross-field bond gyros, shared-neighbor orientation, vacuum-mold alignment,
parallel/orthogonal chi potential, cube-overlap chi potential, neighbor chi
Berry, local chi Berry, gradient chi Berry, and positive gradient chi kinetic
metric.

**Expected symptoms:** An admissible completion must preserve the exact rest
orbit, retain rank three, remove rather than split the exterior/interface chi
unit modes, and expose a signed nonlinear translation branch before any long
Dirac comparison is attempted.

**Root cause:** `W=chi wedge Psi` is blind to exterior chi orientation wherever
exterior Psi vanishes. Position stiffness in the missing channel has the wrong
Hamiltonian/Krein signature: even when engineered to preserve all exact cube
overlaps, it splits unit-circle modes outward. The complete reduced chi-only
Berry and kinetic basis does the same. The remaining obstruction is therefore
not insufficient scalar stiffness; it is a missing *joint* field-dependent
Psi--chi symplectic response at the carrier--vacuum interface, or else the
assumption that the physical rest branch remains coplanar.

**Corrective action:** Retire this ten-basis completion ladder and stop tuning
it. The smallest untested class is a genuinely joint, field-dependent
Psi--chi kinetic connection whose curvature acts only when the carrier and
vacuum exchange roles. Before adding it, first solve for a noncoplanar native
rest deformation; if such a branch exists, re-evaluate translation without a
new action. This is a scoped `ACTION_BASIS_NO_GO`, not a universal no-go for
local reciprocal two-field actions. Canonical V40 remains unchanged.
# C19 Bargmann-loop connection (2026-11-23)

**Issue:** A native C19 Bargmann triangle can preorient receiving `chi`, and
its reversal-invariant joint kinetic connection preserves the exact rest
orbit and produces a numerical rank-three center response, but the response
does not close on the retained two-copy eight-component tangent.

**Why it matters:** V4-006 requires the same localized four-component carrier
to move.  Rank-three motion in complementary or exterior modes is not a
Dirac transport proof and cannot be projected back at runtime.

**Symptoms:** The positive loop potential has rest DEL RMS `2.9846e-2` and
rank two.  The kinetic connection has rest DEL norm `3.17e-13` and rank three
at the screen strength, but `99.45%`--`99.97%` of its Hessian image is outside
the tested eight-component tangent and its cross-copy blocks are nonzero.

**Expected symptoms:** A passing native response would preserve the exact
rest orbit, have nonzero signed response in all three axes, remain closed on
the same four-component embedding including its required shell coordinates,
and have a zero opposite-copy block before any nonlinear movement test.

**Root cause:** The scalar Bargmann loop sees the sender phase/orientation
needed to turn `chi`, but it discards the cubical flavor grading.  Its full
variation therefore redirects most of the response into complementary and
exterior degrees of freedom instead of the geometry-derived Clifford fiber.

**Corrective action:** Do not tune the coefficient or project the response.
Derive the full localized embedding including first-shell `Psi`, `chi`, and
both time layers, then enumerate the lowest-degree joint C19 connections whose
Hessian commutes with the native flavor involution as a consequence of cubic
geometry.  Test exact sector closure and the three compressed generators
before solving a moving orbit.  If no such invariant basis exists, record a
scoped action-basis no-go rather than modifying V4 claims.

## Canonical translation-zero-mode audit (2026-11-23)

**Issue:** The spatial overlapping-cube embedding has rank four and spans the
three configuration translations, but the same directions are not neutral
directions of the canonical three-layer Euler--Lagrange update.

**Why it matters:** V4-006 needs a dynamical carrier, not merely a spatial
coordinate map or a launch that can redraw the next cube.  If translation is
not preserved by the native update, the localized four-component readout does
not intertwine the canonical action with the moving Dirac operator.

**Symptoms:** At update rates `128, 256, 512, 1024`, the co-rotating
translation defect decreases exactly as `dt^2`, while defect divided by
`dt^2` converges to `0.618865`, `0.775402`, and `2.256808` for x, y, and z.
The combined continuum force obstruction converges to `1.434933`.  More than
`99.9%` of its mass-normalized power is in `Psi`; the C19 area term is the
dominant aligned contribution.  Refitting only the local and area terms leaves
a `21.36%` residual and asks for a negative area coefficient.

The positive spectral square `|| (D19-27) Psi ||^2` preserves the exact rest
carrier but explains only `0.1549%` of the obstruction and asks for negative
strength.  The complete declared positive-square position basis leaves
`95.47%` of the obstruction.  An unconstrained position fit is algebraically
exact only by almost cancelling the canonical area bond and using a negative,
unbounded onsite coefficient.  The reversal-safe joint Bargmann connection
also preserves rest, but its response on the simultaneous three-layer
translation is essentially zero: it explains only `5.27e-6` of the squared
obstruction and requires an unproved coefficient near `-2.18e11`.

**Expected symptoms:** A passing action must make the translated tangent close
under the same three-layer DEL equation in all axes, retain the exact rest
orbit, and do so with a bounded reciprocal action whose coefficients have
native provenance.  The resulting four-component compression must then give
the mass and all three transport generators from the same embedding.

**Root cause:** The previously derived spatial map is not a dynamical
intertwiner.  Discrete C19 translation symmetry relates whole lattice-cell
copies but does not provide a continuous infinitesimal translation symmetry
for the sharply localized eight-site carrier.  The canonical area energy
therefore pins the carrier, most strongly in z.  The earlier Bargmann
rank-three result measured controllability of an inverse launch in the larger
native space; it did not cancel this pinning or close on the localized
four-component tangent.

**Corrective action:** Do not promote V4-006 and do not install the rejected
spectral or Bargmann terms.  The next admissible mathematical branch is a
relative-periodic lattice translation (a finite cell shift after a solved
period), which does not require a continuous zero mode, together with an exact
Peierls--Nabarro barrier calculation.  If that branch is absent, enumerate a
larger bounded joint `Psi`--`chi` action basis and require it to remove the
finite barrier rather than fitting a Dirac trajectory.  Canonical V40 remains
unchanged and the current result is `ACTION_HESSIAN_DEFICIENT` for the tested
continuous-translation realization, not a universal lattice no-go.

## Finite one-cell relative-periodic shooting audit (2026-11-23)

**Issue:** Removing the infinitesimal-zero-mode requirement did not produce a
freely repeating one-cell translation of the same localized carrier.

**Why it matters:** A discrete lattice may support a finite traveling orbit
without continuous translation symmetry.  This was the remaining way the
canonical action could have transported the carrier despite the tangent
pinning obstruction.

**Symptoms:** Canonical centered-update shooting was performed with `2, 4, 8,
16, 32` updates per cell.  The timestep and global `Psi`/`chi` phases were
solved, and no field was reset during evolution.  The relative two-layer
boundary residual improved from `0.9154` to `0.1814`, but did not converge.
The second unreset cell missed by `1.50` or more for the longer candidates.
The solved cell-crossing time approached about `0.46` rather than one update.

Allowing both initial layers to deform at fixed total `Psi` charge reduced the
first boundary residual to `0.0873`.  That solution was not the electron:
eight-site `Psi` fidelity fell to `0.4838`, core containment to `0.5738`, and
the second-cell residual rose to `2.4652`.  The canonical charge-normalized
interpolation has no positive Peierls--Nabarro barrier; instead it contains an
intermediate potential trough about `37448.84` below the localized endpoints.

**Expected symptoms:** A pass would drive both layer-boundary residuals and
the unreset second-cell residual to the declared tolerance while retaining
the eight-site pattern, `Psi` containment, the `chi` track, charge, and energy.

**Root cause:** The canonical action can move and redistribute field energy,
but the exact localized rest carrier is not the stable member of a traveling
family.  Along the one-cell path the action energetically favors a spread,
deformed intermediate configuration.  Additional time resolution lowers the
shooting error but does not make the eight-site state reconstruct; permitting
deformation lowers the residual by abandoning the carrier.  Thus the failure
is action-level binding/translation compatibility, not a one-tick timing
assumption and not merely a missing initial velocity.

**Corrective action:** Mark the current canonical-action finite-orbit search
`INCONCLUSIVE` rather than a universal no-go, but do not spend more iterations
on the same seeds.  V4-006 remains open.  The next repair must change the
research action so that the localized rest orbit and its translated family
have a flat or dynamically traversable constrained-energy path while
preserving the already-proved rest equations.  Derive that response from the
measured constrained-energy trough and the first pattern-loss mode; do not
install a translated target, Dirac velocity, or holding projection.

## Canonical noncoplanar chi-branch audit (2026-11-23)

**Issue:** The freely solved noncoplanar branch suggested by the triangle
research action had not been separated from canonical V40, refined to an
ordinary-vacuum box, or tested against the missing fixed-charge transport
channel.

**Why it matters:** A native complex-chi orientation branch could provide the
directional memory missing from the coplanar carrier without adding an action
term.  If it also supplied the third transport response, it would be the
smallest canonical route toward V4-006.

**Symptoms:** The old triangle/gyro branch is noncanonical, violates the
locked opposite-corner/chi-pair identities, and does not improve its complete
spectrum.  Repeating the calculation with canonical V40 found 12 transverse
side-three quotient directions, four after enforcing the opposite-pair
tangent constraints.  A finite side-three rest family was solved, but direct
transfer into side-four/five vacuum left residuals proportional to amplitude,
showing that this particular family was a small-period texture.

The complete side-five canonical Hessian nevertheless contains one genuine
pair-preserving transverse quotient direction.  Range correction closes its
nonlinear orbit residual to `1.36e-14` at `0.1%` and `3.98e-15` at `0.3%`.
The Psi field and all four opposite-corner relations remain unchanged to
roundoff; the deformation is a complex-chi orientation branch.  Despite this,
the exact canonical Noether-center audit remains fixed-charge rank `[2,2,2]`
at branch fractions `0`, `0.01%`, `0.1%`, and `0.3%`.  The weakest relative
singular value stays between `1.36e-12` and `4.59e-12`.  The side-three branch
also only slightly reduces an already unstable spectral radius
(`1.0040551` to `1.0040448`) and never becomes linearly stable.

**Expected symptoms:** A passing branch must preserve the same eight-site Psi
identity and produce three nonzero fixed-charge carrier-center responses from
the same canonical action and Noether density, before finite translation or a
Dirac-current comparison is attempted.

**Root cause:** Canonical V40 has additional internal chi orientation freedom,
but that freedom is dynamically orthogonal to the missing translational
current channel.  At the tested `0.3%` branch point, Psi changes by only
`2.74e-14`, chi changes by `1.73%`, every chi norm is unchanged to roundoff,
and every normalized `|chi^dagger Psi|` remains below `5e-15`.  Thus chi is
rotating inside the complex subspace orthogonal to Psi; it does not open a
parallel overlap or charge-transfer channel.  It changes the substrate
orientation while leaving the Psi carrier and the rank-two transport image
unchanged.  The earlier triangle
clue therefore identified a real chi degeneracy but not the absent odd
translation response.

**Corrective action:** Preserve the canonical chi branch as a discovered
native rest family, but do not promote it as V4-006 or use it for a cell-orbit
shoot.  The next action research must target the measured missing odd
fixed-charge response at the carrier--vacuum interface while remaining zero
on both the original rest orbit and this chi family.  Reuse the exact rank-two
response matrix as the inverse requirement; admit only a local reciprocal
bounded action with coefficient provenance, then re-run rest identity,
spectrum, three-axis rank, finite translation, and Dirac-current gates.

## Canonical rotor-wake and transfer-kinetic audit (2026-11-23)

**Issue:** The rotating carrier does not prepare the adjacent cube through the
canonical C19 area term, and the first two minimal overlap one-forms do not
restore the missing fixed-charge transport response.

**Why it matters:** The proposed physical mechanism is that the internal
rotation should stir complex chi around the carrier, so the neighboring cube
is oriented before Psi arrives.  V4-006 cannot close unless that wake carries
an odd, directional current while the same resting electron remains a native
orbit.

**Symptoms:** On the exact carrier, every incoming
`W=chi wedge Psi` vector is parallel to the receiving vacuum chi.  Each of the
three occupied C19 bonds at every adjacent face site therefore gives exactly
zero receiving-Psi force, and the incoming native phase current is below
`2.5e-32`.  On the exact 0.3% noncoplanar chi branch the individual forces are
nonzero only at about `1e-13` and cancel to a maximum net `2.57e-15`.  A common
temporal phase preserves this geometry and cannot create a spatial phase lag.

Canonical rebuilds of the local and symmetric-bond overlap symplectic
one-forms preserve the rest DEL at roundoff but leave transport rank
`[2,2,2]`; their weakest relative singular values remain below `1e-11`.
Positive reciprocal transfer-overlap kinetic metrics were then tested.  Both
the bond-even and bond-odd metrics produce robust rank `[3,3,3]`; at unit
screen strength the weakest relative channels are `0.137` and `0.186`.  The
pure `D_t W` gradient metric remains rank two.  At bond-odd strength `1e-4`,
the weakest relative transport channel is still `0.131` and the previous-layer
block condition number is `1197.00077`, but the undeformed rest DEL is
`1.38e-3` RMS.

**Expected symptoms:** An admissible repair must preserve or freely re-solve
the resting carrier, retain a nonsingular reciprocal time update, provide
three fixed-charge responses in every direction, and then pass the same-state
current, stability, and finite-translation gates without a receiver preload.

**Root cause:** The canonical spatial action transports only the antisymmetric
area variable `W`.  The eight-site carrier collapses its in-plane Psi/chi
orientation into a W direction parallel to exterior chi, so the receiver has
no native lever arm with which to reconstruct Psi.  This is why a temporal
rotor is not automatically a spatial wake.  The successful response basis
shows that the missing mathematical object is a reciprocal kinetic transfer
of the joint sender/receiver Psi--chi orientation, not more scalar stiffness
and not a larger W Laplacian.

**Corrective action:** Do not admit either transfer metric yet.  The corrected
DEL Fredholm condition has no first-order obstruction (`7.1e-12` relative),
but the nonlinear fixed-frequency continuation has only reduced the residual
to `5.75e-4` RMS while deforming the state by `1.43%`; it has not solved an
exact orbit.  Continue with a joint state/frequency continuation and an
action-basis coefficient derivation.  Retain the bond-odd metric as the
leading response basis because it supplies the strongest missing channel.
Only after an exact orbit exists may spectrum, Noether current, two-cell
translation, and discrete-Dirac comparisons be run.  Canonical V40 remains
unchanged and V4-006 remains open.

The subsequent joint state/frequency solve fixed the action-derived Psi U(1)
charge rather than a raw amplitude.  It did not close: the combined objective
decreased, but the final field residual was `8.02e-4`, relative charge error
`1.72e-3`, and state deformation `2.05%`.  The failure is therefore not cured
by retiming alone.  The coefficient/state branch remains unresolved.

The separate continuum gate passes.  Around uniform vacuum the transfer-wake
action scales as perturbation amplitude `epsilon^3.9999979` and has exactly
zero tested Hessian-vector response.  It therefore does not change the
quadratic symbol used by the existing free-Dirac factorization.  The C19
symbol approaches `6|k|^2` with remainder powers `1.9974--1.9983` along axis,
face-diagonal, and body-diagonal directions.  This preserves the continuum
Dirac sector at linear order, but it does not rescue the unsolved localized
branch.

## Transfer-oscillator action nullspace (2026-11-23)

**Issue:** The kinetic transfer wake supplied the missing rank-three response
but changed the exact rest equations, so it could not be admitted by itself.

**Why it matters:** V4-006 needs one native action that supports both the same
resting eight-site carrier and three-direction motion.  Repairing transport by
destroying the rest electron would merely exchange one failure for another.

**Symptoms:** The kinetic-only odd transfer metric has rest-response norm
`534.518382572895`.  Its matching positive transfer-coordinate potential has
the same norm, `534.518382572897`, with normalized response correlation
`-0.9999999999999996`.  The two-column action-response matrix has singular
values `1.4142135623730947` and `9.20e-15`, with null vector
`(0.7071067811865479, 0.7071067811865472)`.

Installing that null combination as a discrete harmonic action for
`Z_ij=chi_i^dagger(chi_j wedge Psi_j)-chi_j^dagger(chi_i wedge Psi_i)`
restores the original carrier DEL to `1.84e-13` RMS at unit normalized
strength and `1.86e-14` at strength `1e-4`.  It retains fixed-charge rank
`[3,3,3]`; the weakest relative transport singular value is `0.1915` at unit
strength and `0.1342` at `1e-4`.  The discrete Legendre block remains
nonsingular with condition number about `1205`.

The completed action also passes the uniform-vacuum continuum gate.  Its
perturbation scaling is `epsilon^3.9999963`, its tested quadratic
Hessian-vector is exactly zero, and the C19 symbol keeps its isotropic
`6|k|^2+O(|k|^4)` limit.  Thus it leaves the existing continuum free-Dirac
factorization unchanged at linear order.

**Expected symptoms:** The same exact rest orbit must survive, all three
fixed-charge directions must be nonzero, the time Legendre map must remain
invertible, the added Hamiltonian must be bounded below, and the uniform
low-momentum Dirac symbol must remain unchanged before nonlinear motion is
tested.

**Root cause:** The previous candidate was only half of a native oscillator.
The transfer coordinate already rotates with the carrier at the exact native
frequency.  Adding its kinetic response without its reciprocal restoring
energy necessarily changed the rest equation.  The matching potential is not
an opposing fitted force: it is the other half of the same positive harmonic
degree of freedom, with relative coefficient fixed exactly by
`2(1-cos theta)=dt^2 chi0^2`.

**Corrective action:** Promote the odd transfer oscillator only to the next
research gate, not to canonical V40.  Its overall unit normalization follows
from one-half kinetic normalization, directed-bond double counting, and the
`chi0^4` field-dimension normalization; no Dirac velocity was fitted.  Next
derive its implicit canonical update and native current, then run spectrum,
finite two-cell handoff in both directions, and same-state discrete-Dirac
current comparisons.  V4-006 remains open until those tests pass.

## Transfer-oscillator spectrum gate (2026-11-23)

**Issue:** The completed transfer oscillator repairs the missing three-axis
linear response but does not make the exact localized rest orbit linearly
stable.

**Why it matters:** A finite translation or Dirac-current comparison cannot
certify the same electron if the starting carrier has an exponentially
growing native deformation.  The spectrum gate must distinguish a physical
action defect from an integrator artifact before any long handoff is credited.

**Symptoms:** The action-derived implicit update is locally reversible and
conserves both native phase charges to the tested tolerances.  At odd strength
`0.5`, the leading side-five Floquet modulus is `1.0010054665`, improved from
the canonical value `1.0040551110` but still outside the unit circle.  Adding
the independently rest-null even transfer oscillator gives a shallow optimum
near even strength `0.15`, with leading modulus `1.0009834292`.  Stronger even
coupling worsens the result.  Timestep refinement gives growth rates
`0.2516323` at `dt/2` and `0.2516318` at `dt/4`, so the growth survives the
continuous-time limit.  The leading mode is `81.8%` Psi, `96.7%` on the
carrier cube, and only `1.03%` in the three translation tangents.  A positive
onsite `|chi^dagger Psi|^2` stiffness leaves the unstable eigenvalues unchanged.

**Expected symptoms:** A passing action must keep every leading Floquet
modulus on the unit circle under timestep refinement, while preserving the
exact rest orbit, the three-axis transfer rank, the continuum C19 symbol, and
the independently derived native charges.

**Root cause:** The transfer oscillator supplies the previously absent
sender--receiver orientation channel, but the retained exact carrier still
has an internal Psi-dominated shape/phase instability.  It is not primarily a
translation mode, an exterior-chi mode, a pointwise Psi--chi alignment mode,
or a timestep instability.  The current odd/even transfer basis therefore
has the right transport rank but insufficient restoring curvature in the
remaining carrier tangent sector.

**Corrective action:** Keep the transfer oscillator as a successful response
basis but do not admit it or run a claimed two-cell translation.  Build the
rest-tangent Hessian decomposition of the unstable mode and enumerate the
next symmetry-allowed positive reciprocal action basis against that measured
sector.  Any repair must be rest-null without a prepared ray, projector,
corner label, copy selector, or Dirac matrix; it must then repeat the implicit
update, continuum, spectrum, rank, and finite-handoff gates.  Canonical V40 is
unchanged and V4-006 remains open.

## Exact-rest inverse-action feasibility (2026-11-23)

**Issue:** Individual action-term tests were not determining whether the
complete admissible exact-rest basis could satisfy stability and transport at
the same time.

**Why it matters:** Continuing to add one stiffness after another would be a
tuning loop, not a derivation.  The exact carrier, positive energy, C19
symmetry, continuum Dirac symbol, transport rank, and time-map invertibility
must be imposed together.

**Symptoms:** The positive exact-rest cone contains the odd and even transfer
oscillators, the density-balance square, and the pointwise parallel-overlap
square.  The spectral area-curvature square was rejected before tuning because
its unit rest defect is `0.0258556` RMS and its rest energy is `2986.17`.
The parallel-overlap square changes the leading radius by only `3.97e-13`.
A joint resolved screen reduces the physical growth rate from canonical
`1.03601` to `0.0749828` at diagnostic coefficients
`(odd,even,density)=(0,2,0.5)`, a `92.76%` reduction, while keeping the rest
DEL below `4e-13` and the Legendre condition number finite at `1212.66`.
Nevertheless the leading modulus remains `1.0002929446`; neighboring resolved
points in odd, even, and upper-density directions are worse.  The lower-density
edge did not converge efficiently and is retained as inconclusive.

**Expected symptoms:** A feasible coefficient region must put every leading
modulus at or below `1+1e-7`, preserve exact rest and rank-three response, keep
the Legendre map nonsingular, and leave the uniform quadratic Dirac symbol
unchanged.

**Root cause:** The declared positive exact-rest basis can strongly suppress
the carrier instability but does not eliminate the last internal mode in the
resolved coefficient region.  Requiring every added energy to vanish on the
old amplitudes removes the action curvature needed to move the carrier onto a
nearby stable branch.  The remaining obstruction is therefore not another
missing coefficient in this cone; it is the exact-old-amplitude restriction.

**Corrective action:** Stop coefficient tuning in this basis and do not admit
the diagnostic minimum.  The next inverse problem must permit the action to
re-solve a nearby rest orbit while preserving the eight-site corner identity,
opposite-pair relations, charges, and containment rather than preserving every
old amplitude exactly.  Solve state and action coefficients together, then
repeat spectrum, transport rank, continuum, implicit-update, and finite-handoff
gates.  Canonical V40 is unchanged and V4-006 remains open.

## Internal-deformation localization (2026-11-23)

**Issue:** It was not yet known whether the remaining unstable eigenmode was
translation of the localized carrier, deformation of its eight-site internal
pattern, or radiation into the exterior.

**Why it matters:** A transport repair would be misdirected if the carrier is
already internally unstable at rest.  Conversely, an ordinary wave-packet tail
must not be mistaken for loss of the electron's defining corner relations.

**Symptoms:** At the best resolved diagnostic coefficients
`(odd,even,density)=(0,2,0.5)`, the leading mode is `49.45%` Psi and `50.55%`
chi.  `89.50%` of its power remains on the eight carrier sites and only
`1.20%` projects onto the three translation tangents.  The corner response is
strongly uneven: sites `(1,1,2)` and `(2,2,2)` carry `21.65%` and `19.11%` of
the mode, while `(1,2,1)` carries only `3.73%`.  The exterior part is almost
entirely Psi (`10.24%` of total power) rather than chi (`0.25%`).  A direct
full-field nearby-branch solve was attempted, but its dense bordered system
was ill-conditioned and computationally inefficient; it was stopped without
claiming a solution.

**Expected symptoms:** A preserved moving carrier should have its neutral
direction in the translation tangent, retain balanced opposite-corner and
Psi--chi orientation relations, and attach only a controlled convergent tail.
A nearby stable rest branch should solve the unrestricted native equations
before any launch is applied.

**Root cause:** The remaining failure is an internal coupled Psi--chi
twist/shear mode of the prepared eight-site carrier.  Chi changes orientation
and amplitude unevenly across the corners while Psi responds out of phase;
Psi then supplies the first-shell leakage.  Motion excites and exposes this
mode but is not its source.  This is a candidate-state/action mismatch, not a
Dirac wave-packet spreading result and not primarily a timestep error.

**Corrective action:** Replace the brute-force full-lattice branch solve with
a symmetry-adapted reduced basis containing the eight carrier sites, the first
shell, the measured unstable vector, frequency, and phase constraints.  Solve
the nearby joint Psi--chi rest branch there, lift it to the unrestricted
lattice, and require a small full residual.  Only a lifted branch that passes
the full spectrum, rank-three transport, continuum, conservation, and two-cell
handoff gates may be admitted.  Canonical V40 remains unchanged and V4-006 is
open.

## Complete-frame inverse-action RCA (2026-11-24)

**Issue:** The current action transports the oriented area
`W = chi wedge Psi`, but the long-time four-component packet still develops
an in-plane `Psi` defect. The omitted parallel coordinate
`s = chi^dagger Psi` was tested as the proposed missing personal-touch
channel.

**Why it matters:** The pair `(s,W)` is the complete local relational frame,
because `rho_chi rho_Psi = |s|^2 + ||W||^2`. If its exact rest-preserving
symplectic action directions spanned the failure, V4-006 could proceed to a
coefficient derivation without inserting a spinor, projector, or Dirac
force.

**Symptoms:** Two additional exact rest-force null directions were derived
from the native tensor/area one-forms. Their rest DEL norms are `3.81e-15`
and `7.00e-16`, with the required symmetric mass/stiffness and antisymmetric
gyro blocks. The complete-frame span reduces the one-cycle relative defect
from `0.587366` to `0.325805`. Combining it with all seven previously derived
reciprocal invariants raises the response rank from six to eight and reduces
the best diagnostic residual to `0.191699`, a `67.36%` improvement. It does
not close the `<0.05` gate. The fitted common/relative coefficient is
`558.26` in the one-percent response coordinates and has neither geometric
provenance nor an admissible interpretation.

**Expected symptoms:** A closing action family must contain a
rest-preserving direction whose native response reduces the independently
measured endpoint residual below `0.05`, retains positive kinetic structure,
and reproduces the same improvement at coefficients derived from native
geometry or degeneracy rather than from the Dirac comparator.

**Root cause:** The parallel overlap is a real part of the missing response,
but local and symmetric C19-gradient complete-frame Berry forms are still
insufficient. Their spatial response is predominantly even in lattice
momentum, whereas the unresolved Dirac transport channel is the signed,
component-selective part that is odd in momentum. Existing C19 Bargmann
triangle controls can turn quiet `chi`, but their exact translation-zero-mode
test explains only `5.27e-6` of the required squared response and their copy
compression leaks outside the eight-component tangent. Thus the remaining
failure is not simply weak `chi` stiffness or an omitted scalar overlap
spring; it is a missing admissible spatial-temporal connection.

**Corrective action:** Do not install the fitted complete-frame
coefficients. Derive the lowest-degree cubic/C19 spatial-temporal one-form
whose orientation is supplied by the native `(s,W)` frame, whose rest
variation vanishes, and whose Bloch response is odd in momentum. Its inverse
gate must simultaneously test both copies, all three axes, positivity of the
kinetic mass, and the native current. If no such rest-preserving direction
exists, re-solve a deformed rest orbit under the full connection before
repeating the same tests. V4-006 remains open and neither the canonical nor
research action was changed.

## Personal-touch full-frame action ladder (2026-11-23)

**Issue:** Ten consecutive mathematically distinct native full-frame action
families failed to improve upon the preserved exact-rest candidate's finite
growth rate `0.0749828398`; V4-002 and therefore V4-006 remain unclosed.

**Why it matters:** The working hypothesis was that the canonical density and
`W = chi wedge Psi` variables discard part of the complex three-prong frame.
If that were the sole missing mechanism, a reciprocal action on the complete
frame or its timing should have removed the internal shear while preserving
translation and the continuum symbol.

**Symptoms:** The canonical leading rate is `1.03600926`, with `99.8553%` Psi
participation, complete carrier support, and only `0.538%` translation
overlap. The preserved unadmitted candidate reduces this to `0.07498284`, but
its residual mode is a coupled corner-asymmetric Psi--chi shear. Local
parallel overlap is silent. Local Gram-area stiffness bottoms out at
`0.701071`; separate C19 dyads do not improve canonical growth. The complete
cross-dyad source has a `3.9479%` Fredholm obstruction. Joint-plane position
and kinetic terms do not improve the spectrum. The rest-preserving Bargmann
connection sends at least `99.4458%` of its response outside the retained
eight-component tangent. Local area and full-cross Berry forms do not beat
the preserved best. Local and C19 cross-frame oscillators only slow the mode
as inertia increases; they do not change its unstable sign. Finally, the C19
cross-frame Berry source has an `8.2974%` Fredholm obstruction. Two additional
bond controls confirmed rather than weakened the ten-round verdict.

**Expected symptoms:** A passing personal-touch action would have an exact or
freely re-solved nearby orbit, a bounded non-symmetry spectrum under timestep
and box refinement, rank-three response, reversible conservation, and a free
relative-periodic carrier completing two translations in both signs of every
axis while retaining identity and matching the discrete Dirac current.

**Root cause:** The compact eight-site relative equilibrium is not missing one
ordinary measurement of its local Psi--chi frame. Position-only complete-frame
energies either have the wrong curvature or cannot continue from the retained
orbit. Positive full-frame kinetic memories add inertia but preserve the
unstable energy sign. The admissible phase connections either leak out of the
four-component tangent or encounter a Fredholm range obstruction. The
remaining failure is therefore a deeper carrier/action-class mismatch, not an
unmeasured scalar overlap, insufficient coefficient, timestep error, or
translation launch error.

**Corrective action:** Stop this action ladder at the mandated ten-round
boundary and retain all negative evidence. Do not promote any coefficient or
modify canonical V40. The next justified research program must either derive
a different native action class with genuinely new response rank, or allow a
noncompact carrier/tail whose constrained energy-momentum index can be stable.
Any such program begins from the preserved best and must independently repeat
rest, spectrum, conservation, C19 continuum, translation, and Dirac-current
gates; it is outside this stopped goal.

## Loop-resolved inverse action derivation (2026-11-23)

**Issue:** The preserved compact carrier still has an exponentially growing
internal mode, so V4-002 and V4-006 remain open.

**Why it matters:** A compact electron cannot support the later motion and
Dirac-current gates if its native rest orbit amplifies an internal deformation.
The correction must arise from a generic reciprocal action rather than an
eigenmode projector or prepared corner rule.

**Symptoms:** The best prior exact-rest candidate has growth
`0.0749828398`.  Its leading mode is a mixed Psi--chi carrier twist/shear.
The cubic full-frame commutator action reduces the measured worst growth to
about `0.0663` before a different Psi/exterior mode becomes limiting.  The
stronger square-face factorization identity has `66.97%` oracle alignment on
the original mode and `57.87%` on the residual mode.  At strength `0.1` it
reduces the converged outer growth to `0.0367516694`, but the sparse solve
converged only two outer eigenvalues and still found growth.  Strength `0.2`
is slightly worse at `0.0393681382`.

**Expected symptoms:** A pass requires the complete refined spectrum to lie
on the unit circle, the exact rest pattern to remain intact, and the same
action to retain conservation, continuum propagation, and later movement
gates without runtime support.

**Root cause:** Pairwise/local frame energies do not control a consistency
condition accumulated around a whole square face.  The missing first response
is loop-resolved.  The full-frame commutator closes the mixed internal shear
but is nearly blind to the subsequently exposed exterior-Psi family.  The
opposite-corner factorization defect responds to both families and halves the
growth, but one positive square alone does not span the full unstable
subspace.  The remaining failure is response-rank incompleteness, not lack of
coefficient magnitude.

**Corrective action:** Retain the face-factorization square as the current
best new action-basis direction, not as an admitted equation.  Derive the
remaining independent zero-on-rest face identity from the residual mode,
form the symmetry-complete loop Hessian basis, and solve its positive
semidefinite stability cone before another nonlinear movement run.  Repeat
the complete spectrum with timestep and box refinement for any survivor.
Canonical V40 remains unchanged.

## Symmetry-complete face cone (2026-11-23)

**Issue:** One square-face factorization term halved the measured instability
but did not close the rest-spectrum gate.

**Why it matters:** If every generic identity that vanishes on the rest carrier
had the same Hessian image, further face terms would only retune one stiffness
and could not repair the remaining mode.

**Symptoms:** Nine native square-face defects were enumerated. Six positive
squares vanish with zero first force on the retained rest orbit. Their joint
Hessian response has rank five, not one. A positive-cone solve selected
`0.7205479983` diagonal factorization plus `0.2794520017` edge commutator. At
total diagnostic strength `0.1`, leading measured growth falls to
`0.0361111401`, versus `0.0367516694` for the first face term and
`0.0749828398` for the incumbent. The rest DEL residual remains
`3.64e-13`. Only two outer Ritz values converged, and both remain outside the
unit circle.

**Expected symptoms:** A passing cone would produce a converged refined outer
spectrum with no multiplier above `1+1e-7`, while preserving the exact rest
orbit and all action symmetries.

**Root cause:** The first face term was response-rank incomplete. The derived
edge-commutator direction is genuinely independent and improves the worst
measured rate, proving that loop-response rank matters. The present two-mode
positive-cone objective is nevertheless incomplete: after the correction, an
unmeasured residual family still limits the spectrum, and the nonnormal sparse
eigensolver does not yet provide a complete outer basis for that family.

**Corrective action:** Preserve the rank-five zero-on-rest face basis and the
derived two-term ratio. Replace repeated ARPACK coefficient trials with a
cached Hessian/quadratic-eigenvalue continuation that returns the full outer
unstable subspace. Add each newly exposed mode to the semidefinite cone until
either the complete spectrum closes or the positive face cone is proved
insufficient. Do not admit the action or proceed to motion meanwhile.

## Complete face-cone limiting mode (2026-11-23)

**Issue:** The complete five-direction positive face cone reduces but does not
eliminate the outer discrete growth rate.

**Why it matters:** Continued face stiffening is justified only if the
remaining unstable state is still an electron face deformation. If the state
lies in a variable the face action cannot observe, coefficient growth can only
hide the problem asymptotically.

**Symptoms:** Exact shift-invert reproduces the incumbent growth to 12 digits
and resolves 32 outer multipliers. The five-term face cone reduces growth from
`0.0749828398` to `0.0120266852` after coordinate refinement. Uniform positive
strength then gives `0.00879317`, `0.00351515`, `0.00159825`, `0.000980982`,
and `0.000551140` as the added strength rises through `0.1`, `0.3`, `1`, `3`,
and `10`; it approaches zero without crossing it. At the strongest point the
limiting vector is `99.9692%` chi, `99.9685%` exterior chi, and `99.9536%`
exterior transverse chi orientation. Its face curvatures are nearly zero.

**Expected symptoms:** A complete face repair would eliminate the relevant
electron deformation at a finite, derived coefficient and leave only exact
symmetry multipliers on the unit circle.

**Root cause:** Every face correction is built from `W = chi wedge Psi`. In
ordinary vacuum `Psi=0`, hence `W=0`, so all W-face identities are blind to
relative transverse orientations of neighboring chi vectors. The native local
potential also depends only on `|chi|`, not its orientation. Consequently the
exterior substrate contains a large set of nearly free chi-orientation hinges.
Face stiffness drives the electron deformation into that blind exterior
channel but cannot close it at finite strength.

**Corrective action:** Stop increasing W-face coefficients. Derive and screen
the lowest native chi-only C19 orientation action, for example the positive
phase-invariant dyad difference
`sum w_ij ||chi_j chi_j^dagger-chi_i chi_i^dagger||_F^2`, with its complete
reciprocal chi force. Re-solve the localized rest orbit because this term need
not vanish inside the carrier. Admit it only if its coefficient follows from a
native degeneracy/bifurcation condition and it preserves the established
rest, continuum, conservation, and Dirac mappings. Canonical V40 is unchanged.

## Chi-orientation action branch (2026-11-23)

**Issue:** The complete W-face cone leaves an exterior transverse chi hinge,
and the proposed generic chi-direction bond action had not been isolated from
the previously tested combined Psi/chi frame action.

**Why it matters:** If a native chi-only interaction cannot see that hinge or
cannot coexist with the localized rest carrier, the face-cone diagnosis does
not provide an actionable repair. If it can, its full re-solved spectrum must
still close before the action can be admitted.

**Symptoms:** The positive C19 dyad-difference action has Hessian curvature
`41.7801653` on the limiting mode, which is `99.9527%` chi in this extraction.
Its source has relative Fredholm obstruction `6.79e-15`, and unrestricted
side-three branches solve below `1e-10` DEL residual while preserving Psi
carrier fidelity at numerical one and zero Psi exterior fraction. Combined
with the five rest-preserving face terms, continuation reduces growth to
`0.1011380` with four unstable modes at diagnostic strength `eta=1`. At
`eta=10`, growth rises to `0.3811237` with 32 unstable modes and chi exterior
deviation grows to `0.3081`.

**Expected symptoms:** A passing repair would produce a finite, independently
derived coefficient at which the freely re-solved branch has no non-symmetry
multiplier outside the unit circle, while preserving the same carrier,
conservation, continuum response, and later movement gates.

**Root cause:** The missing response really is chi-orientation continuity, but
a uniform global chi-direction stiffness is too coarse. It controls the
diagnosed hinge and initially reduces the residual instability, then deforms
and over-constrains the surrounding chi medium, causing other mode families
to become unstable. The required interaction must be localized or
state-sensitive through native invariants; it cannot be obtained by merely
increasing one global stiffness coefficient.

**Corrective action:** Retain the chi dyad difference as a validated response
basis direction, not an admitted action. Derive its lowest native
amplitude-sensitive or carrier-boundary activation from Psi/chi invariants,
with a positive real reciprocal action and no prepared rays, projectors, or
Dirac data. Require coefficient provenance, re-solve the rest branch, and
repeat side-five spectrum, timestep, Hamiltonian/Noether, nonlinear, and
movement gates. V4-006 remains open and canonical V40 is unchanged.

## Unit geometric bare-action candidate (2026-11-23)

**Issue:** Uniform chi-direction stiffness reduced the exterior hinge but
deformed the substrate, while vacuum-only activation left a purely internal
Psi carrier mode untouched.

**Why it matters:** V4-006 cannot close unless the same native action controls
both the surrounding chi orientation and the carrier's internal Psi geometry
without held fields, fitted coefficients, or the earlier transfer/density
scaffolding.

**Symptoms:** Vacuum activation removes 12 of 32 low stiffness directions but
has curvature `1.19e-15` on the remaining mode, which is `99.99993%` Psi and
entirely on the carrier. Adding the C3 Psi tight-frame bond response and the
complete unit-weight face/bond basis reduces the targeted discrete growth to
the unit-circle tolerance. On the bare canonical V40 baseline, seven exact
real/complex shift-invert solves converge with maximum resolved modulus
`1.0000000202809898`; none exceeds `1+1e-7`.

**Expected symptoms:** Full closure requires the complete spectrum to contain
only symmetry/unit-circle multipliers, followed by stable unrestricted
nonlinear evolution, conservation, refinement, and same-carrier Dirac motion.

**Root cause:** Two independent geometric responses were missing from the
bare action: orientation continuity of complex chi in ordinary vacuum and
C3 tight-frame continuity of complex Psi inside the carrier. A uniform chi
penalty addresses the first but damages interfaces; a vacuum-activated chi
term addresses only the first. The symmetry-complete unit action supplies
both without acting on the exact rest solution.

**Corrective action:** Promote the unit geometric action only to the next
research gate. Derive its full reciprocal nonlinear forces and use the
corresponding variational update. Complete a complex-contour eigenvalue count,
then run Hamiltonian/Noether, timestep/box, 100-cycle perturbation, movement,
and discrete Dirac regression gates. Canonical V40 remains unchanged until
all required gates pass.

## Nonlinear rest-orbit and chi-clock gate correction (2026-11-23)

**Issue:** The unit geometric candidate appeared to accumulate about `9.47%`
raw state error over ten cycles when the previous chi layer was given a new
one-percent phase velocity.

**Why it matters:** This could have meant that the linear repair failed
nonlinearly. It could also have meant that the test compared states in
different conserved-charge sectors and therefore demanded impossible
dissipation from a Hamiltonian system.

**Symptoms:** With the one-percent chi clock, the ten-cycle run keeps Psi
carrier fidelity at numerical one and Psi exterior fraction at zero, but chi
density changes by `0.1993` and the area-projector residual reaches `0.2301`.
Charge is conserved. With the chi clock removed, the same unrestricted run
has only `4.02e-15` shape residual after ten cycles. The held-out 100-cycle
run has `5.65e-14` shape residual, zero chi-density drift, zero Psi exterior
population, numerical-unit Psi fidelity, and `2.08e-15` interval-energy
drift.

**Expected symptoms:** The exact zero-chi-charge orbit should remain fixed.
A nonzero-chi-charge test must instead compare with the equilibrium or
relative orbit solved at that same conserved charge.

**Root cause:** Two issues had been conflated. The bare action's original
instability came from missing full-frame geometric communication, especially
where `Psi=0` made `chi wedge Psi` blind to substrate orientation. The later
clock-test breathing was not spontaneous loss of that repaired geometry: the
initial condition injected a new conserved chi charge while retaining the
zero-charge equilibrium radius. The local radial force then produced the
expected charged-sector breathing, and a conservative action had no channel
that could erase the injected charge.

**Corrective action:** Accept the 100-cycle result only as the exact rest-orbit
gate. Replace the invalid clock-recovery criterion with a fixed-charge branch:
solve the nearby chi-charged equilibrium/relative orbit, perturb within its
charge sector, and test bounded recovery there. Ablate the nine unit geometric
identities to find the minimal action, then perform timestep/box refinements,
movement, and same-embedding discrete-Dirac transport. V4-006 and canonical
promotion remain open.

## Minimal interface reduction and charged-branch scope (2026-11-23)

**Issue:** The nine-term unit action could be overcomplete, and the corrected
nonzero-chi-charge gate required a nearby relative orbit rather than the
zero-charge preparation.

**Why it matters:** An overcomplete action would obscure the actual mechanism.
Failure to solve the charged orbit would leave the perturbation gate open even
though the exact rest orbit passes.

**Symptoms:** Sequential two-shift ablation removes eight terms. Omitting the
remaining carrier--vacuum chi interface term reopens a multiplier at
`1.00000242266`. The one-term action keeps the exact rest orbit for ten cycles
with `3.10e-14` maximum shape error, zero chi-density drift, and zero Psi
exterior population. Five of six full-band shifts converge on the unit circle;
one shift returns no eigenvalues, so the full-band result is inconclusive. A
three-amplitude charged-orbit solve leaves unrestricted DEL residual norm
`8949.50` despite having no orthogonal orientation residual.

**Expected symptoms:** The reduced action must close every spectral shift and
the nonlinear/refinement gates. A correct charged branch must have a small
full unrestricted DEL residual, not merely a small residual outside the
chosen radial subspace.

**Root cause:** The stabilizing response is concentrated at the native
carrier--vacuum interface: density activation identifies unlike regimes, and
the chi-overlap square supplies transverse directional pushback. The initial
three-scale charged ansatz is too coarse because distinct shells/sites require
different radial adjustments even though no new orientation defect appears.

**Corrective action:** Retain the single interface term as the minimal leading
candidate, not an admitted action. Retry the one unconverged spectral shift
with an independent contour/count method. Solve the charged relative orbit
with shell-resolved radial amplitudes or the unrestricted symmetry-reduced
Newton system, then run its perturbation gate. Only afterward perform box/dt,
movement, and discrete-Dirac transport. Canonical V40 is unchanged.

## V4-006 motion verdict after corrected bare-action reduction (2026-11-23)

**Issue:** The exact localized rest carrier must translate under the same
native action and reproduce the already derived discrete-Dirac transport and
current. The earlier one-term ablation used a tangent cache containing two
historical diagnostic scaffolds and was therefore rerun from the audited bare
V40 matrices.

**Why it matters:** Rest stability alone is not a Dirac electron. V4-006
requires a freely solved multi-cell trajectory with no preload, projection,
reset, or maintained background. A wrong baseline could falsely identify the
action responsible for stability or motion.

**Symptoms:** The corrected seven-shift greedy reduction retains only the
carrier--carrier chi and Psi tight-frame bond energies. They pass all seven
resolved shifts and a ten-cycle unrestricted zero-charge run with maximum
shape residual `1.96e-14`. Nevertheless the three centered translation
tangents have one-update residuals `6.39e-4`, `6.40e-4`, and `6.80e-4` and are
not neutral. Restoring the carrier--vacuum interface term improves this by
less than `0.1%`. Across the full nine-term position basis, the best
nonnegative inverse response leaves `16.14%` of the translation target; even
unconstrained coefficients leave `15.84%`. Free relative-periodic shooting
reduces the first-interval residual from `0.915` at two updates to `0.197` at
eight updates, but the unreset second interval is `1.311`. The previously
derived joint density-mold C19 kinetic connection, combined with the tight
frames and its correct implicit update, has rest spectral radius
`1.0031050201` and growth `0.7936536`, so it fails admission. No same-state
Dirac-current comparison exists because no native moving branch closes.

**Expected symptoms:** A passing action has a signed three-axis native launch,
a freely solved one-cell relative orbit that repeats for a second unreset
cell in both signs, and native charge/current agreeing with the independently
evolved discrete-Dirac comparator while retaining the same carrier.

**Root cause:** The tight-frame additions are even spatial stiffness. They
preserve the carrier's internal C3 geometry but do not supply the missing odd,
component-selective response that communicates orientation and charge across
the carrier--vacuum boundary. The localized orbit remains lattice-pinned.
The complete tested position basis cannot span that response, while the
tested joint kinetic metric changes radial mold inertia but reopens the rest
instability. Conditional Dirac incidence transport exists as a readout, but
the tested native action does not generate its moving localized state.

**Corrective action:** V4-006 remains open and none of these research actions
is promoted. The next distinct admissible class is the complete lowest-degree
joint Psi--chi field-dependent symplectic connection on C19
carrier--vacuum/common-neighbor geometry. Enumerate it as an action basis,
derive reciprocal equations and canonical momenta, and run the exact
rest/translation response-rank test before any new nonlinear shooting. Its
coefficients must follow from cubic geometry, positivity, degeneracy, or a
native bifurcation—not from fitting Dirac velocity. The obstruction is scoped
to the tested bases, not a universal no-go for every two-field action.

## Rigid-snap target audit (2026-11-23)

**Issue:** The movement gate had conflated two distinct hypotheses: a rigid
one-cell translation of the entire localized carrier, and the conditional
forward/backward channel transport generated by the cubical Dirac algebra.

**Why it matters:** Tuning the native action against the wrong endpoint can
reject a correct Dirac mechanism or encourage an inadmissible scalar
advection rule. Massive Dirac transport is internal-channel dependent.

**Symptoms:** Under the reduced two-tight-frame action, the optimized literal
snap residual is `0.99875`--`0.99905` in all six signed axes. The conditional
target has the same near-unit second-difference residual. Independently, the
algebraic conditional walk approaches the cubical Dirac step with
`2.44e-6` state error at the smallest tested timestep, while the rigid shift
retains `1.42e-2` error. Applying the internal cubical projector directly to
physical corners was previously shown not to establish the required lift.

**Expected symptoms:** A pass requires one translation-equivariant joint
Psi/chi embedding whose two-layer native image remains in the same
four-component sector, with signed `+gamma/-gamma` bond currents and local
mass mixing reproducing the independent discrete-Dirac comparator.

**Root cause:** A centered second-order equation squares away the sign of the
first-order incidence channel: `P+ + P- = I` and `gamma^2 = I`. Consequently
an acceleration-only or rigid-endpoint gate cannot identify Dirac transport.
The current action also does not supply the required translated chi response.
The second Clifford copy duplicates the balanced forward/backward channel
structure rather than acting as a leading or trailing carrier.

**Corrective action:** Retire rigid whole-carrier snapping as the massive
Dirac acceptance target. Reuse the surviving overlapping-cube spatial lift,
extend it under the current action, calculate
`(I-EE^dagger)L_native E`, and pull the native bond current through the same
embedding. The first calculation reduces the cross block from about `71%`
for one center to `8.66%`/`8.43%` for the two copies on the nearest 27-center
envelope. That is progress but not closure. Solve the invariant-graph/Schur
completion of this remaining residual with both committed layers. Only the
unresolved part after that calculation is an action-repair target. V4-006
remains open; no action or canonical change is authorized.

### Invariant-graph result

The complete side-five envelope reduces the spatial cross block to `1.3924%`.
The remaining response is a rank-32, out-of-plane Psi tilt, not exterior chi
and not a high-momentum-only artifact. A native generalized-eigenmode dressing
slaves that tilt to the original carrier with no dimension increase and closes
the configuration tangent below `7e-15` for both copies. The projection is
well conditioned, so this is not a singular redefinition.

The same-rank graph does not yet close the actual two-layer map: its residual
is `0.002164` and rank 32. The enlarged phase-response hull closes near
`2e-11`, proving the missing information is finite and identified, but the
extra phase directions still need to be expressed as the previous-layer map
of the same carrier rather than retained as independent coordinates. The
next corrective action is the nonlinear matrix/Riccati or spectral graph
reduction of this rank-32 time-layer response, followed by the native-current
pullback. Do not change the action unless that same-rank reduction fails.

## Full two-layer stability RCA (2026-11-23)

**Issue:** The completed configuration/phase hull must be perturbatively
stable before its native current can represent a moving localized Dirac
carrier.

**Why it matters:** An exact orbit can remain numerically exact forever while
arbitrarily small nearby perturbations grow. Such an orbit does not satisfy
the V4 electron stability gate and cannot support a robust moving branch.

**Symptoms:** The exact rank-1032 configuration hull closes under the full
two-layer update below `1.9e-13`, but its 2064-dimensional monodromy has 184
growing multipliers. The maximum growth rate converges from `1.0360093` to
`1.0360122` under `dt`, `dt/2`, `dt/4` refinement. The fastest mode has
`0.9999959` target overlap, is almost entirely in-plane Psi, and has bare-V40
generalized stiffness `-1.0807423`. Both tight-frame terms contribute less
than `4e-17`. Its normalized tangent to the Psi and chi conserved charges is
only `5.29e-15` and `8.33e-15`, respectively.

**Expected symptoms:** A passing rest carrier has no charge-neutral growing
multiplier after timestep refinement. Its same invariant embedding must then
carry a native bond current agreeing with the discrete Dirac comparator.

**Root cause:** The tight-frame action repairs orientation and the derived
normal Psi dressing, but it is blind to a bare-V40 in-plane amplitude/phase
direction. In the rotating frame the restoring local and C19 area curvatures
do not overcome the native clock contribution; the exact discrete operator
leaves negative net curvature. This is independent of the earlier
out-of-plane leakage and is not removed by fixing the conserved charges.

**Corrective action:** Reject promotion of the present action. Derive the
complete lowest-degree symmetry-allowed reciprocal action response on this
in-plane mode, subject simultaneously to: zero first variation on the rest
orbit, positive curvature exceeding the native deficit without fitting its
coefficient, preservation of both U(1) charges, cubic/C19 symmetry, and a
nonzero odd three-axis transport response. Re-run the full two-layer spectrum
before nonlinear shooting or current pullback. Canonical V40 remains
unchanged; V4-006 remains open.

## Density-oriented pair-connection RCA (2026-11-23)

**Issue:** A mathematically derived joint Psi--chi transport connection
strongly suppresses the current carrier instability but does not remove every
growing mode or establish a freely translating localized branch.

**Why it matters:** V4-006 requires one native action to preserve the rest
carrier and move the same four-component state in all three directions. A
large improvement is not a pass while any converged growing mode or missing
same-state current remains.

**Symptoms:** Static complete-pair energies either break the exact rest orbit
or are blind to the failing mode. A matched positive pair-wave action
preserves rest but retains maximum growth `0.2812` even at coefficient
`chi0^2`. The density-oriented common-pair one-form preserves rest below
`4e-17`, responds in all three axes, and at the native scale `B*chi0=22743`
reduces growth from `1.0360122` to `0.0710213` (`93.144739%`). Nevertheless,
184 multipliers remain above `1+1e-7`. Naive addition of the relative-pair
channel or reversal of the common-channel sign makes the spectrum worse.

**Expected symptoms:** A passing joint connection has zero first variation on
the rest orbit, a positive kinetic form, no growing two-layer multiplier after
timestep and box refinement, nonzero signed transport response in all axes,
and a nonlinear moving branch whose native charge/current matches the frozen
discrete-Dirac comparator.

**Root cause:** Canonical `chi wedge Psi` carries only antisymmetric pair
orientation. The density-oriented common-pair connection supplies much of the
missing odd, velocity-sensitive handoff information, but its linearized
response does not span the complete residual sector. The relative-pair
channel supplies complementary directional response, especially along z,
yet the naive coefficient/sign combination couples to the wrong remaining
modes. The unresolved response must be identified from the new residual; it
cannot be installed as a fitted counterforce.

**Corrective action:** Keep canonical V40 and the admitted research action
unchanged. Extract and classify the fastest residual mode at the natural
common coefficient by field, lattice site, corner component, copy, and bond
symmetry. Enumerate only the independent reciprocal one-form or positive
curvature channels that are nonzero on that residual while retaining exact
rest cancellation. Solve their response-rank and native coefficient
conditions before another full spectrum or nonlinear motion run. V4-006
remains open.

## Residual mode-switch RCA (2026-11-23)

**Issue:** The density-oriented common-pair connection nearly stabilizes the
Psi carrier, but the complete two-layer spectrum retains a coupled sequence
of Psi phase and chi-interface growing modes.

**Why it matters:** Stabilizing only the originally fastest mode is not a
stable electron. The same reciprocal action must control both the carrier's
corner phase and the receiving substrate orientation before motion/current
can be tested.

**Symptoms:** At `B*chi0`, the fastest mode is a pure `001` eight-corner Psi
phase character. Fine continuation of the common coefficient reaches a
nonzero growth plateau `0.0675966` near factor `0.95`. At that plateau the
fastest mode switches to `93.37%` chi and `93.15%` the third C3 component,
supported on the carrier and first shell. Positive unit chi-interface terms
reduce the growing count from `184` to `30`. Adding the best declared
area-bilinear phase response reduces maximum growth to `0.0631092`, but
`29` growing multipliers remain.

**Expected symptoms:** A passing action has no growing multiplier in the full
two-layer invariant hull, retains a positive kinetic form, preserves the
exact or continuously re-solved rest orbit, and has nonzero signed transport
in every axis before nonlinear two-cell and Dirac-current gates are run.

**Root cause:** There are two independent deficiencies. The canonical wedge
area channel omits part of the joint phase handoff, producing the `001` Psi
phase defect. Once the common-pair connection suppresses it, the absence of
chi orientation communication where `Psi=0` exposes an out-of-plane
carrier--vacuum chi mode. A single scalar connection or static phase spring
cannot close both. The six native scalar orientations span five response
directions, excluding a one-channel collapse but not yet proving that the
available space can reach zero growth. The remaining question is a joint
spectral-degeneracy/coefficient problem.

**Corrective action:** Do not install any screened combination. Project the
five independent action responses onto the remaining 29-mode invariant
subspace and solve the Hamiltonian bifurcation conditions for a positive-mass,
zero-growth coefficient relation. Accept only coefficients derived from
native normalization, cubic degeneracy, positivity, or the bifurcation
condition. Validate any survivor in the full 2064-dimensional spectrum under
timestep/box refinement before nonlinear movement. Canonical V40 remains
unchanged and V4-006 remains open.

## Cube-character bifurcation RCA (2026-11-23)

**Issue:** The complete declared scalar-oriented pair-connection screen and a seventh
positive cube-phase stiffness both fail to remove the carrier's residual
growing spectrum.

**Why it matters:** This was the last position-only repair suggested by the
identified carrier and interface modes. Continuing to tune this family would
fit coefficients without adding the response that the remaining mode needs.

**Symptoms:** The six-response continuous solve retains all `29` growing
multipliers and lowers growth only from `0.0631092` to `0.0625757`. Its
reduced prediction agrees with the full `2064`-dimensional calculation. The
fastest mode is wholly on the carrier, lies in the first two C3 components,
and gives the same `001` corner character in Psi and chi, with restricted-C19
eigenvalue exactly `8`. Adding the positive area-bilinear phase response gives
best continuous growth `0.0601106` with `27` growing multipliers; the exact
cube factor `12/8` gives `0.0608864`, also with `27`.

**Expected symptoms:** A passing repair must put every full-spectrum
multiplier on the unit circle with positive kinetic mass, preserve the exact
rest orbit, and retain nonzero odd transport response in all three axes.

**Root cause:** On the common Psi--chi `001` phase motion, all six scalar
pair connections have effectively zero restoring-stiffness quadratic. They
alter inertia but do not rotate the phase-space response needed at reversal.
The positive cube-phase spring acts on the mode, but it cannot repair the
remaining Hamiltonian phase coupling by stiffness alone. The limiting defect
has moved from the quiet exterior into the carrier's joint phase space.

**Corrective action:** Retire further coefficient tuning of the seven tested
position-only responses; this is a scoped research decision, not a universal
no-go theorem. Enumerate the complete lowest-degree cubic and C19
incidence-based symplectic Psi--chi action basis, including reciprocal
variations and all three cube characters simultaneously. Use the exact
eigenvalue-8 residual as the inverse requirement but never as a selected-axis
rule. Require a native degeneracy/positivity coefficient, positive kinetic
form, exact rest preservation, and a zero-growth full-spectrum pass before
nonlinear movement or Dirac-current testing. Canonical V40 remains unchanged;
V4-006 remains open.

## Compact-hop test interpretation RCA (2026-11-23)

**Issue:** The motion program treated the eight-site internal carrier as the
complete spatial support of the electron and required it to repopulate an
exactly quiet neighboring cube.

**Why it matters:** Free Dirac transport evolves an extended four-component
wavepacket. A compact-support hopping requirement can fail even when the
native low-momentum carrier envelope has the correct continuum motion.

**Symptoms:** Compact launches produce no exterior Psi support. In contrast,
the self-contained overlapping-cube packet remains inside the native spatial
lift to `6.75e-5`, `1.69e-5`, and `4.23e-6` under timestep refinement. Its
native-versus-C19-Dirac mismatch falls from `0.00362501` to `0.000968164` to
`0.000246562`, with observed orders `1.90466` and `1.97330`.

**Expected symptoms:** The same joint Psi--chi packet should retain its
four-component identity, carry the native Noether current at the discrete
Dirac group velocity, converge with timestep and box refinement, and survive
unrestricted nonlinear propagation without a maintained background.

**Root cause:** The failed test confused the cubical spinor's internal unit
cell with the complete spatial wavefunction. It therefore imposed an
artificial quiet boundary where the electron's own moving envelope should
have supplied a small leading tail. It also initially compared a laboratory
Dirac phase with native tangent blocks expressed in the carrier co-rotating
frame.

**Corrective action:** Retain compact hopping only as a scoped ablation. Use
the frozen overlapping-cube joint-field lift, subtract the independently
derived positive-frequency rest phase, and test broad low-momentum packets.
Next derive native Noether current on this same packet, then run box/width and
nonlinear multi-step refinements. Do not modify the action unless that
corrected same-state test exposes a nonvanishing continuum residual.
Canonical V40 remains unchanged and V4-006 remains open.

## Cube-incidence symplectic repair RCA (2026-11-23)

**Issue:** The rest carrier was linearly unstable under every tested
position-only completion because its joint Psi--chi `001` timing mode lacked
a reciprocal phase-space response.

**Why it matters:** A localized carrier cannot support a moving Dirac readout
if the same state has growing rest perturbations. Stabilizing the exact rest
orbit is the prerequisite for deriving its translation tangent and current.

**Symptoms:** The incumbent full spectrum had growth `0.0640520` and `31`
growing multipliers. Six scalar connections had zero stiffness on the common
`001` mode, and the seventh positive phase spring still left `27` modes.

**Expected symptoms:** An admissible response must preserve the exact rest
DEL equations, obey reciprocal mass/gyro/stiffness identities, retain
positive kinetic mass, and put every full-state multiplier within the
declared unit-circle tolerance without selecting a direction or spinor copy.

**Root cause:** The previous action communicated density, static orientation,
and some odd transport, but did not gear the common and relative Psi--chi
phase motions through cube incidence. The residual was therefore an internal
phase-space timing defect, not insufficient stiffness or exterior forcing.

**Corrective action:** Retain for the next gate the unit-normalized derived
combination `A_area+,C19 - 4 A_tensor+,local + 4 A_tensor-,local
- 4 A_area+,local`. Its integer coefficients are fixed by the exact rest-force
nullspace, and its uncompressed spectrum has zero declared growing modes.
Derive the nonlinear implicit discrete variational update, then run energy,
charge, 100-cycle perturbation, unrestricted movement, and same-state Dirac
current gates. Reject it if those gates fail; do not use ordinary Verlet.
Canonical V40 remains unchanged and V4-006 remains open.

## Quiet-receiver support obstruction RCA (2026-11-23)

**Issue:** The unit cube-incidence action stabilizes the localized rest orbit,
but the tested freely solved launches do not transfer any `Psi` support from
the original eight sites into the quiet receiving sites.

**Why it matters:** V4-006 requires the same localized carrier to supply
native motion and the Dirac current. Internal redistribution inside a fixed
cube is not carrier translation, regardless of its center-of-density shift.

**Symptoms:** After correcting the Noether cut so it lies outside rather than
through the carrier, the instantaneous exterior current is zero. All 24
first-update tests (four native phase launches, both signs, all three axes)
leave exterior `Psi` exactly zero. In the eight-update common-launch test,
exterior `Psi` remains exactly zero in every completed layer while density
moves only among the original corners. Three signed runs also exhaust the
fixed-rest-Jacobian nonlinear solver before update eight. A symmetric
endpoint-mean interface Berry basis has rest-force rank four and nullity zero,
so it cannot simply be added while retaining the exact rest orbit.

**Expected symptoms:** A passing native launch must create signed bond current
and nonzero receiving-site support, preserve the identifiable eight-site
relations as the envelope translates, repeat in both signs of all three axes,
and agree with the independently derived component-selective Dirac current.

**Root cause:** The stabilizing C19-gradient Berry term is activated by the
product of the endpoint `Psi` activities. That product is zero on every
carrier--quiet bond. At the retained rest geometry the carrier wedge `W` is
parallel to the vacuum `chi` direction, so the canonical wedge force on
receiver `Psi` also cancels. Meanwhile the canonical receiver-`chi` variation
contains local `Psi` and vanishes before `Psi` arrives. This creates a
chicken-and-egg invariant boundary: neither native field can be first to
orient the quiet receiver under the tested compact-support launches.

**Corrective action:** Do not tune the unit stabilizer or preload a receiver.
First replace the rejected global phase ramp with the already-derived
component-selective four-spinor tangent and test a freely solved moving branch
with its self-consistent small exterior tail; the Dirac comparator requires
opposite signed `gamma` channels, not rigid or common-phase motion. If that
branch does not close, enumerate a rest-compatible C19 triangle/joint
`Psi`--`chi` connection whose `chi` variation remains nonzero at
`Psi_receiver=0`. It must retain the stable cube-incidence spectrum, preserve
both copies without projection, derive its coefficient natively, and pass the
same nonlinear movement/current gates. V4-006 remains open; canonical V40 is
unchanged.

## Component-selective Dirac-tail RCA (2026-11-23)

**Issue:** Replacing the rejected compact common-phase launch with the proper
translation-equivariant four-component envelope improves the native mismatch
but does not produce a robust invariant two-layer carrier.

**Why it matters:** Massive Dirac transport acts on four signed internal
channels and generally carries a spatial tail. V4-006 can close only if one
and the same recoverable native embedding supplies those components, their
time evolution, and their current.

**Symptoms:** Retaining all 125 translated centers lowers the configuration
cross-block to `0.00925131` for both copies. Their complete native spatial
images coincide to projector defect `1.93e-10`. Selecting exact invariant
configuration modes closes the spatial cross-block below `6.5e-15`, but the
projection back to the intended carrier has minimum singular value
`5.25e-9` and condition number `1.90e8`; its two-layer cross-block remains
`0.00551294` with rank-58 residual. A direct Riccati graph attempt is better
conditioned (`0.0179` minimum projection singular value) and reduces the
two-layer mismatch to `0.000163095`, but stalls with configuration cross-block
`0.00807833` and a rank-55 time residual.

**Expected symptoms:** A passing embedding must be injective and
well-conditioned, invariant under both committed-layer maps, keep both
four-component copies explicit, and compress to the same mass, three signed
transport generators, density, and current with a vanishing continuum
remainder.

**Root cause:** The four-component coordinates are not a configuration-only
subspace of `Psi` and `chi`. The stable action's gyro/time-layer response
generates additional phase information, predominantly an out-of-plane `Psi`
component. Configuration-only eigenmode selection can hide the mismatch only
by choosing a nearly singular sector. The missing state is therefore motion
information in the two-layer phase space, not another static halo amplitude.

**Corrective action:** Stop fitting configuration-only `chi/Psi` ratios.
Construct the phase-space embedding from current and previous native layers,
including the action-derived out-of-plane response, and solve its invariant
graph directly in the 3000-dimensional two-layer update. Require a
well-conditioned pullback and then derive the compressed mass, all three
transport generators and Noether current. Only if that phase-space graph
fails should the action be changed to a rest-compatible C19 triangle/joint
connection. V4-006 remains open and canonical V40 is unchanged.

## Same-image two-layer controllability RCA (2026-11-23)

**Issue:** The exact previous-layer motion on the fixed all-center
four-component spatial image cannot produce a self-reproducing native
first-order sector.

**Why it matters:** A second-order LFM system can reproduce Dirac only if the
current and previous layers form one closed first-order graph. Cancelling a
single update without reproducing that graph is not transport.

**Symptoms:** At a relative singular cutoff of `1e-4`, the memory cross block
has resolved rank `16` for a rank-`1000` carrier image. The bounded solution
leaves `0.999966665` of the original leakage and has graph-reproduction
defect `0.992110301`. More aggressive cutoffs use prior-layer norms from
`4.59e7` to `2.22e9` and produce reproduction defects from `1.00e10` to
`1.85e13`. A direct action ablation shows zero resolved control without the
density connection and the same rank `16` with it; the unit cube-incidence
completion adds no resolved memory direction at this boundary.

**Expected symptoms:** A passing graph must be full rank, well-conditioned,
cancel the normal response, and satisfy the same graph equation after the
next update without a fitted Dirac generator.

**Root cause:** The action's bounded previous-layer response reaches only a
small density-oriented subspace that is nearly orthogonal to the
out-of-plane normal response created by the current layer. The stabilizing
cube-incidence term repairs the rest spectrum, but its activity structure
does not supply the missing carrier-to-quiet phase-space memory channel.

**Corrective action:** Retire same-spatial-image preparation as the V4-006
repair. First permit a well-conditioned phase-space embedding with a native
spatial tail and re-test invariance. If that fails, derive a rest-preserving
reciprocal C19 kinetic/gyroscopic connection whose variation has nonzero
overlap with the measured normal response; constrain it by the existing
stable rest spectrum and derive its coefficient from geometry or a native
degeneracy, never from the Dirac target. Canonical V40 remains unchanged and
V4-006 remains open.

## Extended-packet final-gate RCA (2026-11-23)

**Issue:** The corrected extended joint `Psi`--`chi` packet passes tangent,
width, refined positive-frequency current and short nonlinear tests, but does
not yet satisfy the full V4-006 acceptance contract.

**Why it matters:** V4-006 claims a freely moving localized branch, not merely
a local tangent that resembles Dirac for a few updates. Promotion requires
long movement, box convergence, the same-state current and a formal bridge.

**Symptoms:** Timestep refinement reaches `2.46562e-4` state error at `dt/4`;
the width screen remains below `3.37317e-4`; refined density/current errors
are `5.46154e-5` and `0.00308875`; and four nonlinear updates remain within
`0.0254437` of the independent lift with `0.000664285` leakage. However, the
raw phase-charge contraction has `0.00606744` relative conservation defect,
the nonlinear run is only four updates, and the action implementation fixes
the box side to five.

**Expected symptoms:** A pass must show the same unrestricted packet complete
two cell translations in both directions, retain the eight-corner internal
fiber for 100 moving cycles and perturbations, converge with timestep and box
size, and carry the action-derived local current that limits to the same
Dirac current.

**Root cause:** The former physical misunderstanding is repaired: the
eight-corner object is an internal fiber of an extended packet, not a compact
cube that must jump into vacuum. The remaining failure is an implementation
and proof-depth boundary. The nonlinear action is hard-coded to side five,
the long translation time has not been integrated, and the nonlinear local
Noether charge has not been pulled through the positive-frequency readout.

**Corrective action:** Generalize every research-action term to arbitrary
periodic side without changing its local formula; reproduce the side-five
action/hash controls; run box and timestep refinement; then evolve the same
packet through two translations and 100 cycles while deriving its local
Noether continuity identity. Only after those pass should the localized
motion bridge be formalized in Lean. V4-006 remains open; canonical V40 and
the research action are unchanged.

## Long-time extended-packet sector RCA (2026-11-23)

**Issue:** The extended packet matches the discrete Dirac comparator for an
infinitesimal update but leaves that carrier sector over one internal cycle.

**Why it matters:** V4-006 requires one freely evolving native state to carry
the rest algebra, movement, density, and current. A local tangent match that
does not reproduce itself is not a moving Dirac electron.

**Symptoms:** The action was generalized without formula changes and exactly
reproduces the side-five action. Rest DEL RMS is below `3.36e-14` on sides
five through seven. The normalized packet DEL residual converges from
`0.00108863` to `0.00108233` on sides five through nine. Nevertheless the
phase-aligned packet error is about `0.63` after one cycle and `2.356` after
ten cycles. The ten-cycle result is unchanged at `dt/8` and `dt/16`, and a
native first-step history does not repair it. After one cycle, `99.99997%` of
the defect power is in `Psi`, `98.2235%` is on carrier sites, and only
`3.346%` lies in the ordinary phase/translation symmetry subspace. Scanning
the two existing connection coefficients can reduce the one-cycle error only
to `0.17018`; larger values become unstable.

**Expected symptoms:** A passing native sector must retain the same
four-component packet under repeated unrestricted updates, with bounded norm,
shrinking timestep and box errors, and the same current as the independent
discrete Dirac comparator.

**Root cause:** The short-time readout is tangent-correct but is not an
invariant phase-space embedding of the current action. Native evolution
drives additional in-plane `Psi` carrier directions that are neither a common
phase nor translation. This is not a boundary artifact, timestep error,
simple launch-history error, or delayed `chi` orientation failure. The
existing density and cube-incidence connection span lacks the response needed
to cancel the secular channel while preserving the rest orbit.

A final response-rank screen included those two connections plus the already
derived joint phase-lock, relative and area bilinear phase, and common and
relative density-oriented pair invariants. Their endpoint response has rank
six, but even a Dirac-fitted linear combination leaves `0.28103` relative
one-cycle error. Because that fit itself is inadmissible, the result is a
stronger scoped obstruction: the existing reciprocal action vocabulary does
not span the missing response.

**Corrective action:** Stop long nonlinear runs and coefficient tuning. Use
the measured in-plane carrier residual as an inverse requirement, then derive
the smallest new local reciprocal action invariant whose Hessian spans it.
Require exact rest preservation, positive energy, both copies, all-axis
short-time Dirac transport, and Noether-current closure before any endurance
run. No residual force, projector, Dirac matrix, or fitted coefficient may be
installed. Until such an action response is derived, V4-006 is structurally
open and the current research action is not promotable.

## Moving-substrate hypothesis RCA (2026-11-23)

**Issue:** A lattice or substrate drift was proposed as the missing stabilizer
for a carrier that is relatively stationary in its terrestrial environment.

**Why it matters:** If the current action already contained a freely running
substrate branch, the failed fixed-background packet test might have used the
wrong native state rather than the wrong action response.

**Symptoms:** Symmetric translation velocities in all three axes and separate
`Psi` and `chi` phase velocities were added to the exact rest orbit. Their
nonlinear DEL residuals are nonzero and scale linearly with speed; the minimum
tested translation residual is `0.00183679`. In the full linear tangent
recurrence, independently evolving each background drift and subtracting it
from the combined evolution changes the packet by at most `6.78e-10` over ten
cycles.

**Expected symptoms:** A native moving substrate would be an exact or
continuable nonlinear orbit with vanishing DEL residual, and linearization
about that new orbit would change the packet Hessian rather than merely add an
independent perturbation.

**Root cause:** The lattice labels are the substrate coordinates; assigning
them a coordinate velocity is not a new physical state. Simple rest-orbit
translation and phase velocities are not solutions of the current nonlinear
action. At tangent order they superpose independently, so they cannot alter
the carrier defect.

**Corrective action:** Do not add an imposed drift or reinterpret coordinates
as a stabilizing background. A remaining version of the hypothesis requires
solving a distinct finite-amplitude running `Psi`--`chi` branch from the same
action, then linearizing around it. Such a branch must carry its own energy
and current, remain stable for both velocity signs and multiple magnitudes,
and improve the same-carrier Dirac residual without being held. The simple
moving-substrate repair is rejected; the finite-amplitude branch question is
open.

## Complete C19 Hodge-connection RCA (2026-11-24)

**Issue:** The complete scalar--vector first-order C19 connection exists and
has a compatible deformed rest branch, but it does not close the one-cycle
localized Dirac-motion residual.

**Why it matters:** The prior inverse audit required an odd-in-momentum,
spatial-temporal response derived from the complete local `Psi`--`chi` frame.
This is the smallest rotationally covariant candidate in that class. If it
closed the residual with a native coefficient, it would provide the missing
action-level transport mechanism without a Dirac matrix or selected track.

**Symptoms:** For `tau=(chi dot Psi)/chi0` and
`W=(chi cross Psi)/chi0`, the native C19 operator
`D(tau,W)=(-div19 W, grad19 tau+curl19 W)` satisfies
`D^2=-sum_a G_a^2` with relative errors below `2.0e-16`. None of its four
one-forms preserves the old orbit exactly. The activated-relative form is
Fredholm compatible: its branch equation residual is `2.27e-10`, its state
tangent is `6.892%` of the rest norm per unit coefficient, and the directly
evaluated nonlinear stationarity residual converges at order `1.999996`.
Its motion response raises the complete response rank from nine to ten. Alone
it lowers the endpoint error from `0.587366` to `0.468565`; in the full prior
span it lowers `0.1792865` only to `0.1788594`. The inverse coefficient is not
an admitted constant.

**Expected symptoms:** A closing term must have a geometry-selected
coefficient, retain a finite freely solved rest orbit, and reduce the same
fixed-readout endpoint residual below the declared gate while preserving the
rest, current, copy, and all-axis conditions.

**Root cause:** The old action vocabulary did lack an independent first-order
odd spatial response, and the Hodge branch supplies it, but that channel is
only one component of the unresolved endpoint defect. More importantly, the
candidate's natural four-component object is `(tau,W)`, whereas the inverse
endpoint test still uses the historical corner-spinor embedding. The
remaining residual therefore cannot yet be classified as an action defect:
it may contain a measurement-bridge mismatch between two inequivalent
four-component coordinates. Fitting the residual into the action would be
circular.

**Corrective action:** Freeze the action and coefficient. Derive the
four-component phase-space readout induced by `(tau,W)` and its velocities at
the same localized orbit, prove its relation (or nonrelation) to the retained
corner embedding, and rerun mass, three-axis transport, density, and current
with that readout. If the two embeddings are equivalent and the residual
persists, resume action search using the residual orthogonal to the ten-term
span. If they are inequivalent, classify which readout is selected by the
native symplectic form before any action admission. V4-006 remains open and
no governing action changed.

The requested readout test sharpens this result. Density activation confines
the differential `(tau,W)` image to rank `32` and leaves `93.54%` of its Hodge
response outside that image; it turns the connection off on receiving sites.
Removing the activation raises configuration rank to `500` with condition
number `1.037`, but still leaves about `80.05%` Hodge cross-block. Adding the
previous layer restores the full `1000` real directions for both copies, so
the omitted half is native velocity information, not a missing field. A
single uniform positive-frequency combination returns rank `500`, as expected
for one energy grade; the four-component Dirac readout must assemble both
native grades through the independently derived beta grading. The full
phase-space map is not Hodge-invariant, so this assembly has not yet been
proved. The unactivated Hodge action branch is independently material but
also insufficient: its full-span inverse residual is `0.1783271`. These
results replace the tentative action-versus-readout ambiguity with a specific
open obligation: derive the native frequency grading on the Hodge phase space
before any further action repair.

## Complete cubical grading and fixed-center comparator RCA (2026-11-24)

**Issue:** The reduced `(tau,W)` Hodge chart cannot supply the Dirac mass
grading, and the fixed-center all-envelope comparator is not an exact motion
test for the localized carrier.

**Why it matters:** V4-006 requires one native carrier to retain the same mass,
three transport generators, current, and identity while translating. A
readout that lacks beta can never meet that gate, while a comparator already
wrong at zero momentum cannot identify the missing transport force.

**Symptoms:** The projector-derived beta is an involution but has Hodge
anticommutator defect `1.2` and retains `0.424264` of transport inside each
putative grade. Direct linear algebra is exact: the four-complex-dimensional
scalar/vector Hodge generators have oriented volume `i I` and beta
anticommutant dimension zero. In contrast, the complete eight cube cochains
are exactly wedge-plus-contraction on degrees 0/1/2/3. They reproduce the
existing corner generators with zero defect, corner parity anticommutes with
all three axes with zero defect, and the flavor involution splits them into
two closed rank-four copies with zero leakage. When the current action is
compressed through translated copies of the fixed localized frame, however,
the one-step error is already `0.02425` to `0.04779` at zero momentum and
mixes `0.0633` to `0.0765` of the recovered field out of the zero mode.
Subtracting that rest response does not repair the finite-momentum map; its
worst relative transport-increment error is `13.80`.

**Expected symptoms:** A valid native movement test must reproduce the exact
rest update at zero momentum, preserve the two rank-four cube copies, and
then add the signed C19 transport response without mixing unrelated momenta.
For a localized object this must be expressed as a finite translated orbit or
collective-coordinate family, not as Fourier modes about one fixed center.

**Root cause:** The `(tau,W)` reduction identifies the two-form and three-form
information with one scalar/vector chart and thereby selects only one spatial
chirality. No complex-linear beta exists on that chart. The full cube geometry
already contains the missing parity grade; it was discarded by the reduction,
not absent from the native fields. Separately, the all-center Fourier test
superposes translated carrier frames while the action Hessian is evaluated at
one localized rest carrier. That Hessian is spatially inhomogeneous, so
momentum is not an exact label and the test imports a rest mismatch before it
measures movement. The exact rest orbit's previously measured absence of a
translation zero mode confirms that infinitesimal Fourier transport is not
the correct localized branch.

**Corrective action:** Retire configuration-projector beta and fixed-center
Fourier mismatch as action-admission criteria. Keep the exact complete
eight-cochain Clifford grading. Use it on a finite joint `Psi`--`chi`
relative-periodic ansatz whose endpoint is the integer-translated carrier and
whose translation time, phases, previous layer, and surrounding field are
solved from the unchanged action. If no orbit converges, decompose the finite
orbit residual by cochain degree, field, site, and action term. Only then may
the smallest local reciprocal action invariant spanning the missing degree be
derived. V4-006 remains open; no action or equation changed in this audit.

## Finite relative-periodic handoff RCA (2026-11-24)

**Issue:** The complete eight-cochain carrier reaches the exactly translated
endpoint under a smooth diagnostic path, but that path is not a freely
evolving orbit of the current action.

**Why it matters:** This is the first movement test that simultaneously uses
the correct complete cube grading, an exact one-cell endpoint, live `Psi` and
`chi`, both time layers, and the nonlinear action. It distinguishes an
instantaneous-snap error from a genuine moving-profile requirement.

**Symptoms:** A literal one-cell translation in one update has best nonlinear
DEL norm `6259.62` relative to the state norm. Replacing the snap by an exact
periodic spectral translation over 64 updates reduces the best maximum
residual to `4.22422`; the endpoint agrees with the integer lattice shift to
`1.86e-16`. Slowing the path from speed `1` to `0.5` and `0.25` lowers its
best residual from approximately `15.4` to `7.72` and `4.22`. At the common
rest layer, reversal decomposition gives an even residual range
`3.85036`--`4.42475` at speed `0.25`, versus an odd range
`0.284625`--`0.338593`. The even residual is almost entirely `chi` and has
correlation above `0.999996` with the canonical V40 action contribution. The
density-oriented connection dominates the smaller odd response; the
cube-incidence symplectic contribution is two orders of magnitude smaller.

**Expected symptoms:** A freely moving carrier must make the full DEL residual
vanish at every intermediate layer while arriving at the exact translated
endpoint. Its complete cochain beta/copy decomposition must remain closed,
and the same orbit must repeat without reset.

**Root cause:** Gradual handoff repairs the impossible instantaneous
rearrangement, but the diagnostic path translates the resting `chi` mold
rigidly. A moving localized solution generally cannot retain its rest profile:
the canonical `chi` inertia requires a velocity-dependent deformation and/or
traveling tail whose spatial force balances the moving-frame acceleration.
The missing balance is mainly even under direction reversal, so adding a
purely orientation-odd transport term would address the wrong dominant
channel. This is a preparation/orbit-shape obstruction until an unrestricted
path solve is attempted, not yet proof that the action must change.

**Corrective action:** Use the exact fractional path as the seed for an
unrestricted relative-periodic boundary-value solve. Allow all intermediate
`Psi`, `chi`, and first-shell values to deform, while constraining only native
charge, phase gauges, periodic translation, and the two committed endpoint
relations. Measure the derived moving `chi` profile and repeat it for both
directions and axes. Only if the unrestricted residual cannot close should
its even `chi` component be used to enumerate a new local reciprocal action
class. No action changed in this screen; V4-006 remains open.

## Inverse-Dirac frame-transport RCA (2026-11-24)

**Issue:** The current carrier can reproduce the first discrete-Dirac motion
increment, but its four-component frame does not remain synchronized across
successive C19 centers, so the agreement is lost within one internal cycle.

**Why it matters:** V4-006 requires the same native state and one frozen
readout to supply rest mass, all three transport directions, density, and
current. A first-step match followed by frame drift is not emergent Dirac
motion and cannot be repaired by redefining the readout during evolution.

**Symptoms:** Alternating beta parity gives first-step error `4.97e-6` but
one-cycle phase-aligned error `0.270390`. A local spin-diagonal Clifford frame
improves those values to `4.56e-6` and `0.217340`, about 65% below the prior
approximately `0.63` one-cycle shape error. The simplest exact two-copy
exchange instead gives `0.574070`. A native overlap synchronization gives
`0.5026` and nearly singular local blocks. An expanded 32-direction local
position-action basis leaves rest-preserving relative residual `0.249661`;
its remaining response is almost entirely carrier-localized in-plane `Psi`
amplitude/phase.

**Expected symptoms:** One native, geometry-derived local frame must make the
first-step error and the one-cycle error converge together, retain both copies,
and work for face, edge, both movement signs, and held-out directions without
refitting or runtime support.

**Root cause:** Dirac transport is not a scalar displacement. It couples a
signed neighbor transfer to an internal Clifford-frame rotation. The current
action can launch the correct short-time transfer, but the tested native
position-only terms do not define the required full-rank phase-space
connection at the receiving center. Face bonds reverse simple checkerboard
parity while C19 edge bonds preserve it; a two-color preparation therefore
cannot represent both with one beta flip. The hand-chosen spin-diagonal frame
approximates the missing connection, but the unbiased `Psi`--`chi` overlap
construction is rank deficient and cannot derive it. The mismatch then
accumulates as an in-plane amplitude/phase defect.

**Corrective action:** Do not add the successful spin frame as a readout or
force. Enumerate the smallest real, local, reciprocal field-dependent
kinetic/symplectic `Psi`--`chi` action class built from the complete native
frame (`chi^dagger Psi`, `chi wedge Psi`, and C19 neighbors). Impose zero
rest-orbit first variation, positivity/nondegeneracy, both reciprocal field
equations, cubic symmetry, and both-copy retention. Rank-test that admissible
class against the measured one-cycle residual before choosing any coefficient.
If the response is absent, retire the class. If present, derive the coefficient
from geometry or a native degeneracy condition, re-solve the rest carrier, and
retest the frozen discrete-Dirac comparator. Canonical V40 remains unchanged
and V4-006 remains open.

## Movement-action completion RCA (2026-11-24)

**Issue:** Newly authorized reciprocal movement terms improve parts of the
Dirac packet response but do not preserve and translate the complete localized
carrier without increased complementary-sector leakage.

**Why it matters:** V4-006 requires one action to preserve rest, move the same
eight-site identity, and carry its Dirac density/current together. A lower
shape residual with more leakage is not a completed electron handoff.

**Symptoms:** The exact complete-frame action leaves the rest DEL at
`3.59e-13` RMS. At one cycle it lowers shape error from `0.24154` to
`0.20711` and norm drift from `0.04795` to `0.02800`, but raises leakage from
`0.01631` to `0.02124`. At ten cycles shape and norm remain better than the
incumbent (`1.28501` versus `1.42102`, `0.21729` versus `0.44868`) while
leakage worsens (`0.07810` versus `0.05069`). A C19 triangle-current action
whose quadratic current has the correct continuum limit is incompatible with
this localized carrier and produces large long-time growth.

**Expected symptoms:** A passing term lowers shape error, norm drift, and
leakage together, preserves the exact rest orbit, gives the correct signed
current in every axis, and reproduces the same carrier after two integer
handoffs without a reset.

**Root cause:** The cube-symplectic null direction supplies some missing signed
orientation response, while the curl--overlap square keeps the parallel and
transverse live-frame channels coherent. Their response is not confined to
the retained four-component carrier. The remaining missing structure is a
native invariant low-rank routing channel: the action must direct the signed
complete-frame response into the carrier sector rather than into the full
high-rank cube-incidence response. The extended-wave triangle current has the
right continuum current but the wrong localized sector.

**Corrective action:** Preserve the prior stable action and the rejected
candidate artifacts. Before another nonlinear action trial, derive the
rest-null response basis on the complete first-shell phase space and require a
single cubic-covariant action invariant whose Hessian has rank four on the
carrier tangent and zero cross block into its complement. Only then form its
reciprocal Psi and chi equations and rerun rest, current, two-handoff, and
long-time gates. Canonical V40 is unchanged; no research action is admitted;
V4-006 remains open.

## Ten receiver-orientation mechanisms RCA (2026-11-24)

**Issue:** None of the ten scoped receiver-orientation implementations closes
the same-carrier movement gate under free native evolution.

**Why it matters:** These mechanisms covered the leading alternatives from
wave transport and coupled-system research: native frame transport,
coin-and-shift, nonlinear coherent hopping, zero-mode restoration, traveling
tails, excitable `chi`, adiabatic transfer, phase entrainment, autonomous
pumping, and two-copy relaying. A winner had to preserve the resting carrier
and match the frozen discrete-Dirac motion comparator without runtime support.

**Symptoms:** The best spin-diagonal launch has first-step error `4.56e-6`,
but the derived cube-incidence connection retains best one-cycle error
`0.217340` and held-direction error `0.534444`. Thirty-two position-only
coherent-hopping directions leave rest-preserving residual `0.249661`. Native
translation curvature remains `0.218`--`0.220`. The 32-update deformed-tail
solve reaches boundary residual `0.047596`, but `Psi` fidelity is `0.566977`,
containment `0.624086`, and the unreset second-cell residual is `2.47070`.
The bistable `chi` and phase-lock actions break the rest/spectrum gates. The
rest rotor produces receiving force only `2.57e-15` and no resolved phase
current, so the tested autonomous dark handoff does not start. A simple
internal cycle has no center shift or directed launch current. Exact copy
exchange leaves one-cycle error `0.574072`.

**Expected symptoms:** At least one mechanism must retain high carrier
fidelity and containment, produce the same signed response in both movement
directions and held-out axes, repeat an unreset second handoff, and lower both
first-step and one-cycle discrete-Dirac errors with one native action and one
frozen readout.

**Root cause:** The receiver needs a full-rank, direction-sensitive
phase-space relation between the leading field and all four retained internal
components. The present action either supplies only a short-time orientation
increment, supplies a scalar or rank-deficient phase relation, or deforms the
carrier while reducing the boundary residual. Simple internal rotation is
exactly canceled at quiet receiving sites by the cubic neighbor geometry.
Position-only actions cannot span the remaining amplitude/phase response, and
the currently derived symplectic connection does not control all held
directions. This is a missing action-response rank, not a need for a stronger
coefficient in one of the tested terms.

**Corrective action:** Retire coefficient tuning inside the ten tested
implementations. Calculate the one-cycle residual in the complete native
phase-space frame, classify it under the cubic/C19 symmetry group, and derive
the smallest new real reciprocal kinetic action basis whose admissible
rest-preserving Hessian contains that representation. Require the response
rank before choosing a coefficient. If no such two-site basis has rank,
advance to the already-authorized three-site C19 common-neighbor kinetic class.
Do not preload the receiver, install the spin frame, or treat the failed solver
as a universal no-go. Canonical V40 remains unchanged; V4-006 remains open.

## No-intravoxel site-transfer RCA (2026-11-24)

**Issue:** Replacing rigid fractional carrier translation with literal
site-amplitude transfer does not by itself close the one-cycle motion gate.

**Why it matters:** A cubic substrate need not contain positions inside a
voxel. Apparent continuous motion may emerge from amplitudes transferring
between discrete sites, as image motion emerges from changing pixels. The
test therefore removes an unnecessarily strong compact-shape requirement.

**Symptoms:** The direct `Psi` site-density comparison remains below five
percent only for about `0.20`--`0.31` internal cycles across the two Clifford
copies and tested axial/diagonal directions. At one cycle the worst
C19-averaged site-density error is `0.425752` and the worst site-density-center
shift error is `0.418338`. The separate tangent Noether readout is also open,
with worst averaged-density error `0.539292`, center error `0.479902`, and
charge drift `1.20544`; consequently it is not used as the sole motion
criterion. In a controlled plus-copy x-positive ablation, the bare existing
action gives maximum site-density error `0.310854`. Adding only the reciprocal
density connection improves this to `0.219796` and cuts the center error from
`0.195944` to `0.094242`. Adding only the cube-incidence symplectic completion
worsens the density error to `0.554701`; combining it with the density
connection returns the error to `0.309907`.

**Expected symptoms:** The independently evolved native and discrete-C19
Dirac packets should redistribute `Psi` density among the same lattice sites,
with their C19-averaged densities and center shifts remaining within declared
tolerances without assigning any fractional position to the carrier.

**Root cause:** The pixel analogy corrects the kinematic target but not the
action-response defect. The current packet starts with the correct tangent
and transfers amplitude between sites, then accumulates the previously
identified in-plane `Psi` amplitude/phase mode. The reciprocal density
connection supplies part of the needed site-transfer response. The current
cube-incidence timing completion is not complementary to it at one-cycle
packet scale and drives the packet away from the frozen Dirac comparator.
This is neither proof of intravoxel motion nor a receiver-`chi` orientation
failure.

**Corrective action:** Retain the extended packet and site-amplitude observable
as the V4-006 target. Do not return to fractional rigid snapshots. Decompose
the residual left by the density-only connection in the complete native
phase-space/cubic representation and derive a rest-preserving reciprocal
kinetic response that complements it. The cube-incidence completion must be
re-derived for this role or excluded from the motion candidate; its coefficient
must not be fitted to the Dirac endpoint. Recheck rest, both copies, all axes,
Noether continuity, and positive energy before endurance tests. No action was
changed or admitted; canonical V40 is unchanged and V4-006 remains open.

## Common/relative site-transfer information RCA (2026-11-24)

**Issue:** The native conjugate/relative `Psi`--`chi` phase channel materially
improves site transfer, but neither it nor the remaining scalar/anisotropic
two-site information channels closes V4-006.

**Why it matters:** The no-intravoxel target requires neighboring cubes to
transfer the complete internal carrier state, not merely its total density.
The common pair connection communicates the co-rotating relation
`chi tensor Psi`; the relative pair connection communicates the independent
counter-rotating relation `chi tensor conjugate(Psi)`.

**Symptoms:** With the common connection alone, the worst x-positive error
over both Clifford copies is `0.361178`. Adding the relative channel reduces
the minimax error monotonically through the stable scan: `0.111` near factor
`12`, `0.102526` at the geometric `chi0=19` factor, and `0.097991` at factor
`32`. Factors `48` and above generate non-finite evolution, so gain cannot be
increased indefinitely. The factor-32 held axial/diagonal test still has
worst density error `0.137642` and center error `0.136180`. Around the
geometric factor-19 incumbent, unit-sign additions of `rho_chi`, area density,
`Psi` anisotropy, `chi` anisotropy and area-bilinear anisotropy do not provide
a material repair: the best training error is `0.100891` and worst held error
remains `0.137643`.

**Expected symptoms:** One local reciprocal connection should preserve both
copies and keep the literal C19-averaged `Psi` site density and center motion
within tolerance for both movement signs and a held diagonal, without a
copy selector or fitted gain.

**Root cause:** The present links transmit quantity and both common and
relative phase relations, but contract the internal C3 tensor before coupling
it to the C19 bond direction. The receiver therefore learns that a structured
wave is arriving, but not a full-rank rule identifying which internal prong
continues along which spatial bond. Scalar and aggregate anisotropy channels
cannot supply that missing component-to-direction map. The prior native
joint-projector frame connection independently has rank two and misses one
physical direction, consistent with this diagnosis.

**Corrective action:** Derive the lowest-degree bond-direction-resolved native
frame connection. It must build a complete local frame from `chi`, the part
of `Psi` orthogonal to `chi`, and `W=chi wedge Psi`, contract that frame with
each C19 bond vector, and enter one real reciprocal positive kinetic action.
Before any coefficient is chosen, rank-test its rest-preserving Hessian against
the measured residual and require all three spatial directions, both copies,
and positive mass. A two-site failure advances to the C19 common-neighbor
triangle version; it does not justify a copy projector or Dirac matrix. No
action was changed or admitted; canonical V40 is unchanged and V4-006 remains
open.

## Bond-frame and C19-triangle information-transfer RCA (2026-11-24)

**Issue:** No admissible two-site live-frame construction closes the third
fixed-charge transport direction. The previously derived C19 face--edge
triangle action closes it, but its native-inertia branch still lacks a
resolved long-time stability and same-state Dirac-continuum proof.

**Why it matters:** V4-006 needs one native mechanism to preserve the rest
carrier and transfer all three components of its motion. A rank-two response
can never reproduce the three spatial transport generators, while a
rank-three linear response that grows nonlinearly cannot carry the electron
for the declared 100-cycle gate.

**Symptoms:** The ordered-frame rule `E_i n` initially appeared rank three,
but it is not cubic covariant: under rotation it gives `R E R n`, not
`R E n`. It is rejected as a hidden field-role/spatial-axis identification.
The covariant complete tensor
`Q=chi chi^dagger/R^2 + Psi Psi^dagger/R^2 + W W^dagger/R^4` preserves rest
when neighbors are summed before comparison, with DEL RMS `1.89447e-12`, but
has rank `[2,2,2]`. Comparing `Q_i n` bond by bond also has rank `[2,2,2]`
and perturbs rest, with DEL RMS `0.0353139`. By contrast, the C19 face--edge
triangle action reproduces the earlier body-gyro action and first variation
to below `1e-16`, preserves rest with DEL RMS `1.89740e-12`, and has
fixed-charge ranks `[3,3,3]`, with relative third singular values `0.04539`,
`0.04521`, and `0.09051`. Its 0.1% perturbation runs three cycles with maximum
phase-aligned shape residual `0.00246368`, while a 1% nonlinear run fails
before one cycle. The radius `1.00294963` belongs to the symmetric center of
the rest-force coefficient family, not the native-inertia normalization. The
native coefficients `c=(a+B/4,a-B/4,-a/2,0)` instead give radii about
`1+5e-8` over four timestep refinements. Those offsets do not converge cleanly
enough to certify stability and remain near the double-precision reciprocal
pairing floor. A numerical generalized-unit quotient still leaves a 100-cycle
translation-velocity amplification dominated (`98.5%`) by exterior `chi`.

**Expected symptoms:** A passing action must retain the triangle mechanism's
rank-three all-axis response and exact rest cancellation, resolve the
near-unit spectrum with a structured symmetry/family quotient, and eliminate
any surviving physical exterior-`chi` drift. It must then pass the same-state
discrete-C19 Dirac current, two-handoff, perturbation, timestep, box, and
100-cycle gates.

**Root cause:** Two-site phase-invariant projector data splits into the
`Psi`--`chi` plane and its normal. Adding the normal projector lets a pair
measure that direction but does not mix it into the transport response.
Handed transport first appears through an oriented three-site path. The C19
triangle exchanges `W=chi wedge Psi` together with the common area velocity
`chi wedge Psi_dot + chi_dot wedge Psi` and the relative area velocity
`chi wedge Psi_dot - chi_dot wedge Psi`. That supplies the missing rank, but
its current coefficient combination also excites a carrier-phase/exterior-
`chi` transverse mode; the leading finite obstruction is about 82% `chi`,
mostly transverse, and about 78% of that `chi` residual lies outside the
eight carrier sites.

**Corrective action:** Keep the exact C19 triangle factorization as the
incumbent information channel and retire further two-site frame tuning. First
replace the numerical SVD quotient with independently derived phase, charge,
frequency, translation, and nearby-relative-equilibrium tangents and repeat
the spectrum at higher precision. If exterior-`chi` growth survives that
physical quotient, use its measured mode to derive the admissible nullspace
of additional three-site reciprocal kinetic invariants that vanish on the
rest orbit and on the already-correct transport tangent. Do not fit a
coefficient to the Dirac endpoint and do not install a held frame. No action
is admitted; canonical V40 is unchanged and V4-006 remains open.

## C19-triangle continuum-symbol RCA (2026-11-24)

**Issue:** The rank-three C19 triangle channel had not been checked against
the low-momentum continuum symbol required by the Dirac factorization.

**Why it matters:** Three independent lattice responses do not by themselves
establish Dirac transport. Their second-order spatial symbol must approach an
isotropic multiple of `|k|^2`, and the discrete gyro normalization must have
a finite continuum limit before the same operator can be square-rooted.

**Symptoms:** The exact triangle/body symbol is
`8[1-cos(kx)cos(ky)cos(kz)]`. Across axis, face-diagonal and body-diagonal
directions its normalized relative remainder converges with fitted powers
`1.99829`, `1.99657`, and `1.99503`. At `|k|=0.025` the directional spread is
`6.94446e-5`. More decisively, both pre-existing four-component cubical
copies obey the Clifford relations to numerical precision, so their exact
C19 spatial square is `4|s19(k)|^2 I4`. The triangle symbol approaches this
same square with relative-error powers `2.00864`, `2.00429`, and `2.00633`.
Their massive group velocities approach one another with powers `2.02193`,
`2.01526`, and `2.01842`; at the finest sampled momentum their relative
errors are `3.12581e-4`, `2.08371e-4`, and `1.73649e-4`. The discrete
coefficient `a=2 dt/sin(chi0 dt)` approaches `2/chi0` with relative-error
power `2.00083`.

**Expected symptoms:** The symbol must equal `4|k|^2+O(|k|^4)` isotropically,
match the square of the independently derived C19 Dirac symbol through the
same order, yield the same low-momentum group velocity, and have coefficient
`2/chi0+O(dt^2)`.

**Root cause:** The C19 face--edge triangle is an exact factorization of a
cubic endpoint correlation. Cubic symmetry forces its leading nonconstant
symbol to be the isotropic quadratic term; anisotropy first appears in the
quartic lattice remainder.

**Corrective action:** Mark the triangle transport as passing the spatial
Dirac-square, dispersion, and group-velocity continuum gates, not as a
completed Dirac derivation. The cubical alpha/beta readout was not inserted
into the action. Next prove that the full live-Psi/live-chi triangle tangent
preserves that same four-component sector and compare the action-derived
Noether current with its Dirac current. Only that common-state calculation
can close the equation map. V4-006 remains open.

## C19-triangle same-state first-order RCA (2026-11-24)

**Issue:** The symmetric C19 triangle candidate reproduces the Dirac spatial
square and quadratic current in the continuum, but it does not derive the
first-order `alpha dot k` generator needed to close V4-006.

**Why it matters:** Matching a squared dispersion relation is insufficient.
The first-order Dirac equation also specifies the sign and component routing
of transport. A second-order action can have the right square while selecting
the wrong square root or no native square root at all.

**Symptoms:** A coordinate audit found that the earlier component-ramp helper
mixed ordinary action coordinates with the mass-normalized diagnostic unpack,
shrinking `chi` by `1/sqrt(B)`. With consistent physical coordinates, the
triangle action still has fixed-charge ranks `[3,3,3]` at every finite tested
timestep. The exact target is inside its declared embedding to about `1e-15`.
Own-copy previous-layer leakage decreases as `0.336776`, `0.191635`,
`0.0995520`, `0.0502795`; combined-copy leakage decreases as `6.89518e-5`,
`3.81187e-5`, `1.96567e-5`, `9.90895e-6`, approaching first-order convergence.
However, the smallest fixed-charge singular responses in two ramp families
also decrease approximately linearly with `dt`, while the third remains
finite. The quadratic action-derived Noether current converges to the cubical
Dirac current with second-order relative error, but this follows from the
squared symbol. Analytically the present triangle correlation is even:
`8 cos(kx) cos(ky) cos(kz)`, with zero gradient at `k=0`. The required Dirac
symbol is odd, and each copy has nonzero derivative norm `4` in all axes.

**Expected symptoms:** The same live-field embedding must have a finite,
nondegenerate odd-in-momentum compressed generator in all three axes, with
the cubical alpha anticommutators, while its action-derived current and sector
leakage converge under timestep refinement.

**Root cause:** The exact triangle construction used so far is a symmetric
endpoint correlation written as an average of face--edge paths. It preserves
the information in the Dirac square but loses the orientation/sign information
that distinguishes its first-order roots. No momentum-independent change of
four-component basis can turn its zero first derivative into the required
nonzero alpha generators.

**Corrective action:** Retain the symmetric triangle as the squared-symbol
control, but do not admit or promote it. Derive a genuinely oriented C19
face--edge reciprocal one-form whose cubic sum is odd in momentum, whose
orientation comes only from live `Psi`, `chi`, and `W=chi wedge Psi`, and
whose first variation vanishes on the exact rest orbit. Rank-test its
compressed first derivative before any nonlinear run. The lowest pairwise
axial contraction was tested immediately and cancels identically after the
full cubic/translation sum, even on random fields. The previously tested
genuine Bargmann three-field loop does not cancel, but sends more than 99% of
its response outside the frozen carrier sector and mixes the copies. The next
candidate must therefore be a higher joint loop or a derived first-shell
extension of the embedding; neither failure justifies dropping cubic symmetry
or inserting a handed selector. V4-006 remains open.

## Bargmann first-shell embedding RCA (2026-11-24)

**Issue:** The nontrivial oriented Bargmann connection leaves more than 99%
of its response outside the frozen eight-component carrier tangent, and it
was unknown whether this response was the physical first-shell tail of a
moving localized carrier.

**Why it matters:** If the response were a translated first-shell tail, the
existing action could contain the missing motion and only the measurement
embedding would require repair. If it remains internal to the original cube,
adding spatial support cannot close V4-006.

**Symptoms:** The same four internal carrier amplitudes were translated over
the 19 C19 neighboring centers and over every center of the side-five box.
For each axis and each copy, the exact Bargmann Hessian image has carrier-core
power `1` and C19-first-shell power `0` to double precision. The combined
two-copy translated basis fits the response with worst residual `5.46e-16`,
but the correct single-copy comparison leaves worst residual `0.730628`.

**Expected symptoms:** A genuine moving tail would place nonzero response
power on first-shell sites and would be representable by translated copies of
the same four-component flavor without requiring the opposite copy or
unrelated internal core directions.

**Root cause:** At the exact localized rest orbit, the tested Bargmann loop's
linear response acts on internal field directions of the occupied carrier
sites. Its orientation information is not routed as a spatial tail, and its
compressed action mixes the two cubical copies. The apparent closure of the
combined translated basis is a dimension effect, not carrier transport.

**Corrective action:** Retire the first-shell extension for this Bargmann
candidate. Do not project its response back into the desired copy. Derive a
higher reciprocal oriented loop whose odd momentum derivative is closed on
one native copy, or derive a different joint copy embedding directly from the
full native operator and verify its dimension and current before any nonlinear
run. Canonical V40 is unchanged and V4-006 remains open.

## Bargmann fixed-copy rotation RCA (2026-11-24)

**Issue:** The Bargmann response could mix the two recorded Clifford copies
because the historical copy basis was chosen incorrectly, rather than because
the action creates genuinely additional internal motion.

**Why it matters:** If one fixed rotation of the existing eight abstract
components separated the response into two invariant four-component sectors,
the action candidate could be retained and only the readout would need repair.
If no such rotation exists, relabeling the copies cannot close V4-006.

**Symptoms:** Each individual axis maps the eight candidate inputs to an
eight-dimensional response. Taken together, however, the three axes span 24
independent native response directions. The common commutant of every
directional response Gram matrix has dimension one and is the identity to
relative residual `1.01e-15`. Its traceless symmetric rank is zero. The
historical copy grading has normalized commutator defects from `0.732` to
`1.463`, depending on the directional pair.

**Expected symptoms:** A fixed relabeling repair requires a nontrivial
symmetric grading in the common commutant, with two four-dimensional
eigenspaces, and a joint three-axis response rank no greater than eight.

**Root cause:** The tested Bargmann action routes the three transport
directions into mutually independent internal native responses. The response
does not preserve any common four-plus-four grading inside the existing eight
abstract component labels. The apparent mixing is therefore not merely a bad
choice of the two-copy basis.

**Corrective action:** Retire fixed rotations and copy relabeling for this
Bargmann candidate. Do not expand the spinor readout to 24 components merely
to fit its response. Use the 24-rank residual as the rejection control while
deriving a higher reciprocal oriented loop whose joint three-axis response
has native rank at most eight and contains a nontrivial geometry-derived
four-plus-four grading. Canonical V40 is unchanged and V4-006 remains open.

## Exact translation precursor and live dual-frame RCA (2026-11-24)

**Issue:** The exact same-assignment one-cell map is kinematically known, but
the native action must communicate the complete `Psi`--`chi` carrier to the
receiver and reproduce that translated layer without a copy rule.

**Why it matters:** A fitted receiver, a projected endpoint, or a small raw
action residual is not native electron movement. V4-006 requires the action's
own implicit update to yield the identical translated carrier.

**Symptoms:** With the eight core sites held exact, a canonical first-shell
precursor reduces the nonlinear recurrence residual to `0.675`--`0.706` in
all six signed axes but does not close. The historical live-frame chi action
and the paired Psi extension also fail. Replacing the approximate three-axis
projection by the exact regularized dual frame improves the response with a
small precursor. At the derived regularizer `kappa=1/63`, the bounded profile
residual remains `0.22694`; `82.14%` of its power is chi, `62.84%` is on the
carrier core, and `66.10%` is in component two. The implicit-next-layer LSMR
audit reaches its 200-iteration cap and already requires a correction equal
to `17.96%` of the exact translation step. Sending the regularizer toward zero
makes the action metric nearly singular and inflates its absolute residual;
its small relative residual is not a pass.

**Expected symptoms:** A passing native conveyor has a bounded positive
kinetic metric, a small freely solved precursor, a converged implicit update,
and next-layer correction below numerical tolerance while returning the
identical translated eight-site `Psi`--`chi` carrier.

**Root cause:** The existing bond-frame candidate transports chi velocity by
spatially averaging live-frame projections. The dual Gram map repairs
nonorthogonal-frame mixing, but the action still lacks a well-conditioned
make-before-break connection between the sender's committed frame and the
receiver's next-layer frame. Consequently the receiver is partially prepared
while carrier-core chi release and clock response remain unsynchronized.

**Corrective action:** Lock the exact endpoint in
`EXACT_TRANSLATION_MAP_LOCK.md`. Retain the dual-frame representation but do
not admit its present kinetic term. Derive a real reciprocal spacetime C19
bond connection activated by native signed bond current, with no selected
direction and exact rest reduction. Gate it first by the converged implicit
next-layer error on this exact map, then by two unreset handoffs in both signs
of every axis. Canonical V40 remains unchanged and V4-006 remains open.

## Direct six-prong communication RCA (2026-11-24)

**Issue:** The receiver must obtain the complete complex `Psi_x/y/z` and
`chi_x/y/z` pattern from native neighbor communication while the exact rest
orbit and locked same-assignment translated endpoint remain unchanged.

**Why it matters:** Communicating only density, a wedge frame, or an averaged
orientation discards phase-bearing information. Conversely, installing a
copy rule or fitted endpoint would make movement scaffolding rather than an
emergent update and cannot close V4-006.

**Symptoms:** Undirected direct-prong averaging, native-current-directed
averaging, and a density-oriented prong one-form all fail the exact update.
The unconstrained same-component position/velocity bond basis spans `96.48%`
of the required squared DEL response, but its near-closing fit uses negative
coefficients without provenance; the positive cone spans only `71.32%` and
is not admitted. A density-continuity material action preserves rest and
improves the handoff, but its cornerwise velocity is too large by approximately
`11/3` and has transverse components that cancel only globally. The full
complex-prong best-matching action removes the local common phase and lets all
real/imaginary `Psi` prongs determine a common local velocity which transports
both fields. It preserves rest and reduces the optimized x-positive profile
residual to `0.06987` with a precursor norm `0.04963`. Short equal-budget
signed-axis controls give residuals `0.0767`--`0.1033` in x/y and about
`0.249` in z. The implicit next-layer solve remains ill-conditioned: its DEL
residual stays `0.49`--`0.66`, so its small early correction estimates are not
accepted.

**Expected symptoms:** A passing componentwise connection preserves the rest
orbit, gives one cubic-covariant velocity from all six complex prongs, closes
the exact translated DEL equation with a converged implicit solve in all six
signed axes, and repeats two unreset handoffs without projection or preload.

**Root cause:** Density alone underdetermines a three-direction local velocity,
while the complete complex `Psi` prongs nearly determine it. The remaining
defect is now the anisotropic and near-null part of that best-matching kinetic
metric: the prepared carrier plane supplies weaker independent information for
one spatial direction, and the substituted least-squares velocity creates an
ill-conditioned next-layer Hessian. Adding `chi` to the velocity vote with
its inertia `B` worsens the profile; `chi` should be transported by the
`Psi`-derived connection rather than dominate its direction estimate.

**Corrective action:** Retain the exact phase-quotiented `Psi`-prong normal
equation as the current research incumbent, but do not admit it. Derive the
missing cubic completion from the first-shell complex `Psi` gradients (or an
equivalent Schur-complement formulation with velocity kept as an auxiliary
variational coordinate and eliminated only after the Hessian audit). Require
the completion to remove the z anisotropy and the metric null directions with
no fitted coefficient. Then rerun the converged exact-update, conservation,
rest-spectrum, two-handoff and discrete-Dirac-current gates. Canonical V40 is
unchanged and V4-006 remains open.

## Complete live-frame communication RCA (2026-11-24)

**Issue:** Direct same-component communication still fails to rebuild the
joint `Psi`--`chi` carrier on the receiving shell even though its common
velocity channel is linearly stable and has the correct axial sign.

**Why it matters:** V4-006 requires the receiver to recover the same complete
three-prong orientation and phase information, not merely accept density or
one oriented-area component.

**Symptoms:** At update two, `48.5808%` of the `Psi` error is parallel to live
`chi` and `51.4192%` is transverse.  A parallel-overlap material wave, the
missing positive parallel C19 bond, and a complete overlap-plus-area material
wave reduce selected ten-cycle shape errors from `1.42102` to `1.31284`,
`1.26279`, and `1.19937`, respectively.  None reduces the full Dirac departure
through update 64 or preserves the axial motion: the complete wave crosses the
error threshold one update earlier and reverses the final axial displacement.
`Psi`-only and density-completed velocity metrics are linearly unstable.
The signed connection-only completion remains `1.41914` at ten cycles because
the parallel overlap vanishes on the rest orbit and consequently supplies no
linear tangent response.

**Expected symptoms:** The same reciprocal action should preserve the exact
rest orbit and unit-circle tangent spectrum, keep axial current signed along
the launch direction, suppress transverse current, carry `Psi` and `chi`
together, and reproduce the same eight-site assignment at the first and second
integer handoffs.

**Root cause:** The receiver needs both live relative-frame invariants,
`chi dagger Psi` and `chi cross Psi`.  Existing and newly tested unsigned
positive squares can constrain those invariants but cannot determine which
neighbor is the sender.  They therefore trade deformation for leakage or
reverse the useful forward current.  The missing response is signed transfer
of the complete frame, not additional scalar stiffness or generic frame
inertia.

**Corrective action:** Derive a real reciprocal C19 spacetime bond term whose
sign comes from native bond current and whose transported payload is the full
pair (`chi dagger Psi`, `chi cross Psi`).  Require its first variation to span
both measured error directions while annihilating the exact rest orbit and the
already-correct longitudinal tangent.  Rank-test that condition before any
nonlinear tuning.  No tested candidate is promoted; canonical V40 is unchanged
and V4-006 remains open.

## Live-link kinetic communication RCA (2026-11-24)

**Issue:** The localized rest carrier is stable and its live edge phases
derive the full Clifford algebra, but the existing action does not transmit
the joint three-component `Psi`--`chi` frame into a quiet receiving region at
the rate and orientation required for a repeated localized handoff.

**Why it matters:** V4-006 requires the same nonlinear eight-site carrier to
move under the native action. A correct stationary factorization or a norm
arriving in a neighbor is insufficient unless the receiver reconstructs the
same four-corner complex pattern, live `chi`, native current and subsequent
handoff.

**Symptoms:** Scalar live-link candidates vanish at the carrier/exterior
interface because the two nonzero `chi` orientations are exactly orthogonal.
The normalized `chi` wedge stays nonzero there. A smooth squared-wedge kinetic
link produces the exact translated receiver `Psi` face to about `1e-15`,
preserves the rest orbit at `1.33e-12` DEL RMS, and passes eight small-amplitude
four-update controls with worst Dirac error below `0.997%` and leakage below
`0.0273%`. Completing it over all weighted C19 bonds preserves the positive
kinetic bound `[18/19,20/19]`, but the face-only projection then has overlap
`0.826` because edge bonds create a broader precursor. A forced
one-cell-per-update preparation fails by order one and has target DEL RMS
`1.36e6`; it demands speed 256 rather than the discrete-Dirac group speed
`0.01695` at the declared packet momentum.

**Expected symptoms:** A passing result keeps the exact rest orbit, evolves
both copies and both motion signs, carries `Psi` and `chi` through the same
positive C19 kinetic connection, follows the discrete Dirac current over the
derived crossing interval, reconstructs the identical translated eight-site
carrier at the endpoint, and repeats without projection, preload or reset.

**Root cause:** The prior action has no smooth phase-neutral velocity channel
that remains active when neighboring `chi` vectors are orthogonal. The new
squared-wedge link supplies that missing first receiver imprint. The remaining
open problem is no longer the first transmitted `Psi` pattern; it is
integrating the full C19 edge precursor and reciprocal `chi` response over the
physically correct massive-Dirac translation time. Treating numerical `dt` as
a cell-crossing tick incorrectly asks a massive packet to propagate four
orders of magnitude faster than its derived group velocity.

**Corrective action:** Retain the face-pair term only as a mechanism-isolation
control. Continue with the strictly positive normalized C19 joint metric at
`g=1/chi0`, using the independently derived discrete-Dirac group velocity to
set the observation interval, not the force. Expand the phase-space readout
to include the freely generated C19 precursor and solve the long implicit
trajectory with checkpointed conservation/current diagnostics. Admit the
action only if the same full-amplitude carrier completes and repeats the cell
handoff in both signs and all axes without worsening the stationary gates.
Canonical V40 is unchanged and V4-006 remains open.

## Six-face Laplacian transport RCA (2026-11-24)

**Issue:** The smooth six-face reciprocal kinetic connection reconstructs the
first receiving `Psi` face and nearly reconstructs the receiving `chi`, but it
does not preserve the compact moving four-component packet long enough to
complete a cell handoff.

**Why it matters:** V4-006 requires the same unrestricted eight-site joint
carrier to translate and repeat. A correct first receiver acceleration and a
short small-amplitude Dirac match are necessary but not sufficient.

**Symptoms:** The derived per-face coefficient `1/19` produces aggregate
incident strength `6/19`, preserves rest at `1.33e-12` DEL RMS, and passes all
eight four-update controls with worst Dirac error `0.9963%` and leakage
`0.0216%`. At update five the selected moving control first exceeds one
percent error and reaches `3.357%` by update nine, although leakage is only
`0.0854%`. The uniform zero-momentum control remains below one percent at
update nine; localization without momentum reaches `3.073%`. Increasing the
same positive face invariant by more than seven orders of magnitude leaves a
nonzero `2.235%` tangent shape-error floor.

**Expected symptoms:** The compact packet should retain one common
four-component readout while its Noether current crosses a face, arrive after
the independently derived crossing interval as the identical translated
eight-site `Psi`--`chi` pattern, and repeat with conserved energy and charge.

**Root cause:** The six-face action supplies the missing receiver orientation
and keeps the response almost entirely inside the four-component sector. The
remaining mismatch is therefore not absent neighbor communication. It is the
lossy representation of compact spatial gradients in the frozen collective
readout, plus a smaller within-sector transport/current defect. The single
face-Laplacian invariant has insufficient response rank to correct that block;
stiffening it cannot help.

**Corrective action:** Freeze the successful six-face term and its `6/19`
normalization as the current research control, not as canonical law. Compress
the exact DEL response of the compact packet through the same current and
previous-layer readout and derive the residual gradient/time/current block.
Search only native reciprocal invariants whose variation spans that block and
which annihilate the rest orbit and the already-correct first face response.
Retest the exact carrier endpoint, two repetitions, both copies and signed
axes before conservation, 100-cycle, refinement and Lean gates. V4-006
remains open; no canonical promotion follows.

## Frequency-sector correction and endurance RCA (2026-11-24)

**Issue:** The six-face candidate passes the corrected short moving gate but
does not preserve the Dirac packet through one internal cycle or the required
100-cycle endurance interval.

**Why it matters:** V4-006 requires the same unrestricted localized carrier,
four-component readout, action-derived current, and both Clifford copies to
remain mutually consistent during motion. A correct first receiver and four
short updates do not establish a traveling electron.

**Symptoms:** The proved rest orbit advances with positive complex phase, but
the old comparator selected the opposite native frequency sector. Selecting
the matching negative-frequency Dirac branch lowers the nine-update error to
`0.986520%` and makes the native Noether center move in the correct
longitudinal direction. All eight four-update cases pass below `0.235%` error.
At 32 updates the error reaches `6.27058%` while leakage is only `0.189386%`.
The exact one-cycle tangent has order-one accumulated shape/norm error. A
strength scan leaves a `~1.49%` asymptotic shape floor; the triangle addition
worsens the short result and the simple signed face term fails reflection
covariance.

**Expected symptoms:** With the frequency sector fixed once, the same action
should retain the four-component packet, move its Noether center with the
centered C19-Dirac current in both signs and copies, and keep the error bounded
and convergent through the declared translation and 100-cycle gates without
readout refitting.

**Root cause:** Two causes had been conflated. The apparent backward current
was a comparator branch error and is repaired. The surviving error is a
within-sector transport-phase defect: the even six-face oscillator transmits
the complete local frame and suppresses leakage, but its response space does
not contain the remaining oriented first-order phase block. More strength
cannot create that missing response. The available odd face term is not
reflection covariant, and the available covariant triangle term supplies the
wrong compressed response.

**Corrective action:** Keep the negative-frequency comparator locked to the
positive-phase rest orbit. Retain the six-face oscillator as an unadmitted
control, not canonical law. Derive the residual compressed four-component
operator from the exact action tangent, decompose it under cubic reflections,
and enumerate the next reflection-covariant oriented loop/action basis. Admit
only a real local reciprocal invariant whose variation spans that residual,
vanishes on the exact rest orbit, and passes both copies before any nonlinear
endurance run. Do not tune the existing coefficient further. V4-006 remains
open and canonical V40 is unchanged.
