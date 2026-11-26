# V3.2 Global Free-Fermion Closure Plan

## Objective

Starting from the unchanged V3.1 action, equations, C19 geometry, vacuum,
readout, and cumulative theorem, determine whether the complete linear sector
is a globally consistent free massive spin-one-half lattice particle theory.

V3.2 must close every mathematical question that can be answered without:

- a physical length, time, energy, or MeV anchor;
- electromagnetic coupling or an external gauge field;
- nonlinear localization or live packet transport;
- unit-charge, occupation, exchange, exclusion, or CAR assumptions;
- a change to the native action or governing equations.

The release remains a free-particle theorem. It does not call the resulting
object an electron.

## Authorized finite-spacing scope decision

The owner authorized Route 1 after the Stage-2 factorization investigation:

> V3.2 proves the exact continuum free-Dirac closure and carries the exact
> finite-C19 discrepancy as a derived, controlled lattice remainder. It does
> not claim that the existing local cubical Dirac root is an exact
> finite-spacing square root of the C19 action Hessian.

This is an explicit claim correction, not a silent reduction. The exact
finite-C19 scalar spectrum, its gap, the global first-order obstruction, and
the exact local bond decomposition all remain part of the release evidence.
The bond-incidence enlargement is reserved for a separately versioned
research branch.

## Fixed starting point

- Base commit: `61a4798b8b3ebfcffd24065079dc4db9d7b43f75`
- Lean: `v4.35.0-rc2`
- Framework: the V3.1 declared two-field C19 action
- Existing cumulative theorems:
  - `complete_first_principles_free_dirac`
  - `lorentz_spin_half_from_native_action`
- No action, stencil, vacuum, or readout change is authorized in V3.2.

## Stage architecture

V3.2 has **17 stages**. This count comes from sixteen distinct theorem
handoffs plus one cumulative closure stage. Stages are not padded to match an
earlier release. Each stage has a separate falsifiable finish line and one
Lean module when implemented.

### V32-001 — Framework and claim lock

**Module:** `LeanDirac/V2/V32FrameworkLock.lean`

Import the sealed V3.1 cumulative certificates and prove a V3.2 starting
certificate that records the unchanged action, C19 symbol, mass, readout,
two-copy decomposition, Lorentz/spin certificate, and allowed claim boundary.

**Pass:** every later module depends on this certificate; source and axiom
audits confirm no changed premise.

### V32-002 — Global Hamiltonian spectrum

**Module:** `LeanDirac/V2/V32GlobalSpectrum.lean`

Prove Hermiticity and the square relation for the exact continuum/root
Hamiltonian. Separately prove the unchanged action's exact finite-C19 scalar
energy branches, global nonzero-mass gap, and absence of complex-energy
ghosts. Prove and expose the exact finite-spacing obstruction and remainder;
do not identify the continuum root with an exact finite-C19 first-order root.

**Pass:** the continuum/root Hamiltonian has a complete exact spectral
theorem, the finite-C19 scalar action spectrum is exact and globally audited,
and their nonzero finite-spacing difference is explicitly certified.

### V32-003 — Projectors, completeness, and unitary evolution

**Module:** `LeanDirac/V2/V32SpectralProjectors.lean`

Construct positive- and negative-energy projectors from the derived
Hamiltonian. Prove idempotence, orthogonality, completeness, branch
invariance, positive one-particle density, and norm-preserving exact
continuum-time evolution.

**Pass:** the derived operator defines a mathematically consistent free
one-particle evolution on each branch.

### V32-004 — Full Brillouin-zone and doubler audit

**Module:** `LeanDirac/V2/V32BrillouinZoneAudit.lean`

Classify every zero, degeneracy, and low-energy cone of the exact lattice
symbol over the full Brillouin zone. Distinguish momentum-space fermion
doublers from the already-proved two internal rank-four copies.

**Pass:** Lean proves the exact number and location of all low-energy species.
If additional doublers exist, preserve and report them; do not change the
equations in this stage.

### V32-005 — Two-copy equivalence and mixing classification

**Module:** `LeanDirac/V2/V32CopyClassification.lean`

Prove the exact unitary equivalence between the two exhaustive cubical copies,
identify the native symmetry that exchanges them, compute the commutant of the
free operator on the multiplicity space, and classify which unchanged-action
operators can or cannot mix the copies.

**Pass:** the copies are rigorously classified as multiplicity, protected
sectors, or unresolved species; no measurement-selection story is assumed.

### V32-006 — Irreducible massive spin-one-half particle sector

**Module:** `LeanDirac/V2/V32IrreducibleParticle.lean`

Combine the mass-shell and Pauli--Lubanski invariants with the positive-energy
projector. Prove that each two-state positive-energy little-group sector is an
irreducible spin-one-half representation and record the multiplicity of the
full theory.

**Pass:** the result is a representation-theoretic particle classification,
not only a list of matrices with the right commutators.

### V32-007 — Finite continuum Lorentz covariance

**Module:** `LeanDirac/V2/V32FiniteLorentz.lean`

Exponentiate the already-derived rotation and boost generators. Prove the
finite state transformation group laws and retain the already-proved
infinitesimal covariance of momentum, gamma, current, and Pauli--Lubanski
objects. This is the owner-authorized "attempt, then narrow" boundary.

**Pass (narrowed):** finite state group laws follow from the V3.1 generators
without inserting an independent standard spin transformation. Full finite
observable covariance is not claimed.

### V32-008 — C, P, T, and CPT structure

**Module:** `LeanDirac/V2/V32DiscreteSymmetries.lean`

Define parity, time reversal, and charge conjugation from the derived cubical
representation and prove their squares, Hamiltonian covariance or branch
exchange, and the combined map's action on density and spin.

**Pass (narrowed):** the represented free continuum sector has explicit
machine-checked discrete-symmetry maps. No physical positron or interacting
CPT theorem is claimed.

### V32-009 — Free nonrelativistic limit

**Module:** `LeanDirac/V2/V32NonrelativisticLimit.lean`

Remove the native rest-energy phase, eliminate the lower component with a
controlled remainder, and derive the free Schrodinger generator. Prove the
leading kinetic term and the first relativistic correction with explicit
dimensionless error bounds.

**Pass:** the low-energy free Schrodinger limit follows from the LFM-derived
Dirac sector without inserting a Schrodinger equation.

### V32-010 — Uniform continuum operator convergence

**Module:** `LeanDirac/V2/V32ContinuumOperatorConvergence.lean`

Prove the exact finite-C19 scalar split, its uniform bounded-momentum
fourth-order envelope, low-momentum current-remainder convergence, and the
separate centered-time/Verlet limits. Route 1 forbids inventing an exact local
finite-C19 four-component Hamiltonian merely to state projector convergence.

**Pass (narrowed):** explicit spatial/current remainder control and temporal
limits are proved. This does not claim an exact finite-C19 Dirac projector or
finite-time nonlinear packet convergence.

### V32-011 — Leading lattice artifacts and anisotropy

**Module:** `LeanDirac/V2/V32LatticeArtifacts.lean`

Derive the finite-C19 dispersion correction and identify the first geometric
order at which directional anisotropy actually occurs. The audit found that
the fourth phase moment is rotationally isotropic and that the axis/face split
first occurs in the sixth phase moment.

**Pass:** Lean proves the isotropic fourth moment, explicit sixth-order
axis/face values, and the global fourth-order remainder envelope.

### V32-012 — Rest-mass certificate and physical-scale identifiability

**Module:** `LeanDirac/V2/V32ScaleIdentifiability.lean`

Construct one `NativeRestMassCertificate` and prove, for both exhaustive
cubical copies:

1. zero lattice momentum gives the exact Hamiltonian `chi0 * beta`;
2. the two rest-energy branches are exactly `+chi0` and `-chi0`;
3. the positive rest branch has zero spatial current, zero group velocity,
   and nonzero positive density;
4. the invariant extracted from the full dispersion relation is
   `E^2 - c_LFM^2 |p|^2 = chi0^2` in the continuum readout;
5. nonnegativity uniquely selects the rest mass `m_rest = chi0 = 19`;
6. the curvature of the positive energy branch at zero momentum gives the
   same inertial mass after the derived speed normalization;
7. the particle/antiparticle rest gap is exactly `2 * chi0`;
8. the native reduced-Compton and related dimensionless mass ratios follow
   from the already-derived C19 speed and mass;
9. conversion to physical units has the symbolic form
   `E_rest,physical = 19 * E_native_unit`, equivalently
   `m_physical = 19 * E_native_unit / c_physical^2`.

Then classify the overall rescaling freedom of the unchanged equations and
prove whether `E_native_unit` is fixed internally or remains an independent
dimensionful anchor.

**Pass:** Lean certifies an exact rest mass of `19` in native LFM units,
rest/inertial agreement, zero rest transport, equal mass in both copies, and
the physical-unit conversion law. It then either identifies an internal
absolute scale or proves that an independent dimensionful anchor is required
before `19` can be converted to MeV. No measured electron value may be used as
an input.

### V32-013 — Vacuum, normalization, and readout robustness

**Module:** `LeanDirac/V2/V32Robustness.lean`

Prove or delimit invariance under vacuum-orientation changes, identify every
result independent of `B = 1197`, and extend the existing scoped readout and
rotation uniqueness results where the unchanged representation permits it.

**Pass:** every major conclusion has an explicit dependency table and no
declared framework constant is silently promoted to a derived necessity.

### V32-014 — Exact transport observables for V4

**Module:** `LeanDirac/V2/V32TransportObservables.lean`

Derive exact phase velocity, group velocity, Hamiltonian momentum derivative,
Noether/Dirac current velocity, rest behavior, branch signs, and speed bounds
from one spectral package. Prove the equalities and controlled differences
that V4 should measure.

**Pass:** V4 receives exact formulas and tolerances, not fitted numerical
targets.

### V32-015 — Centered-step transfer spectrum for V4

**Module:** `LeanDirac/V2/V32CenteredStepTransport.lean`

Derive the exact one-step/two-layer transfer spectrum of the centered update,
its stable unit-modulus root, recurrence polynomial, exact step angle,
modified-energy invariant, and numerical-frequency continuum limit.

**Pass (narrowed):** every stable V4 plane-wave transport run has an exact
per-step recurrence and frequency reference. No stronger uniform phase-error
rate is claimed than has been proved.

### V32-016 — Finite packet benchmark contract for V4

**Module:** `LeanDirac/V2/V32PacketBenchmarks.lean`

Export the arbitrary finite positive-frequency spectral-packet theorems
already derived from the exact Fourier basis: integrated charge/current,
real-pair reconstruction, and the explicit counterexample rejecting a false
pointwise density bridge.

**Pass (narrowed):** export theorem-backed packet contracts. Numerical fixture
tables are regression conveniences rather than mathematical evidence and are
not made part of the release theorem. This stage does not evolve the nonlinear
LFM system itself.

### V32-017 — Cumulative certificate, hostile audit, and release gate

**Module:** `LeanDirac/V2/V32CompleteFreeFermion.lean`

Assemble the sixteen prior certificates into one cumulative theorem. Run the
sealed source audit, semantic axiom audit, declaration-provenance audit,
hardcoding audit, claim audit, and independent clean build.

**Pass:** no `sorry`, `admit`, custom axioms, inserted Dirac/Schrodinger/CPT
operators, discarded spectral branches, hidden physical units, or equation
changes. Critical and high red-team findings are zero.

## Transport-team deliverables

Stages V32-014 through V32-016 are deliberately analytic and therefore safe
to develop in parallel with V4. They will provide:

1. exact energy, branch, polarization, current, and velocity targets;
2. exact centered-step frequency and stability predictions;
3. spatial-versus-temporal error decomposition;
4. deterministic plane-wave and finite-packet fixtures;
5. expected convergence orders and failure classifications;
6. a rule distinguishing a wrong initial branch/readout from a failure of
   native transport.

V3.2 will not modify V4 code, run V4's live transport closure, or claim that
operator convergence proves finite-time nonlinear packet convergence.

## Release claim boundary

If all stages pass, V3.2 may claim:

> The unchanged LFM action has a formally audited continuum free-Dirac linear
> sector whose spectrum, projectors, spin representation, discrete
> symmetries, low-energy limit, continuum convergence, lattice artifacts,
> scale identifiability, and analytic transport targets are certified. The
> exact finite-C19 scalar spectrum is also certified, together with the
> explicit controlled remainder separating it from the local continuum Dirac
> root.

It may not claim a physical electron, `0.511 MeV`, electric charge, an
LFM-derived electromagnetic interaction, `g = 2`, nonlinear localization,
live transport, or fermionic occupation/exchange statistics.

## Pass/fix rule

For every stage:

1. Run the theorem and audit gates.
2. If it fails, identify the first failed mathematical layer.
3. RCA the failure and attempt the smallest proof or definition repair that
   preserves the V3.1 action and claim boundary.
4. RCA each failed repair in turn.
5. Do not change the equations, discard branches, add a physical anchor, or
   weaken the stated finish line silently.
6. Stop only when the remaining resolution is genuinely unknown or requires
   authority outside V3.2.

## Implementation outcome: proved Stage-2 obstruction

The RCA/fix loop reached a mathematical rather than tactical boundary. The
unchanged action has the exact finite-C19 dispersion
`mass^2 + c19TransportSymbol(momentum)`. The V3.1 local oriented cubical root
squares to `mass^2 + c19QuadraticSymbol(momentum)`. Their difference is the
already-derived `c19SpatialRemainder`.

V3.2 now proves:

- exact finite-C19 positive and negative real energy branches;
- a nonzero-mass global spectral gap;
- the local root is exact if and only if the remainder vanishes;
- at momentum `(pi, 0, 0)` the remainder is `48 - 12*pi^2`, which is nonzero;
- therefore no globally exact local oriented root of the required V3.1 form
  exists for the unchanged finite-C19 action.

This falsifies the original V32-002 pass condition as written. The downstream
continuum-root projector chain remains valid and has been strengthened, but it
cannot be relabeled as an exact finite-lattice Dirac chain. No equation or
operator change was made. The owner subsequently authorized Route 1, so the
release claim is explicitly narrowed to the exact continuum root plus the
exact finite-C19 scalar spectrum and controlled remainder.
