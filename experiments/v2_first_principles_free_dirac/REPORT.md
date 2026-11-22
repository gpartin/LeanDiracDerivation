# First-principles free-Dirac tangent from the native V40 action

## Verification status

The proof sources now follow a preparation-independent **operator route**:
the declared native action is varied, its free tangent is extracted about an
admitted uniform vacuum, and the resulting second-order phase space is mapped
losslessly to a four-component first-order equation. No simulation is a
premise.

The mathematical chain compiles through one canonical spectral readout.  An
independent top-down mathematical audit found no critical/high defect.  The
proof sources, compiled audit support, toolchain, Lake definition, and pinned
dependency manifest are hash-sealed against the recorded proof-source commit;
the final verifier is run only from a clean descendant checkout.

The intended scoped verdict table is:

```text
NATIVE_UNIFORM_VACUUM                         = PASS
TWO_TRANSVERSE_C19_KLEIN_GORDON_CHANNELS      = PASS
ACTION_HESSIAN_EXTRACTION                      = PASS
LOSSLESS_FOUR_COMPONENT_FIRST_ORDER_BRIDGE    = PASS
MASS_TIME_AND_THREE_TRANSPORT_TERMS            = PASS
CUBE_INCIDENCE_CLIFFORD_IDENTIFICATION         = PASS
SAME_READOUT_CENTERED_TIME_LIMIT               = PASS
SAME_READOUT_DIRAC_DENSITY_CURRENT_CONTINUITY  = PASS
SAME_READOUT_EXACT_C19_VERLET_POSITION_CHAIN   = PASS
FINITE_C19_CURRENT_BOUND_AND_LIMIT             = PASS
INTEGRATED_POSITIVE_PACKET_CHARGE              = PASS
INTEGRATED_POSITIVE_PACKET_CURRENT             = PASS_WITH_C19_REMAINDER
POINTWISE_POSITIVE_PACKET_DENSITY_EQUALITY      = PROVED_FALSE
UNIFORM_TANGENT_U1_CONTINUITY                  = PASS
ARBITRARY_NOETHER_EQUALS_POSITIVE_DIRAC_DENSITY = PROVED_FALSE
FINITE_MODE_POSITION_SPACE_DIRAC_EQUATION      = PASS
UNIFORM_VACUUM_DYNAMIC_SELECTION               = OPEN_OUTSIDE_THEOREM
LOCALIZED_NONLINEAR_MOVING_ELECTRON            = SEPARATE_OPEN_GATE
```

This is a first-principles **free linearized Dirac theorem** about one
selected representative of a native zero-energy vacuum. It is not a theorem
that the compact nonlinear rest orbit is already a moving physical electron.

## 1. Native starting law

At every lattice site the two native fields are

\[
\Psi_i,\chi_i\in\mathbb C^3.
\]

The declared action is

\[
L=T-U,
\qquad
T=\frac12\sum_i\|\dot\Psi_i\|^2+
  \frac B2\sum_i\|\dot\chi_i\|^2,
\]

\[
U_{\rm local}=\sum_i\left[
 \frac14(\rho_{\chi,i}-\chi_0^2)^2+
 \frac12\rho_{\chi,i}\rho_{\Psi,i}+
 \frac14\rho_{\Psi,i}^2\right],
\]

\[
U_{\rm area}=\sum_{i,n\in C19}
 \frac{w_n}{2\chi_0^2}\|W_{i+n}-W_i\|^2,
\qquad W_i=\chi_i\wedge\Psi_i.
\]

Here the six face directions have weight 2, the twelve edge directions have
weight 1, \(\chi_0=19\), and \(B=1197\). The only native equation families
are

\[
\ddot\Psi_i=-\frac{\delta U}{\delta\Psi_i^\dagger},
\qquad
B\ddot\chi_i=-\frac{\delta U}{\delta\chi_i^\dagger}.
\]

`NativeVariation.lean` proves the relevant first variation from this action.
The Dirac field, its matrices, and its first-order equation are not native
inputs.

The value \(\chi_0=19\) is identified with the nineteen C19 cube modes: one
center, six faces, and twelve edges. Once \(\chi_0\) is fixed geometrically,
\(\kappa=1/63\) merely parameterizes the remaining chi kinetic normalization
\(B=\chi_0/\kappa=1197\). This report does not claim a derivation of \(B\).

## 2. What “no prepared background” means

The action admits the uniform state

\[
\Psi=0,\qquad \chi=(0,0,r).
\]

Lean proves that this state is critical and, at \(r=\chi_0\), a global
zero-energy minimum. It also proves the stronger degeneracy statement: every
zero-\(\Psi\) texture with \(\|\chi_i\|=\chi_0\) independently at each site
has zero potential. Therefore this derivation imports no period-two pattern,
corner rays, or maintained substrate, but it does select a uniform member of
the vacuum family. The present action **does not dynamically align** arbitrary
sitewise vacuum orientations. That selection and any cosmological origin
story are intentionally outside the free-equation theorem and are not
premises of the result.

## 3. Hessian of the full action

Let the two components of \(\Psi\) transverse to the selected \(\chi\) axis
be complex fields \(f\) and \(g\), scaled by \(\epsilon\). Lean calculates
the restricted native energy exactly:

\[
U=\epsilon^2Q_r(f,g)+
  \frac{\epsilon^4}{4}\sum_i(|f_i|^2+|g_i|^2)^2.
\]

The quadratic form is two identical copies of

\[
\frac12\sum_i r^2|f_i|^2+
\frac12\sum_{\langle i,j\rangle\in C19}w_{ij}|f_i-f_j|^2.
\]

`UniformVacuumEulerLagrange.lean` does not define this tangent by fiat. It
proves that the symmetric second-difference quotient of the full nonlinear
action converges to twice the displayed modal Lagrangian. It then derives the
modal Euler--Lagrange acceleration. The linear \(\Psi\)-\(\chi\) cross blocks
vanish at \(\Psi=0\); the reciprocal \(\chi\) response starts at quadratic
order. The finite-amplitude quartic term remains part of the native theory;
the free theorem concerns its Hessian.

## 4. C19 transport and continuum control

`C19Fourier.lean` derives the C19 character, its isotropic second moment, the
exact quadratic symbol, and an explicit remainder:

\[
K_{19}(k)=12|k|^2+R_{\rm space}(k),
\qquad
|R_{\rm space}(k)|\le 40\,r_k^4
\]

under the stated low-momentum bound. The finite-periodic action Hessian is
also diagonalized on the actual torus characters, and its full state/velocity
generator is proved to preserve the admitted transverse mode sector.

Thus the action Hessian supplies two identical complex C19
Klein--Gordon channels with rest coefficient \(r^2\), rather than assuming a
continuum wave equation.

## 5. Four components and the cubical Clifford root

Each complex transverse channel is second order in time, so its state has a
complex position and complex velocity. Two channels therefore have exactly
four complex phase-space coordinates. `FirstPrinciplesFreeDirac.lean` proves
that its phase-space readout is bijective: no native tangent coordinate is
discarded and no extra coordinate is added.

`CubicalClifford.lean` constructs signed incidence operators from cube
geometry and proves their squares and anticommutators.
`FlavorClassification.lean` proves that the eight-corner fiber is the direct
sum of exactly two disjoint rank-four flavor sectors: every amplitude
decomposes into them, each eigenspace is exhausted by its four-component
embedding, and their intersection is zero. `NativeCubicalCompression.lean`
proves the same
native phase-space generator intertwines with each cubical representation,
with the explicit finite-C19 spatial defect carried through.

This proves the existence of a cube-incidence Clifford square root compatible
with the native tangent. It does not silently identify the abstract eight
coordinates with eight physical lattice sites, and it does not delete the
second flavor. Orientation, the global square-root sign, and flavor are
representation freedoms; the action proves admission of this factorization,
not a unique dynamical selection among equivalent charts.

The two sectors are consequently a prediction of the declared cubical root.
The further claim that measurement accesses one sector without identifying
which one is an interpretation reserved for physical validation, not a
formal consequence proved here.

## 6. Mass, time, and three spatial directions

The native Hessian's rest coefficient is \(r^2\). Positivity chooses the
positive square root, so at the canonical vacuum

\[
m=\chi_0=19.
\]

The three transport generators use the three components of the C19 quadratic
symbol and the cube-incidence Clifford operators. The same mass and the same
lossless readout are used for rest and transport.

`CanonicalSpectralExactChain.lean` applies the actual finite-C19 generator,
centered two-layer update, and Verlet step to one bijective spectral readout.
It proves the cubic local time defect vanishes at second-order scale and
retains the exact spatial C19 remainder. `UniformVacuumVerletDispersion.lean`
also proves the exact stable discrete frequency and its convergence to the
action-Hessian frequency. The numerical timestep is not reinterpreted as a
physical substrate tick.

## 7. Density and current

Two statements are deliberately kept separate.

1. The cube-derived Dirac velocity operator is proved to be the momentum
   response of the same four-component Hamiltonian. One bijective spectral
   readout carries the full positive/negative native generator and reduces on
   a positive-frequency native phase to the normalized positive-energy
   spinor using the same native amplitude.
2. Independently, the uniform native tangent C19 bond operator has its own
   exact U(1) phase-density and bond-current continuity law.

For one positive-frequency mode, Lean proves exact native U(1)-density/Dirac-
density equality and literal native C19 bond flux equals Dirac current plus an
explicit lattice remainder. That remainder obeys

\[
|R_{\rm current}|\le 16 r^3\rho
\]

and tends to zero at low momentum. Pointwise equality is false for arbitrary
multi-mode packets because of interference; Lean records a concrete
counterexample. On a finite torus, character orthogonality restores the
correct integrated packet charge identity and the integrated packet current
identity with the summed explicit C19 remainder. Mixed positive/negative data
remain outside the positive-density identification.

## 8. Position-space equation

`PositionSpaceFreeDirac.lean` takes finite Fourier superpositions of the same
native tangent modes. It proves their time derivative equals the mass term
plus all three spatial derivative terms of the four-component free Dirac
equation. This is the position-space conclusion of the operator chain; it is
not inferred from a numerical trajectory.

## 9. Exact claim boundary

The chain under final revalidation is

```text
native two-field C19 action
  -> compiled first variation
  -> selected uniform zero-energy vacuum representative
  -> exact action Hessian
  -> two complex transverse C19 Klein--Gordon channels
  -> complete four-complex-dimensional phase space
  -> cubical Clifford first-order representation
  -> rest mass + three transport directions
  -> centered-time and finite-spacing remainders
  -> same-spectral-readout density/current and controlled current limit
  -> finite-mode position-space free Dirac equation through that same readout
```

It does not prove dynamic vacuum alignment, uniqueness of the Clifford root,
the compact nonlinear rest orbit's identification with this field, nonlinear
localization, physical electron motion, electromagnetic coupling, or an
origin story. Those are separate theorems.

## 10. Proof and publication hygiene

The audited manifest contains the four native foundation modules, twenty-
five active V2 proof modules, and one additive V2 facade. The retired auxiliary
phase-space bridge is not imported by this chain and is not credited as
evidence. `AxiomAudit.lean` imports the cumulative theorem and invokes Lean's
compiled-environment audit. That audit rejects custom axioms, `sorryAx`, and
dependencies outside

```text
propext
Classical.choice
Quot.sound
```

Every manifest module builds, the cumulative theorem passes the compiled
environment and declaration audits, the red-team repairs pass, and the final
source hashes are recorded.
