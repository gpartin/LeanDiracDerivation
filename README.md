# Lean LFM-to-Dirac Derivation V3.6 Research

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository contains one cumulative production proof line. V2 derives the
complete free linearized Dirac equation from the declared two-field C19 lattice
action. V3 continues from that theorem and derives the Lorentz/spin-one-half
representation carried by both exhaustive cubical copies. V3.2 adds the
free-fermion spectral, rest-sector, symmetry, continuum-error, and transport
closure while preserving the same action and governing equations. V3.3 adds
free-spinor basis, bilinear-observable, axis-aligned helicity, forward Gordon,
and free-resolvent closure without changing those dynamics. V3.4 closes the
covariant free-spinor package with native-derived on-shell spinors, exact
covariant spin sums, arbitrary-momentum helicity projectors, the non-forward
Gordon identity, and a projector/residue decomposition of the free resolvent.
V3.5 completes the negative-branch residue relation, finite generated
Lorentz covariance, the sixteen-channel bilinear/Fierz package, and the
arbitrary-momentum Pauli--Lubanski spin invariant.
V3.6 revises the research action's unresolved relative `Psi`/`chi` kinetic
normalization. It selects one common standard Euclidean cubical Clifford
kinetic metric for the two native grades, derives `B = 1` within that metric
class and hence `kappa = chi0/B = 19`, and proves
that the entire cumulative free Dirac chain survives this change. The former
`B = 1197` value and the C19-weight `B = 19` value remain explicit controls.

The default import is unambiguous:

```lean
import LeanDirac

#check LeanDirac.complete_first_principles_free_dirac
#check LeanDirac.lorentz_spin_half_from_native_action
#check LeanDirac.v32_complete_free_fermion
#check LeanDirac.v32_two_copy_rest_sector
#check LeanDirac.v33_complete_free_spinor
#check LeanDirac.v34_complete_covariant_free_spinor
#check LeanDirac.v35_complete_covariant_observables
#check LeanDirac.v36_live_research_action_mode_to_free_dirac
#check LeanDirac.v36_complete_geometrically_normalized_action
```

The checked-out tree does not contain the historical numbered V1 proof or its
downstream consequence stages. That development remains available in Git
history at tag `v1-legacy-final`.

## Starting point

At each lattice site the native fields are `Psi_i, chi_i : C^3`. The declared
geometry is C19: six face bonds of weight 2 and twelve edge bonds of weight 1.
The vacuum normalization is `chi0 = 19`, identified with the nineteen C19
cube modes (center, six faces, and twelve edges). V3.6 keeps both native fields
as `C^3` arrays, represents `Psi` in the vector grade, and maps `chi`'s three
coordinates to the corresponding oriented-bivector grade for the kinetic
metric comparison. Unit blades in those grades have
equal norm, so the live research action uses `B = 1` and the equivalent
parameterization `kappa = chi0/B = 19`.

The earlier `(kappa,B)=(1/63,1197)` pair is retained as a historical control,
not as the live action. The C19-weight pair `(1,19)` is a second ablation. The
finite-lattice transverse theorem is uniform in `B`: at the proved vacuum its
linear transverse `Psi` sector has no linear `chi` motion, and its Dirac mass
and transport generator contain no numerical use of any candidate value. The
value of `B` remains relevant to live-`chi` inertia, nonlinear response, and
any future localized moving-electron construction.

The proof starts from one real reciprocal action `L = T - U`, derives both
Euler--Lagrange equations, and derives the centered two-layer update. The full
action is stated in [docs/FRAMEWORK.md](docs/FRAMEWORK.md).

## Result

The cumulative theorem is
`LeanDirac.complete_first_principles_free_dirac`. Its dependency chain
derives:

- reciprocal native field variation from the action;
- an exact native zero-energy vacuum and the full finite-lattice tangent;
- the closed four-complex-component phase space;
- the cubical Clifford root and an exhaustive direct-sum decomposition into
  exactly two rank-four copies;
- the standard alpha/beta form, nonnegative mass, and three transport terms;
- centered/Verlet time evolution with an explicit vanishing time defect;
- the position-space free Dirac equation with the explicit C19 remainder;
- the positive-frequency density/current bridge and its controlled remainder.

There is no external, prepared, period-two, maintained, or Dirac-shaped
background. The expansion point is one exact zero-energy vacuum solution
proved from the native action. Selecting a representative of the degenerate
vacuum family is not claimed to be a cosmological origin mechanism.

Within the declared cubical root, the two four-component sectors are a
formal prediction, not discarded doubling. Lean proves that both sectors
are invariant, disjoint, exhaustive, and carry the same free Dirac operator.
The interpretation that an electron measurement accesses one sector without
revealing which one was accessed is a physical hypothesis; it is not itself
a theorem in this repository.

The theorem is a free linear tangent result. It does not claim a nonlinear
localized electron or any later physical consequence. V3.6 proves that this
result remains valid for the geometrically normalized live action `B = 1`.

## V3 Lorentz and spin extension

V3.1 continues from the proved free-Dirac boundary without
changing the action, equations, C19 stencil, vacuum, or readout.  It derives
the spatial rotation generators directly from the cubical gamma operators,
proves that their unique nonzero closure scale is `1/2`, proves that this
normalization is also uniquely selected within the explicit local
plane-bivector class by covariance of the native transport directions, and
then proves the
spin-one-half spectrum, Casimir `3/4`, density-preserving finite rotations,
`2*pi` sign reversal, `4*pi` return, and the rotation/boost Lorentz algebra.
It also proves the finite rotation group law and generator derivative,
intertwines rotations and boosts through both exhaustive cubical copies,
proves infinitesimal covariance of the low-momentum Dirac operator while four-momentum and
the spinor transform together, and constructs the Pauli--Lubanski vector from
four-momentum and the derived Lorentz generators. Its boost and rotation
four-vector laws are proved for all axes before obtaining its rest Casimir
`-3 m^2 / 4`.

The cumulative local theorem is
`LeanDirac.lorentz_spin_half_from_native_action`.  The exact chain and
claim boundary are documented in
[docs/LORENTZ_SPIN_CHAIN.md](docs/LORENTZ_SPIN_CHAIN.md).

This is an infinitesimal low-momentum operator/representation theorem. It does
not prove fixed-physical-scale finite-time solution convergence, exact Lorentz
invariance at finite lattice spacing, unrestricted uniqueness of physical
rotations, a localized nonlinear electron, electromagnetic coupling, or a
dynamical rule selecting one copy.

## V3.2 free-fermion spectral closure

V3.2 changes no action, governing equation, C19 geometry, vacuum, or V3.1
readout. Its cumulative theorem is
`LeanDirac.v32_complete_free_fermion`. The machine-checked chain adds:

- the exact continuum free-Dirac energy branches and spectral projectors;
- the exact finite-C19 scalar spectrum, global gap, and no-extra-zero audit;
- a proof that the existing local four-component root is not an exact global
  finite-spacing factorization, together with the explicit controlled
  continuum remainder;
- the two-copy commutant, rest-spin irreducibility, and a combined two-copy
  rest-sector certificate;
- represented free-sector C, P, T, and CPT identities;
- the free Schrodinger limit and its first correction;
- centered-step and finite spectral-packet transport references for later
  transport work;
- native rest mass `19` and rest gap `38`, plus a proof that an absolute MeV
  conversion cannot be selected without an independent dimensionful anchor;
- an explicit theorem that the free transverse tangent result is uniform in
  the arbitrary live-chi kinetic normalization `B`.

The finite-spacing distinction is part of the result: V3.2 proves the exact
C19 second-order spectrum, but does not relabel the continuum cubical root as
an exact finite-C19 local Dirac operator. It also does not claim a nonlinear
localized electron, electromagnetic coupling, `g = 2`, a MeV prediction, or
V4 transport closure.

See [the V3.2 release notes](docs/V32_RELEASE_NOTES.md),
[stage status](docs/V32_STAGE_STATUS.md),
[hostile audit](docs/V32_RED_TEAM.md), and
[dependency matrix](docs/V32_DEPENDENCY_MATRIX.md).

## V3.3 free-spinor observable closure

V3.3 remains downstream of the unchanged V3.2 action and equations. It proves
a complete orthonormal four-component basis and outer-product resolution,
constructs both on-shell energy branches from the V3.2 projectors, derives the
normalized scalar/density/current bilinears, proves exact axis-aligned moving
helicity states, proves the forward equal-momentum Gordon relations, and
constructs a two-sided free momentum-space resolvent away from its pole.

The cumulative theorem is `LeanDirac.v33_complete_free_spinor`. The Gordon
result is not the non-forward electromagnetic vertex; the propagator is not
interacting; and V3.3 does not claim `g = 2`, charge quantization, localization,
transport, a physical electron, or a MeV calibration.

## V3.4 covariant free-spinor closure

V3.4 changes no action, equation, coefficient, C19 geometry, vacuum, readout,
or scale. Its cumulative theorem is
`LeanDirac.v34_complete_covariant_free_spinor`. The seven-stage chain proves:

- positive- and negative-energy covariant branch spinors derived from the
  existing native frequency embeddings;
- exact two-channel injectivity, branch resolution, and orthogonality;
- covariant particle and antiparticle spin sums with the repository's stated
  normalization;
- arbitrary nonzero-momentum helicity whose square is `1/4`, together with
  complete idempotent helicity projectors;
- the unequal-momentum Gordon identity from the derived gamma algebra and
  explicit on-shell conditions;
- spectral decomposition of the free resolvent into the existing V3.2 energy
  projectors, identification of both pole numerators, and agreement of the
  positive spin sum with the positive projector after beta conversion.

This is still a covariant free linearized theorem. It does not prove an
interacting or localized electron, electromagnetic emergence, `g = 2`, charge
quantization, packet transport, a MeV calibration, or an exact finite-spacing
Dirac root. See the [V3.4 plan](docs/V34_COVARIANT_FREE_SPINOR_CLOSURE_PLAN.md)
and [stage status](docs/V34_STAGE_STATUS.md).

## V3.5 covariant-observable closure

V3.5 continues from the unchanged V3.4 certificate. Its eight-stage chain:

- connects the antiparticle spin sum to the negative-energy projector and
  negative propagator pole;
- builds finite transformation words only from the existing cubical rotors
  and boosts and proves exact composition and inverse laws;
- proves finite rotation/boost covariance of the free Dirac residual and
  covariance under arbitrary words generated from those transformations;
- derives gamma-five from the cubical gamma operators and defines the standard
  scalar,
  pseudoscalar, vector, axial-vector, and tensor bilinears;
- proves the sixteen-channel Fierz completeness identity in the same cubical
  representation;
- evaluates the Pauli--Lubanski square at arbitrary momentum, obtains
  `-3 m^2 / 4` on shell, and identifies V3.4 helicity with its normalized time
  component.

The cumulative theorem is
`LeanDirac.v35_complete_covariant_observables`. The transformation result is
an exact free momentum-space theorem; it is not finite-packet transport or a
nonlinear substrate evolution result. V3.5 still does not prove a localized
or interacting electron, electromagnetic coupling, physical charge, `g = 2`,
fermionic quantization/statistics, or an MeV calibration. See the
[V3.5 plan](docs/V35_COVARIANT_OBSERVABLE_CLOSURE_PLAN.md) and
[stage status](docs/V35_STAGE_STATUS.md).

## V3.6 geometrically normalized action

V3.6 changes the action's live relative kinetic normalization and compares it
with two controls:

| Interpretation under audit | `kappa` | `B` |
|---|---:|---:|
| Historical 63-mode control | `1/63` | `1197` |
| C19 kinetic-weight ablation | `1` | `19` |
| **Live common Clifford metric** | **`19`** | **`1`** |

Lean proves that the oriented cube Hodge map between vector coordinates and
plane-bivector coordinates is invertible and norm-preserving. Constructing
the kinetic term from that one standard Euclidean Clifford metric gives equal
unit-blade weights and therefore `B = 1`. The two fields remain distinct
native fields; the Hodge map represents `chi`'s coordinates and does not
identify the fields. Only their kinetic metric is shared.
The `B = 19` result requires a separate C19 kinetic-multiplicity postulate and
is therefore retained only as an ablation. Cube symmetry alone preserves the
tested kinetic metric for arbitrary `B` and is not misreported as the reason
for the live value.

The inherited radial potential gives exact squared small-response values
`38/63`, `38`, and `722` for the three candidate pairs. Faster response is a
consequence, not the selection rule. The existing free transverse Dirac
tangent is formally valid for every candidate and therefore survives the
action change but cannot itself justify it.

The cumulative theorem is
`LeanDirac.v36_complete_geometrically_normalized_action`. V3.6 does not derive
nonlinear electron mobility, an Einstein limit, or repository-wide promotion
of this V3.6 research action to canonical v40.1. See the
[V3.6 plan](docs/V36_GEOMETRICALLY_NORMALIZED_ACTION_PLAN.md),
[decision matrix](docs/V36_DEPENDENCY_AND_DECISION_MATRIX.md), and
[stage status](docs/V36_STAGE_STATUS.md). The
[release notes](docs/V36_RELEASE_NOTES.md) and two hostile reviews
([technical](docs/V36_RED_TEAM_1_TECHNICAL.md),
[release](docs/V36_RED_TEAM_2_RELEASE.md)) record the verification boundary.

The direct theorem
`LeanDirac.v36_live_research_action_mode_to_free_dirac` names the live
`v36ResearchC19Lagrangian` in its tangent quotient and proves the full
finite-periodic action-to-free-Dirac conclusion for that action. This avoids
inferring the connection merely from two certificates packaged side by side.

The finite covariance theorem is for the free Dirac residual and its solution
space. The bilinear branch proves the five standard definitions, their
Clifford relations, and Fierz completeness; it does not separately prove the
finite transformation law of every individual bilinear.

## Prior work and attribution

The oriented-cube construction belongs to the established
Dirac--Kähler/discrete-differential-form lineage. Appendix A of F. L.
Teixeira’s 2013 review was the direct design clue for looking for a Clifford
algebra in oriented cube incidence rather than inserting standard Dirac
matrices. Foundational and contextual sources include Dirac; Iwanenko and
Landau; Kähler; Becher and Joos; Rabin; Nielsen and Ninomiya; Noether;
Clifford; Fierz and Pauli; Wigner; Lubański; and Schrödinger.
The Lean 4 and mathlib projects are also credited as the formal-verification
infrastructure on which the checked artifact depends.

The exact attribution, DOI links, and boundary between this prior art and the
repository’s claimed contribution are recorded in
[docs/V32_RELATED_WORK.md](docs/V32_RELATED_WORK.md). Citation metadata for the
software artifact and the same reference set is in
[CITATION.cff](CITATION.cff).

## Reproduce

Install Lean through `elan`, then run:

```bash
python scripts/verify_release.py
```

This is the one authoritative V3.6 release command. It builds the cumulative
facade and compiled environment audit, checks the V3.6 proof and release
surfaces, rejects unfinished proofs and custom axioms in the audited production
closure, and admits only `propext`, `Classical.choice`, and `Quot.sound`.

## Repository map

- `LeanDirac/`: cumulative production proof modules;
- `scripts/verify_release.py`: authoritative current release gate;
- `docs/`: current proof map plus versioned plans, audits, and release records;
- `Challenge.lean`, `Solution.lean`, `comparator.json`, and
  `formalization.yaml`: preserved V2 external-submission interface, not the
  current V3.6 production entry point;
- `project/dirac_v2_stage_status.json` and `experiments/`: preserved historical
  release evidence.

The challenge file intentionally contains theorem holes for an external solver;
`Solution.lean` fills them. Neither file is imported by the production facade
or counted by the production no-`sorry` claim. See
[the historical-artifact map](docs/HISTORICAL_SUBMISSION_ARTIFACTS.md) and
[documentation index](docs/INDEX.md).

See [the proof chain](docs/PROOF_CHAIN.md), [claims](docs/CLAIMS.md), and
[reproducibility instructions](docs/REPRODUCIBILITY.md).
