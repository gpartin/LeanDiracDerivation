# Lean LFM-to-Dirac Derivation

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository is the production Lean 4 environment for a noncanonical
attempt to derive the free Dirac equation from a local two-field cubic-lattice
substrate.

All **22 of 22** derivation gates pass. The chain begins with
one positive complex-Psi/complex-chi C19 action, proves that its restriction
and variation commute with the retained eight-site embedding, and derives a cubical
low-momentum square root with an explicit fourth-order lattice remainder, and
proves that both native rank-four copies are equivalent to the conventional
Dirac alpha/beta representation, organizes both frequency signs, and uniquely
selects the nonnegative rest coefficient from the native running frequency,
and derives a first-order branch from the centered recurrence, with its exact
C19 spatial defect and vanishing generator error. It then derives the native bond flow,
exact local phase continuity, and the positive-frequency continuum Dirac
density/current limit from the same action and embedding. The final theorem
makes the Klein--Gordon/Dirac relationship explicit: the action-derived
massive C19 tangent has a Klein--Gordon continuum symbol, and the derived
four-component Dirac Hamiltonian squares to that symbol. This is a conditional
free-field continuum result, not a localized physical electron proof.
DIRAC-016 then proves a finite-cube structural result without changing that
action: unsigned microscopic translation exchanges the two retained
rank-four copies in one cube direction and a second translation returns to
the original copy. The signed incidence readout, by contrast, preserves each
copy. This supports a copy-alternation motion hypothesis but does not prove
that native time evolution or a localized electron uses it.
DIRAC-017 then derives the rotation generators directly from the same signed
cube-incidence algebra. The actual rotation commutator uniquely selects the
nonzero scale `1/2`, the Casimir is `3/4`, and both retained rank-four copies
intertwine exactly with the conventional four-component spin generators. The
component spectrum, positive-definite native inner product, continuous unitary
rotation group, `2π` sign/`4π` return, mass-grading compatibility, and both
quarter-turn and all-angle interpreted-Hamiltonian covariance are proved.
Covariance and unitarity also
uniquely characterize the positive quarter-turn inside the native plane-rotor
family. This proves a conditional spin-one-half representation of the derived
continuum sector. It does not prove that the native dynamics uniquely selects
this square-root orientation or realizes it as a localized physical electron.
DIRAC-018 closes that orientation question inside a precise native candidate
class. For real, local, first-order, flavor-preserving combinations of the
existing cubical incidence operators, cubic covariance forces equal transport
coefficients, scalar squaring fixes their magnitude, and the independently
derived positive mass/current orientation fixes their sign. The proof applies
independently to both exact rank-four copies, proves those copies exhaust the
eight-corner space, and proves that the exact C19 scalar symbol has no extra
zero in the first Brillouin zone. This is a scoped classification, not a claim
about arbitrary nonlocal or flavor-mixing roots, an exact finite-spacing
Dirac factorization, all forms of fermion doubling, or a physical electron.
DIRAC-019 then proves sustained linear transport for arbitrary finite
positive-frequency spectral families. The stable recurrence-derived continuum
branch advances every mode with unit norm for every integer update count, the summed
current remains equal to the density-weighted group-velocity moment at every update, massive modes are
strictly subluminal, and the finite-family generator converges to the same
Dirac generator. This is not yet a coherent real-space theorem in Lean or a
nonlinear localized-electron transport result.
DIRAC-020 then turns that finite spectral family into one coherent spatial
field. It constructs an explicit normalized Fourier frame on every finite
abelian periodic lattice and proves its orthonormality from character
orthogonality. On the periodic cubic lattice, the one-site phase of every
Fourier character is proved to match the momentum stored in the corresponding
positive-frequency mode, so the spatial and dynamical labels cannot drift
apart. Parseval gives exact integrated-density preservation, while the
integrated spatial current remains the density-weighted transport velocity at
every stable update. A finite-update pointwise recurrence-versus-continuum
error bound is also proved. This remains a linear continuum-branch result,
not exact native finite-spacing or nonlinear live-chi electron motion.
DIRAC-021 then follows the rest coefficient already selected from the native
action through the zero-momentum Hamiltonian. It proves explicit positive- and
negative-energy rest states and their first-order time equations. Their phases
run oppositely, so a coherent pair has an exact relative angular frequency
twice the rest frequency. In native units those values are `19` and `38`; the
inverse-mass scale is `1/19`, while the native-coordinate reduced Compton
length is `sqrt(12)/19`. A pure branch's global phase is not asserted to be a
directly observable clock, and no SI electron mass or zitterbewegung claim is
made here.
DIRAC-022 now supplies that explicit observable without changing the action.
The existing Hamiltonian momentum derivative couples the proved positive and
negative rest branches.  Their normalized coherent superposition has velocity
expectation `c_eff cos(2 chi0 t)`, native angular frequency `38`, and an
integrated position amplitude `sqrt(12)/38`.  The full peak-to-peak excursion
is `sqrt(12)/19`, the native reduced Compton length.  Pure branches,
same-energy-sign superpositions and incoherent mixtures have zero signal.
This is a free-continuum expectation-value theorem, not a localized native
electron trajectory or an SI prediction.

## Current status

| Range | Status | Result |
|---|---|---|
| DIRAC-001--003 | PASS | Native framework, one positive action, and reciprocal action variations. |
| DIRAC-004--005 | PASS | Reversible centered evolution and the conditional prepared background. |
| DIRAC-006--008 | PASS | Exact native-action restriction/variation bridge, native C19 transport, lossless readout, and exact first-order map. |
| DIRAC-009 | PASS | Cubical Clifford square root of the C19 continuum symbol, with an explicit lattice remainder and both rank-four copies retained. |
| DIRAC-010 | PASS | Exact change of basis from both native copies to the standard Dirac alpha/beta representation. |
| DIRAC-011 | PASS | Common positive/negative-frequency graphs and native nonnegative rest-mass selection. |
| DIRAC-012 | PASS | Action-derived linearized scalar channel and a first-order Dirac branch derived from the centered recurrence, with exact native spatial defect and controlled continuum limit. |
| DIRAC-013 | PASS | Action-derived native bond flow, exact local continuity, and positive-frequency continuum density/current. |
| DIRAC-014 | PASS | Final adversarial audit reports zero critical/high findings. |
| DIRAC-015 | PASS | One cumulative action-to-Dirac theorem and deterministic clean-checkout package. |
| DIRAC-016 | PASS | Exact A-B-A copy exchange under unsigned cube translation, with density preservation, unique coordinate map, and a strict motion-claim boundary. |
| DIRAC-017 | PASS (conditional representation) | Cubical spin-one-half representation: exact standard-spin intertwiner on both copies, explicit spectrum, uniquely characterized quarter-turn, continuous unitary rotations, `2π` sign/`4π` return, and quarter-turn plus all-angle mass-and-transport covariance. Localized dynamical spin remains downstream. |
| DIRAC-018 | PASS (scoped classification) | Within the real local flavor-preserving cubical-incidence class, cubic covariance, exact continuum squaring and the upstream positive orientation uniquely select the same canonical root on each exact rank-four copy. Includes formal wrong-sign and broken-symmetry controls plus the C19 scalar-symbol zero census. |
| DIRAC-019 | PASS (scoped spectral transport) | Every stable positive-frequency mode follows the recurrence-derived phase with conserved density for all update counts; arbitrary finite spectral families carry the density-weighted group-velocity current, remain subluminal when massive, and converge to the derived Dirac generator. |
| DIRAC-020 | PASS (scoped coherent spatial transport) | An explicit normalized finite Fourier frame synthesizes momentum-compatible modes into one coherent spatial field with exact Parseval density, integrated current equal to the density-weighted velocity moment, nonzero current when that moment is nonzero, and an explicit finite-update pointwise phase-error bound. |
| DIRAC-021 | PASS (scoped action-derived rest clock) | The action-selected rest coefficient gives opposite positive/negative rest energies and exact Dirac time phases; their relative phase has native angular frequency `38=2*19`, with explicit native periods and inverse-mass/Compton scales. Pure global phase, SI mass, and zitterbewegung remain outside this gate. |
| DIRAC-022 | PASS (scoped action-derived free zitterbewegung) | The DIRAC-013 velocity and DIRAC-021 rest phases give an exact coherent cross-branch velocity oscillation at `2 chi0=38`; its integrated amplitude is `sqrt(12)/38` and peak-to-peak excursion is `sqrt(12)/19`. Pure, same-sign and incoherent controls have zero signal. Localized native-electron motion remains outside this gate. |

See [the complete proof chain](docs/PROOF_CHAIN.md) for every gate and its
claim boundary.

## Reproduce the formal result

Install Lean with `elan`, then run:

```bash
python scripts/verify_release.py
```

The verifier builds every production module, audits the compiled Lean
environment, checks role-specific manifests, runs adversarial audit controls,
and validates the independently reconstructed DIRAC-010--013 and DIRAC-016--022
evidence, audits the final theorem and publication metadata, and verifies the
release contract. It is read-only: it
does not install Git hooks or alter the index.

## Repository guide

- [Native framework](docs/FRAMEWORK.md): fields, constants, geometry, action,
  equations, and update.
- [Formal proof chain](docs/PROOF_CHAIN.md): all 22 gates and theorem boundary.
- [Claims and non-claims](docs/CLAIMS.md): what DIRAC-015 through DIRAC-022 establish.
- [Reproducibility](docs/REPRODUCIBILITY.md): rebuild instructions.
- [Publication-quality audit](docs/WORLD_CLASS_AUDIT.md): findings, repairs,
  residual scientific boundaries, and acceptance tests.
- [DIRAC-022 red team](docs/DIRAC022_RED_TEAM.md): observable derivation,
  controls, scope, and release findings.
- [Related work and attribution](docs/RELATED_WORK.md): the geometric
  lattice-fermion lineage and the repository's narrower formal contribution.
- [Changelog](CHANGELOG.md): current cleanup and historical milestones.
- `LeanDirac/Publication.lean`: sole cumulative publication theorem.
- `LeanDiracAudit.lean`: explicit theorem axiom reports.
- `project/`: framework, proof, evidence, release, status, and review manifests.
- `experiments/dirac010_standard_map/`: independent matrix reconstruction.
- `experiments/dirac011_frequency_mass/`: independent frequency/remainder and
  rest-mass reconstruction.
- `experiments/dirac012_continuum_evolution/`: independent action-expansion,
  centered-recurrence, and continuum-error reconstruction.
- `experiments/dirac013_density_current/`: independent native flow,
  continuity, positive-frequency density, and continuum-current reconstruction.
- `experiments/dirac016_copy_translation/`: independent copy-translation,
  C19-weight, and signed-incidence reconstruction.
- `experiments/dirac017_spin_half/`: independent spin algebra, spectrum,
  rotation-group, and continuous Hamiltonian-covariance reconstruction.
- `experiments/dirac018_factorization_classification/`: independent cubic
  commutant, square-root, two-copy, and scalar-symbol-zero reconstruction.
- `experiments/dirac019_wavepacket_transport/`: independent all-step modal
  conservation, current/velocity, timestep, and Fourier-packet reconstruction.
- `experiments/dirac020_coherent_spatial_transport/`: independent normalized
  Fourier synthesis, Parseval density, spatial current, finite-update bound,
  movement convergence, and negative-control reconstruction.
- `experiments/dirac021_rest_clock/`: independent rest-energy, Dirac time-phase,
  opposite-branch coherence, native-period, and unit-scope reconstruction.
- `experiments/dirac022_zitterbewegung/`: independent Hamiltonian-velocity,
  coherent-current, position-amplitude, and negative-control reconstruction.

## Proof-security boundary

The production audit imports every Lean source and inspects the compiled
environment. The publication theorem may use only Lean's standard foundations:
`propext`, `Classical.choice`, and `Quot.sound`. A custom axiom,
`sorry`, missing module, changed locked file, incomplete/non-finite evidence,
or stale claim status fails verification.

The native action is locked separately from the downstream continuum readout.
The cubical alpha/beta operators are not terms in the action.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json). The directly relevant prior lattice-fermion
sources are credited in [Related work and attribution](docs/RELATED_WORK.md).
Project-authored content is available under
[MIT](LICENSE) or [CC0-1.0](LICENSE-CC0), at the recipient's option. See
[LICENSING.md](LICENSING.md) for scope and third-party terms.
