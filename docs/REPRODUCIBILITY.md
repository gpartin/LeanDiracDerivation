# Reproducibility

## Formal proof

From a clean checkout:

```bash
lake build
python scripts/verify_release.py
```

The verification command rebuilds every production Lean module, audits all
compiled declarations for nonstandard axioms or `sorry`, checks the locked
manifests, runs negative controls, and verifies the derivation contract.

## Independent DIRAC-010 reconstruction

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python experiments/dirac010_standard_map/validate_dirac010.py
```

The program independently constructs the three cubical operators, beta,
flavor grading, and both rank-four embeddings. It checks the squares,
anticommutators, copy ranks, and flavor eigenvalues at tolerance `1e-12`.
It does not test physical mass, electron stability, or spin.

## Independent DIRAC-011 reconstruction

```bash
.venv/Scripts/python experiments/dirac011_frequency_mass/validate_dirac011.py
```

The program composes the DIRAC-010 copy embeddings with their standard-basis
maps, reconstructs both frequency signs, checks that the exact scalar evolution
differs from the continuum graph only by the explicit C19 remainder, and checks
the nonnegative rest-mass selection at tolerance `1e-11`. It does not prove the
full continuum equation, current, physical spin, or an electron.

## Independent DIRAC-012 reconstruction

```bash
.venv/Scripts/python experiments/dirac012_continuum_evolution/validate_dirac012.py
```

The program independently reconstructs the C19 symbol from all 18 weighted
directions, the native action's quadratic modal expansion, the centered-time
mode defect, and the combined spatial/time bound. It also reconstructs the
first-order branch from the centered recurrence, verifies its exact recurrence,
unitarity in the stable band, exact native C19 defect, and generator defect.
It does not test density/current, nonlinear live-chi evolution, physical spin,
or an electron.

## Independent DIRAC-013 reconstruction

```bash
.venv/Scripts/python experiments/dirac013_density_current/validate_dirac013.py
```

The program independently differentiates the native bond energy, checks the
reverse-bond law and exact local continuity on periodic C19 fields, reconstructs
the positive-frequency density normalization and both cube-copy density scales,
and checks that the exact finite C19 flow approaches the standard Dirac current
at quadratic order. It does not claim positive density for mixed frequencies,
exact finite-lattice current equality, nonlinear live-chi current closure,
physical spin, or an electron.

## Independent DIRAC-016 reconstruction

```bash
.venv/Scripts/python experiments/dirac016_copy_translation/validate_dirac016.py
```

The program reconstructs all three unsigned cube translations, both rank-four
embeddings, their coordinate half-shift, the signed-incidence contrast, and the
C19 copy-class weight split. It does not claim that one native timestep
performs a copy exchange or that a localized carrier moves this way.

## Independent DIRAC-017 reconstruction

```bash
.venv/Scripts/python experiments/dirac017_spin_half/validate_dirac017.py
```

The program reconstructs the native spin algebra, both-copy spectrum,
quarter-turn and continuous rotation identities, and all-angle interpreted
Hamiltonian covariance. It checks the conditional representation; it is not
evidence for localized dynamical spin or a physical electron.

## Independent DIRAC-018 reconstruction

```bash
.venv/Scripts/python experiments/dirac018_factorization_classification/validate_dirac018.py
```

The program independently reconstructs the cubic commutant, the canonical
square identity, the two exact rank-four flavor projectors, the wrong-sign and
reflected-axis controls, and the scalar C19 zero census on every periodic grid
from size 2 through 32. It tests the stated local flavor-preserving coefficient
class; it is not evidence for arbitrary-root uniqueness, an exact finite-
spacing Dirac factorization, a universal doubling theorem, or an electron.

## Independent DIRAC-019 reconstruction

```bash
.venv/Scripts/python experiments/dirac019_wavepacket_transport/validate_dirac019.py
```

The program reconstructs stable all-step modal density conservation, the
density-weighted current/group-velocity identity, strict massive speed bound,
and generator refinement. It also Fourier-synthesizes a narrow linear packet
as a diagnostic and checks its velocity convergence without inserting a
translation rule. This is not a nonlinear localized-electron simulation.

## Independent DIRAC-020 reconstruction

```bash
.venv/Scripts/python experiments/dirac020_coherent_spatial_transport/validate_dirac020.py
```

The program reconstructs the normalized finite Fourier frame, coherent
momentum-character correspondence, Parseval density, integrated spatial
current, density-weighted velocity and
finite-update pointwise error bound. It also tracks a narrow packet under
timestep refinement and rejects unnormalized, zero-current and unstable-step
controls. It is not a nonlinear live-chi electron simulation.

## Independent DIRAC-021 reconstruction

```bash
.venv/Scripts/python experiments/dirac021_rest_clock/validate_dirac021.py
```

The program reconstructs the zero-momentum `+chi0` and `-chi0` rest energies,
their first-order time equations, exact periods, and the doubled relative
phase of a coherent opposite-branch pair. It rejects a wrong-mass control and
a same-sign control and confirms that either pure branch has constant density.
It does not turn global phase into an observable, calibrate an SI mass, or
establish zitterbewegung.

## Independent DIRAC-022 reconstruction

```bash
.venv/Scripts/python experiments/dirac022_zitterbewegung/validate_dirac022.py
```

The program reconstructs the rest Hamiltonian and its momentum-response
velocity, the normalized coherent opposite-branch state, its `2 chi0`
velocity expectation, the integrated position law and native amplitude.  It
also runs pure-branch, same-sign, incoherent-mixture and wrong-mass controls.
It does not simulate a localized nonlinear native electron.

## Independent DIRAC-023 reconstruction

```bash
.venv/Scripts/python experiments/dirac023_phase_harmony/validate_dirac023.py
```

The program reconstructs the continuum mode energy, current-derived transport
velocity, packet-center phase rate, group/phase reciprocity, energy/clock
reciprocity, relativistic clock factor, and inherited zitter scale lock for
rest, axial, oblique, signed and higher-momentum samples. It rejects altered
transport and rest-frequency controls. It does not test exact finite-spacing
time dilation or a localized nonlinear native electron.

## Independent DIRAC-024 reconstruction

```bash
.venv/Scripts/python experiments/dirac024_projective_zitter_spin/validate_dirac024.py
```

The program reconstructs the action-derived zitter period, one-cycle spinor
antiperiod, two-cycle state return, periodic density/velocity/position, and
the `2π`/`4π` endpoint bridges for all three axes and both native copies. It
rejects a half-cycle return and a period shortened by ten percent, and checks
that the temporal and spatial generators remain distinct. It does not equate
time evolution with spatial rotation or simulate a localized nonlinear
electron.

## Independent DIRAC-025 reconstruction

```bash
.venv/Scripts/python experiments/dirac025_velocity_clock/validate_dirac025.py
```

The program reconstructs the four-component Hamiltonian and velocity
matrices, checks five oblique momenta and both positive-energy spin states,
reproduces the exact `3/5 -> 4/5`, `4/5 -> 3/5`, and
`sqrt(3)/2 -> 1/2` ratios, and rejects unnormalized-direction,
transverse-direction and wrong-mass inputs.

## Independent DIRAC-026--030 reconstruction

```bash
.venv/Scripts/python experiments/dirac026_030_free_electron/validate_dirac026_030.py
```

The program independently reconstructs the Lorentz generator algebra, finite
boost, Pauli--Lubanski invariant, infinitesimal Wigner--Thomas factor, free
charge-conjugation intertwiner and Gordon kernel. It also rejects a quarter
Thomas factor, omission of boost terms, failure to reverse momentum, and
omission of the Gordon spin term.

## Final audit and deterministic package

```bash
python scripts/final_chain_audit.py
python scripts/build_publication_package.py --output-dir dist
```

The first command verifies the complete 30-gate dependency and claim boundary.
The second packages exactly the committed Git tree with fixed timestamps,
writes
an internal SHA-256 manifest, reopens the archive, and verifies every byte. It
produces `dist/LeanDiracDerivation-DIRAC-030.zip` and its checksum. The archive
also embeds `PROVENANCE.json` with the exact Git commit, Lean toolchain, action
identifier, highest gate and publication theorem. These steps
test reproducibility; they do not add a mathematical or physical premise.
Reading the committed Git objects, rather than checkout-dependent working-tree
line endings, makes the archive identical across platforms.

## Trust boundary

The Lean theorem is the formal evidence. The Python result is an independent
matrix reconstruction and cannot substitute for Lean. Hash manifests separate
the native framework, formal proof, evidence, and publication files.

The dependency graph is locked by `lake-manifest.json`; reproduction therefore
does not run `lake update`. Updating dependencies is a separate reviewed
maintenance operation, not part of reproducing a published result.
