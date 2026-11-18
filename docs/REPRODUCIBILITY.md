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

## Final audit and deterministic package

```bash
python scripts/final_chain_audit.py
python scripts/build_publication_package.py --output-dir dist
```

The first command verifies the complete 18-gate dependency and claim boundary.
The second packages exactly the committed Git tree with fixed timestamps,
writes
an internal SHA-256 manifest, reopens the archive, and verifies every byte. It
produces `dist/LeanDiracDerivation-DIRAC-018.zip` and its checksum. The archive
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
