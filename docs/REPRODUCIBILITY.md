# Reproducibility

## Formal proof

From a clean checkout:

```bash
lake update
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
mode defect, the combined spatial/time bound, and the normalized equation
error under refinement. It does not test density/current, nonlinear live-chi
evolution, physical spin, or an electron.

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

## Trust boundary

The Lean theorem is the formal evidence. The Python result is an independent
matrix reconstruction and cannot substitute for Lean. Hash manifests separate
the native framework, formal proof, evidence, and publication files.
