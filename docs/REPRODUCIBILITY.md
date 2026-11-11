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

## Trust boundary

The Lean theorem is the formal evidence. The Python result is an independent
matrix reconstruction and cannot substitute for Lean. Hash manifests separate
the native framework, formal proof, evidence, and publication files.
