# Reproducibility

## Requirements

- Git
- `elan`
- Python 3

The exact Lean and Mathlib versions are pinned by `lean-toolchain` and
`lake-manifest.json`.

## Clean reproduction

```bash
git clone https://github.com/gpartin/LeanDiracDerivation.git
cd LeanDiracDerivation
lake build LeanDirac LeanDiracAudit
python scripts/verify_release.py
```

The verifier requires a clean checkout and full Git history because it checks
the sealed proof-source commit as an ancestor of the current revision. It then
builds the cumulative V3.1 dependency closure, checks every required
declaration, runs the historical-sign moving-frame regression, audits compiled
axiom dependencies, and verifies all release-facing source, documentation,
audit, and reproduction seals.

Expected final markers:

```text
LEAN_DIRAC_V3_PAULI_LUBANSKI_REGRESSION_PASS
V3_LORENTZ_SPIN_RELEASE_VERIFICATION_PASS
V3_RELEASE_VERIFICATION_PASS
```
