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

The cumulative source tree is flattened under `LeanDirac/`; release-version
directories are not part of the production API. Because V2 and V3.1 were
originally published under `LeanDirac/V2` and the `LeanDirac.V2` namespace,
the verifier also compares every migrated pre-V3.2 module against its sealed
source after applying only that path/namespace substitution. Any additional
change to a migrated theorem source fails the seal audit. The original layout
remains reproducible from the corresponding Git tags.

Expected final markers:

```text
LEAN_DIRAC_V3_PAULI_LUBANSKI_REGRESSION_PASS
V3_LORENTZ_SPIN_RELEASE_VERIFICATION_PASS
V3_RELEASE_VERIFICATION_PASS
```
