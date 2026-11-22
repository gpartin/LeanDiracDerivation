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
builds the V2 dependency closure, checks every required declaration, audits
compiled axiom dependencies, and verifies the source seals.

Expected final markers:

```text
v2_first_principles_free_dirac: PASS_WITH_SCOPED_OPEN_GATES_AND_COMPILED_AXIOM_AUDIT
V2_RELEASE_VERIFICATION_PASS
```
