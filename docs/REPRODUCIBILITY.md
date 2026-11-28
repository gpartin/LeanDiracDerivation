# Reproducibility

## Requirements

- Git
- `elan`
- Python 3

The exact Lean and Mathlib revisions are pinned by `lean-toolchain` and
`lake-manifest.json`.

## Clean reproduction

```bash
git clone https://github.com/gpartin/LeanDiracDerivation.git
cd LeanDiracDerivation
python scripts/verify_release.py
```

`scripts/verify_release.py` is the only authoritative current release command.
It delegates to the V3.6 verifier, which:

1. checks all seven V3.6 handoff modules and the revised foundation/action;
2. checks the direct live-action theorem and current release metadata;
3. builds the complete `LeanDirac` facade;
4. builds `LeanDiracAudit`, whose compiler-environment audit examines every
   declaration in every imported production module;
5. rejects `sorryAx`, custom axioms, and nonstandard axiom dependencies; and
6. confirms that the cumulative V3.6 theorem and direct live-action theorem
   use only `propext`, `Classical.choice`, and `Quot.sound`.

Expected final markers:

```text
[PASS] V3.6 geometrically normalized action verification
V36_RELEASE_VERIFICATION_PASS
```

## Historical verifiers

The `verify_v32_research.py` through `verify_v35_research.py` scripts preserve
the release seals used by those versions. They intentionally reject changes to
the then-sealed foundation and native action. V3.6 deliberately revises those
files, so historical verifiers are evidence for their tagged releases, not
valid gates for the current V3.6 tree.

## Production and challenge surfaces

`import LeanDirac` is the production facade. `Challenge.lean` is a preserved
V2 external-submission challenge and intentionally contains theorem holes;
`Solution.lean` supplies its answers. Neither is imported by `LeanDirac` or
`LeanDiracAudit`. See `HISTORICAL_SUBMISSION_ARTIFACTS.md` for the complete
classification.
