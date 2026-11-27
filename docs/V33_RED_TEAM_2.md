# V3.3 Red Team 2

## Hostile review target

This pass reviewed the post-fix branch as if evaluating an external claim that
V3.3 materially changes the LFM action or completes a physical electron.

## Checks

- Confirmed the only production-facade change is the cumulative V3.3 import;
  no pre-V3.3 Lean physics module or action definition was edited.
- Re-ran the V3.2 Route-1 verifier after making its facade check cumulative.
  All V3.2 proof and claim gates passed.
- Re-ran the V3.3 cumulative build. All ten modules passed.
- Scanned every V3.3 module for `sorry`, `admit`, new `axiom`, and new opaque
  `constant` declarations. None were found.
- Checked the cumulative theorem's printed dependencies. Only Lean/mathlib's
  standard logical implementation axioms (`propext`, `Classical.choice`, and
  `Quot.sound`) remain; no project-specific physical axiom was introduced.
- Compared every public V3.3 claim with its theorem hypotheses and narrowed
  the documentation wherever a reader might infer arbitrary-momentum
  helicity, non-forward Gordon decomposition, interacting propagation,
  transport, `g = 2`, MeV calibration, or a physical electron.
- Ran `git diff --check`; no whitespace errors were found.

## Finding

Two low-severity integration issues were found. First, the historical V3
release verifier intentionally refuses a dirty checkout. Second, its sealed
source allowlist recognized `V32*` cumulative extensions but not `V33*` files.
After the first commit exposed the second issue, the allowlist was narrowly
extended to V3.2/V3.3 modules and documentation without permitting edits to
sealed V3.1 proof sources. The clean-tree verifier is rerun after that fix.

## Final pre-commit verdict

Critical open: 0. High open: 0. Medium open: 0.

The branch is suitable for a research V3.3 commit, subject to the clean-tree
historical release-verifier run documented above. V3.3 remains a free
linearized spinor-observable closure and not a physical-electron closure.
