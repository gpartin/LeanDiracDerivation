# DIRAC V2 production port contract

V2 is developed as additive proof modules and will not rewrite the accepted
`DIRAC-001`--`DIRAC-040` history in `gpartin/LeanDiracDerivation`.

## Compatibility rules

1. Keep every published `LeanDirac/DiracNNN.lean` file and public declaration
   name intact unless a separately versioned replacement is required.
2. Add the proved `LeanDirac/V2/` modules as the implementation layer.
3. Expose compatibility theorems at the existing numbered stage boundaries.
4. Update only cumulative imports, the stage tracker, README, axiom audit, and
   release manifest needed to make the completed chain visible.
5. Rebuild the complete historical and V2 targets from a clean checkout before
   promotion.
6. Do not publish exploratory movement scripts or result directories as part
   of the production proof patch unless a release artifact explicitly cites
   them.
7. If an accepted theorem is superseded, retain it and add a versioned stronger
   theorem; never silently change its meaning.

## Current source-control evidence

The **current uncommitted V2 proof diff** does not modify
`LeanDirac/Dirac001.lean` through `LeanDirac/Dirac040.lean`.  The research
branch history, however, contains earlier numbered-stage work and must not be
merged or cherry-picked wholesale into production.

Production promotion therefore starts from the intended public-main commit
and copies an explicit allowlist consisting of the active `LeanDirac/V2/`
proof modules plus the reviewed release artifacts and entry-point updates.
The resulting production diff must be checked independently to confirm that
no numbered historical file changed.

The clean-base build established one required compatibility patch.  The five
non-numbered foundation/audit files `Foundation.lean`, `ExteriorArea.lean`,
`C19Geometry.lean`, `NativeAction.lean`, and `EnvironmentAudit.lean` must use
Lean 4.33 `module`/`public import` headers so that the new module-form V2 files
can import them.  Their mathematical declarations are unchanged.  Without
that header-only conversion Lean rejects the import before checking any V2
theorem; with it the clean V2 target builds.

## Promotion sequence

```text
research V2 modules
  -> final compiled-environment and adversarial audit
  -> fresh branch from the intended public-main commit
  -> allowlisted minimal production patch
  -> verify no historical numbered source changed
  -> clean historical + V2 build
  -> review
  -> public main
  -> follow-up submission to maintained downstream mirrors
```
