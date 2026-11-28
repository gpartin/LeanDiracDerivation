# V3.6 Hostile Red Team 2: Release and Verification

Date: 2026-11-28

## Scope

This review attacked repository consistency, verifier design, stage ordering,
release naming, and the boundary between the branch-local research action and
repository-wide canonical v40.1.

## Findings and repairs

### High: historical verifiers cannot certify an intentional action change

The V3.2 verifier correctly rejects any modification to `Foundation.lean` or
`NativeAction.lean`. Calling it from V3.6 caused a failure even though the new
cumulative theorem compiled.

Repair: the historical verifier remains unchanged. The V3.6 verifier now
builds the complete cumulative facade and compiled environment audit under the
revised action, scans the changed foundation/action and every V3.6 source, and
audits the transitive cumulative theorem for `sorryAx` and nonstandard axioms.

Status: RESOLVED without weakening a historical seal.

### High: planning and proof-chain documents still said no action changed

Two files retained the initial audit-only interpretation and contradicted the
actual code.

Repair: the plan, dependency matrix, framework, claims, README, changelog, and
proof chain now state that V3.6 changes the branch-local research action from
`B = 1197` to `B = 1`, while preserving the old values as controls.

Status: RESOLVED.

### Medium: release-facing names still described an audit

The final module and plan names preserved the initial misunderstanding.

Repair: the release-facing module is now
`V36CompleteGeometricallyNormalizedAction`, the cumulative theorem is
`v36_complete_geometrically_normalized_action`, and the plan filename matches.

Status: RESOLVED.

### Medium: stage order and status were stale

The status files listed the framework lock before the geometric derivation and
marked compiled stages as in progress.

Repair: the seven stages now follow the actual dependency order and are marked
PASS only after the cumulative rebuild and axiom audit.

Status: RESOLVED.

### Low: inherited linter and deprecation warnings remain

The cumulative build reports pre-existing Mathlib deprecation and tactic
linter warnings in older modules. They do not change theorem statements or
introduce unsoundness.

Status: ACCEPTED as inherited non-blocking maintenance work.

## Final release verdict

Critical findings: 0 unresolved.

High findings: 0 unresolved.

V3.6 is locally release-ready as a research branch after the strict verifier
passes. Promotion to public main or to repository-wide canonical v40.1 is a
separate owner decision and is not implied by this review.
