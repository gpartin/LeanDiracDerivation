# V3.6 Hostile Red Team 3: Release Hardening

Date: 2026-11-28

## Finish line

The research release must present one logically ordered route from the
geometrically normalized live V3.6 action to the complete free linearized
Dirac conclusion, expose that route as a direct Lean theorem, provide one
working public verification command, and keep every nonlinear or physical
electron claim outside the proved boundary.

## Attacks and repairs

### High: the action and Dirac result were packaged but not directly joined

Repair: `periodicV36ResearchLagrangianTangentQuotient` is defined directly
from `v36ResearchC19Lagrangian`.
`v36_live_research_action_mode_to_free_dirac` proves the complete
finite-periodic action-to-Dirac conclusion using that live quotient. The final
V3.6 certificate carries the theorem explicitly.

Status: RESOLVED.

### High: the advertised verifier and CI invoked incompatible historical seals

Repair: `scripts/verify_release.py` is the one current gate and delegates to
the cumulative V3.6 verifier. GitHub Actions invokes only that gate. Historical
V3.2--V3.5 verifiers remain unchanged and are documented as tag-specific
evidence, not current V3.6 gates.

Status: RESOLVED.

### High: citation and archive metadata described older releases

Repair: `CITATION.cff` and `.zenodo.json` now identify V3.6, describe the
scoped common-metric result, and repeat the physical non-claims. The CFF and
JSON parse checks pass.

Status: RESOLVED.

### Medium: geometric justification appeared downstream of the completed chain

Repair: the inverse and full norm-isometry theorems now live in `Foundation`
upstream of `NativeAction`. The V3.6 geometry module no longer imports V3.5;
it packages the upstream facts and the native antisymmetric product.

Status: RESOLVED.

### Medium: the proof map omitted V3.2--V3.5

Repair: `PROOF_CHAIN.md` now records every cumulative release handoff and the
specific V3.6 action-to-result route.

Status: RESOLVED.

### Medium: historical challenge files looked like unfinished production code

Repair: the README and `HISTORICAL_SUBMISSION_ARTIFACTS.md` distinguish the
intentional V2 challenge holes from the `LeanDirac` production closure. The
files remain at their registered paths to preserve external-submission
compatibility.

Status: RESOLVED.

### Medium: accumulated documentation had no authority map

Repair: `INDEX.md` identifies current entry points, historical release records,
and external-submission artifacts. The changelog is cumulative and ordered.

Status: RESOLVED.

### Medium: contributor and vulnerability-reporting guidance was absent

Repair: `CONTRIBUTING.md` now defines the expected proof, scope, documentation,
and verification discipline. `SECURITY.md` gives a private reporting route and
separates security defects from scientific disagreements.

Status: RESOLVED.

### Low: inherited Lean warnings obscured the release signal

Repair: straightforward deprecated imports, deprecated conditional lemmas,
sequential-focus warnings, unused tactics, and unnecessary `simpa` calls were
cleaned without changing theorem statements. Remaining tactic suggestions are
informational elaborator suggestions, not proof or release failures.

Status: RESOLVED.

## Verification evidence

The authoritative command

```text
python scripts/verify_release.py
```

completed with:

```text
[PASS] V3.6 geometrically normalized action verification
V36_RELEASE_VERIFICATION_PASS
```

`LeanDiracAudit` inspected every declaration in the imported production
closure. The cumulative theorem and direct live-action theorem depend only on
`propext`, `Classical.choice`, and `Quot.sound`; neither depends on `sorryAx`
or a custom axiom. Markdown local-link checks, JSON parsing, CFF parsing, and
`git diff --check` also passed.

## Remaining non-release claims

The release still does not prove a nonlinear localized electron, nonlinear
transport, indivisible occupation, charge, exchange statistics, magnetic
response, SI/MeV calibration, Einstein closure, or canonical V40.1 promotion.
Independent human review remains the second tier of the release process and is
not represented as complete.

## Verdict

Critical findings unresolved: 0.

High findings unresolved: 0.

The V3.6 branch is technically ready for owner review as a research release.
