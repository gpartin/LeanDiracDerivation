# Historical external-submission artifacts

The current production entry point is `import LeanDirac`; the current release
gate is `python scripts/verify_release.py`.

The following root files preserve the narrower V2 challenge submitted to
external formalization services:

| File | Role | Production dependency? |
|---|---|---|
| `Challenge.lean` | Problem statement with intentional theorem holes | No |
| `Solution.lean` | Machine-checked answers to the challenge | No |
| `comparator.json` | External comparator declaration list | No |
| `formalization.yaml` | Historical V2 submission metadata | No |
| `project/dirac_v2_stage_status.json` | Historical V2 status record | No |
| `.github/workflows/palomar-v2-preflight.yml` | Pinned historical preflight | No |

The challenge file's theorem holes are not unfinished production proofs. The
production no-`sorry` claim applies to the transitive `LeanDirac` facade and is
enforced by `LeanDiracAudit`.

These files remain at their registered paths to avoid silently breaking an
external submission contract. Their V2 labels are historical provenance, not
the current repository release version.
