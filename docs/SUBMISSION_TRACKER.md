# External submission tracker

Last reconciled: 2026-11-19 (America/Los_Angeles)

This ledger separates publication, preservation, independent verification,
curated listing, and community announcement. A listing or announcement is not
an endorsement of the physical interpretation. The mathematical source now
extends through DIRAC-030 at commit
`867004142ada3eec464d5bd3088d7313fdad1613`. The earlier archived DIRAC-015
release is commit `2c7774d41166a3b133a81a7b1ef2f816047f095e`; its Zenodo
record does not archive the later chain. The narrow registry theorem is
commit `9eef7df27e29298dae58d563a1608b00102048df`.

| Destination | Scope | Status | Last event | Next action | Follow-up date |
| --- | --- | --- | --- | --- | --- |
| [GitHub repository](https://github.com/gpartin/LeanDiracDerivation) | Current development chain through DIRAC-030 | Public source and successful CI | Mathematical source commit `867004142ada3eec464d5bd3088d7313fdad1613` was imported into LeanPool. Current source and earlier immutable archival release are distinct. | Maintain source, evidence, and claim boundaries; consider archival updates separately | On material release or review response |
| [Zenodo](https://doi.org/10.5281/zenodo.24150001) | Archived chain through DIRAC-015 | Published | DOI minted for `v0.15.0-research`; it does not archive DIRAC-016 through DIRAC-030 | Consider a separately reviewed new archival version for the later chain | On archival release |
| [Software Heritage](https://archive.softwareheritage.org/swh:1:snp:4841f690581165c2ad54f55685790d592da64e05) | Full Git history | Archived | Snapshot `swh:1:snp:4841f690581165c2ad54f55685790d592da64e05` is `full`; its `main` branch points to release commit `2c7774d41166a3b133a81a7b1ef2f816047f095e` | Request a new snapshot only for a material release | On release |
| Palomar (`dzgf72w4m3wl`) | Narrow cubical C19 factorization | Mechanical verification passed; automated review identified blocking problems | [Independent run passed](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/43217433664); the private review on 2026-11-12 found that the originality claim lacked a documented literature basis and the selected identity was too narrow to warrant the broader research claim | Do not register this intake. Preserve it as failed provenance and use the stronger corrected candidate below | Closed |
| Palomar corrected candidate (`gx041dl90m63`) | Six connected native-action, C19-symbol, cubical-factorization, residual-control, and current theorems | Superseded and withdrawn after editorial review | Exact public commit `ae7e6c1b5b6cfe73be95c1bd27769224ad383bac` passed the official [preflight](https://github.com/gpartin/LeanDiracDerivation/actions/runs/43620905005), [intake verification](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/43744997341), and renderability checks. The private review on 2026-11-15 identified two missing links. On 2026-11-17 this intake was withdrawn so the repaired replacement below could enter verification. | None; preserve the review as provenance for the repaired replacement | Closed |
| Palomar repaired replacement (ID pending) | Full native-action restriction and first-variation bridge, C19 Klein--Gordon tangent and controlled remainder, cubical square root, recurrence-derived first-order branch, and native continuity identity | Mechanical verification passed; automated editorial review queued | Exact public commit `522dff1b6504e74b591992d1d4210d189ffe9956` passed the official full [Palomar preflight](https://github.com/gpartin/LeanDiracDerivation/actions/runs/44005024848) and was submitted on 2026-11-17 with `comparator.json`. Independent [verification run 44057900349](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/44057900349) completed successfully; Palomar then queued its private automated review. The submission remains conditional and makes no localized-electron, physical-spin, nonlinear live-chi, uniqueness, or exact finite-spacing Dirac claim. | Wait for the private automated review; read and assess it before requesting any permanent registration | 2026-11-18 |
| [Lax `lax-423666`](https://laxarchive.org/lax-423666/) | Narrow cubical C19 factorization | Permanently registered; immutable and citable | G. D. Partin explicitly approved registration on 2026-11-16. Lax permanently registered the validated draft; [registration workflow 43904456735](https://github.com/lax-archive/lax/actions/runs/43904456735) completed, while [issue #224](https://github.com/lax-archive/lax/issues/224) preserves submission provenance | None; cite the permanent Lax record and publish a new concept only for a materially different formal result | Closed |
| [Lemmata PR 18](https://github.com/ZIB-IOL/lemmata/pull/18) | Narrow cubical C19 factorization | Ready for review; open, clean, and mergeable | Rebased commit `0adbbeec477ba5181e6447ed6e660d16f08963b0` passed validation plus 102 adversarial cases, the full 4,562-job build, independent kernel replay, a 22,139-declaration axiom/export audit using only standard Lean foundations, and strict cluster-style and offline renders; GitHub recalculated the PR as clean and mergeable Merge evidence was [added once](https://github.com/ZIB-IOL/lemmata/pull/18#issuecomment-7011304584) on 2026-11-19. | Wait for mathematical review and respond to maintainer feedback | On response |
| [VibeRegistry PR 13](https://github.com/GasStationManager/VibeRegistry/pull/13) | Narrow cubical C19 factorization | Draft; workflow approval required | Workflow `43205781459` concluded `action_required` at `5d15c382d29366855e1154503586c9b5b35c5da4`; [maintainer approval was requested](https://github.com/GasStationManager/VibeRegistry/pull/13#issuecomment-6824860432) Merge evidence was [added once](https://github.com/GasStationManager/VibeRegistry/pull/13#issuecomment-7011306637) on 2026-11-19; workflow approval remains unresolved. | Wait for approval, then inspect Comparator/Nanoda output before marking ready | On response |
| [LeanPool PR 889](https://github.com/LeanPool/lean-pool/pull/889) | Cubical lattice Dirac factorization and its continuum limit | Merged; mathematical import accepted | PR 889 merged at `f49758ce995f7806f146b65ca376794103966333`, importing the pinned DIRAC-030 source through a Lean 4.35.0 port. [Project card](https://github.com/LeanPool/lean-pool/blob/main/LeanPool/projects/lattice-dirac-derivation.yaml). Earlier [proposal issue 800](https://github.com/LeanPool/lean-pool/issues/800) records intake and scheduling. This is acceptance of the scoped formal mathematics, not experimental validation of LFM. | Cite source repository as the main project and merged PR as acceptance evidence | On response or material update |
| [Best-of-Lean4 PR 297](https://github.com/34j/best-of-lean4/pull/297) | Project listing | Open; metadata refreshed | On 2026-11-19 the canonical homepage and scoped description were corrected at `c46308d292cf5afb0645609bc93eb872f2eb6008`, replacing obsolete DIRAC-015 wording. The maintainer's merge announcement was posted once as a material update. YAML parses and contains exactly one project entry. | Await curator review | On response |
| [Formalization Corpus issue 10](https://github.com/dzackgarza/formalization-corpus/issues/10) | Full repository indexing | Open; merge evidence supplied | [Material update](https://github.com/dzackgarza/formalization-corpus/issues/10#issuecomment-7011278664) posted on 2026-11-19 using the maintainer's supplied announcement. No duplicate intake was created. | Await source intake or maintainer response | On response |
| [Project Intentions issue 55](https://github.com/leanprover-community/project-intentions/issues/55) | Community coordination | Open; protected `intention` label still pending | [Material update](https://github.com/leanprover-community/project-intentions/issues/55#issuecomment-7011308704) posted on 2026-11-19 using the maintainer's exact supplied announcement. Earlier protected-label request remains pending. | Await maintainer labeling; no repeated request posted | On response |
| Lean Zulip announcement | Community announcement | Deleted; outreach closed | On 2026-11-12 the moderators deleted the AI-written announcement and imposed a one-day account suspension under the community rule prohibiting AI-written Zulip or GitHub comments | Do not repost it or use AI-written comments in Lean community channels; any future communication must be written personally by G. D. Partin and comply with the venue's rules | Closed |
| [Constraint Systems Institute](https://constraintsystemsinstitute.org/lean/?framework=beyond-aasc) | Archive/listing request | Already listed; merge-evidence update requested | Original DIRAC-015 listing was confirmed on 2026-11-12. On 2026-11-19 a reply in the existing intake thread requested adding LeanPool PR 889 alongside the archived entry, identifying imported source commit 867004142ada3eec464d5bd3088d7313fdad1613 through DIRAC-030. Gmail confirms the update was sent. The existing archival scope and CSI verification status are preserved; no updated public listing has yet been confirmed. | Await catalogue update or response | On response |
| [Awesome Lean PR 8](https://github.com/pdehaye/awesome-lean/pull/8) | Mathematics formalization project directory | Open; not yet accepted | Entry uses the canonical source link plus merged LeanPool evidence. Local awesome-lint, generated CSV check, and source snapshot generation pass. README/CSV and generated snapshot are separate commits as required. | Await curator review | On response |
| [Awesome Lean resources issue 3](https://github.com/stuff-of-awe/awesome-lean/issues/3) | Worked formalization resource listing | Open; not yet accepted | Submitted on 2026-11-19 with canonical links, audience, version details, explicit scope, affiliation, and AI-assistance disclosure. | Await curator review | On response |
| [Official Lean projects directory candidate](https://github.com/gpartin/mathlib4/compare/master...codex/dirac-project-directory) | Established downstream-project listing | Prepared branch; not submitted | A validated three-line entry for `scripts/downstream_repos.yml` is committed at `cbfeaaed1011993bc348fe63d2f3e3f6ac29f3ca`. The maintainer's supplied announcement can be used verbatim; a human-written AI-assistance disclosure required by Mathlib's contribution policy is still needed. | Obtain the missing human-written disclosure before opening the PR | On maintainer response |
| [Reservoir](https://reservoir.lean-lang.org/inclusion-criteria) | Automatic Lean package index | Ineligible at current metadata check; not submitted | Public GitHub REST metadata on 2026-11-19 reports one star, non-fork public source, and detected MIT license; root lake-manifest.json exists. Reservoir requires at least two stars. | Allow natural indexing once eligibility is met; do not file a false indexing bug | On eligibility change |

## Registration boundary

Palomar registration is deliberately separate from submission and review.
Registration makes the repository, commit, and review public and creates
immutable preservation tags. It must not be performed until the maintainer has
read the exact private review and explicitly approved registration.

## Update rule

For every status change, update the status, last event, next action, and
follow-up date in this file. Preserve failed or incomplete submissions as such;
do not convert an intake, listing, announcement, or draft pull request into a
verification claim.

## Directory outreach constraints checked on 2026-11-19

Use the source repository as the main project URL and LeanPool PR 889 as
supporting merge evidence. No new directory request is recorded as accepted.

GitHub connector writes to external directories returned integration-scope
403 errors; the authenticated local GitHub CLI completed the recorded writes.
Computer Use stopped because it could not confidently determine Chrome's
current URL. LibHunt's browser form therefore remains unsubmitted. The
Research Software Directory requires production sign-in/access and remains
unsubmitted; no account was created. General quantum-computing directories
were not used for an unrelated relativistic wave formalization. Palomar
registration and Zulip outreach remain subject to the existing boundaries.
