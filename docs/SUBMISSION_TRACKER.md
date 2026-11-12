# External submission tracker

Last reconciled: 2026-11-12 (America/Los_Angeles)

This ledger separates publication, preservation, independent verification,
curated listing, and community announcement. A listing or announcement is not
an endorsement of the physical interpretation. Unless stated otherwise, the
public release is commit `2c7774d41166a3b133a81a7b1ef2f816047f095e` and the
narrow registry theorem is commit
`9eef7df27e29298dae58d563a1608b00102048df`.

| Destination | Scope | Status | Last event | Next action | Follow-up date |
| --- | --- | --- | --- | --- | --- |
| [GitHub repository](https://github.com/gpartin/LeanDiracDerivation) | Full 15-gate release | Published | Release `v0.15.0-research` is public | Maintain reproducibility and respond to issues | On change |
| [Zenodo](https://doi.org/10.5281/zenodo.24150001) | Full archived release | Published | DOI minted | Publish a new version only for a material release | On release |
| [Software Heritage](https://archive.softwareheritage.org/swh:1:snp:4841f690581165c2ad54f55685790d592da64e05) | Full Git history | Archived | Snapshot `swh:1:snp:4841f690581165c2ad54f55685790d592da64e05` is `full`; its `main` branch points to release commit `2c7774d41166a3b133a81a7b1ef2f816047f095e` | Request a new snapshot only for a material release | On release |
| Palomar (`dzgf72w4m3wl`) | Narrow cubical C19 factorization | Verification passed; review pending | [Independent run passed](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/43217433664); private status/review API subsequently returned HTTP 500 | Do not duplicate. Retry status; report the service problem if it persists. Show the private review to the maintainer before registration | 2026-11-13 |
| [Lax `lax-423666`](https://github.com/lax-archive/lax/issues/224) | Narrow cubical C19 factorization | Intake initialized; content not registered | Database initialization succeeded at archive commit `4deef5d284798f22ac38ea744d58f72861e36b9d`; the pinned `lax-archive@0.1.51` client reproduced the contradictory inspector failure and the [exact diagnostic was reported](https://github.com/lax-archive/lax/issues/224#issuecomment-6824336363) | Wait for archive guidance, rerun validation exactly as directed, then register only after a clean result | On response |
| [Lemmata PR 18](https://github.com/ZIB-IOL/lemmata/pull/18) | Narrow cubical C19 factorization | Ready for review; open and mergeable | Full build, 748-module kernel replay, 22,118-declaration axiom audit, cluster render, and offline render passed at `fe38b03`; the PR documents a repository-wide Windows browser-route failure reproduced on untouched upstream `main` | Wait for mathematical review and respond to maintainer feedback | On response |
| [VibeRegistry PR 13](https://github.com/GasStationManager/VibeRegistry/pull/13) | Narrow cubical C19 factorization | Draft; workflow approval required | Workflow `43205781459` concluded `action_required` at `5d15c382d29366855e1154503586c9b5b35c5da4`; [maintainer approval was requested](https://github.com/GasStationManager/VibeRegistry/pull/13#issuecomment-6824860432) | Wait for approval, then inspect Comparator/Nanoda output before marking ready | On response |
| [Lean Pool issue 800](https://github.com/LeanPool/lean-pool/issues/800) | Full project proposal | Open; no response | Proposal filed | Wait for fit/licensing guidance; prepare a content PR only if invited | 2026-11-19 |
| [Best-of-Lean4 PR 297](https://github.com/34j/best-of-lean4/pull/297) | Project listing | Open, non-draft, mergeable | Listing PR filed at `9ed687d1caf405907b4419a42dd41c0c67c8feb5` | Respond to review; send one polite ping only if inactive | 2026-11-19 |
| [Formalization Corpus issue 10](https://github.com/dzackgarza/formalization-corpus/issues/10) | Full repository indexing | Open; no response | Inclusion request filed | Ask whether a direct source-list PR is preferred if no response | 2026-11-26 |
| [Project Intentions issue 55](https://github.com/leanprover-community/project-intentions/issues/55) | Community coordination | Open; topic labels applied; protected `intention` label pending | Maintainer label request posted | Wait for a maintainer to apply the protected label | 2026-11-19 |
| [Lean Zulip announcement](https://leanprover.zulipchat.com/#narrow/channel/579630-Project-announcements/topic/LeanDiracDerivation.3A.20lattice.20KG.20to.20free.20Dirac/with/639628422.01) | Community announcement | Posted | Announcement published | Answer questions; post only material verification or archive milestones | On response |
| [Constraint Systems Institute](https://constraintsystemsinstitute.org/) | Archive/listing request | Email sent; no response | Request sent to `amos@constraintsystemsinstitute.org` | Send one concise follow-up if unanswered | 2026-11-23 |

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
