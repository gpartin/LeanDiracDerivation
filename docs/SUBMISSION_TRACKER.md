# External submission tracker

Last reconciled: 2026-11-17 (America/Los_Angeles)

This ledger separates publication, preservation, independent verification,
curated listing, and community announcement. A listing or announcement is not
an endorsement of the physical interpretation. Unless stated otherwise, the
public release is commit `2c7774d41166a3b133a81a7b1ef2f816047f095e` and the
narrow registry theorem is commit
`9eef7df27e29298dae58d563a1608b00102048df`.

| Destination | Scope | Status | Last event | Next action | Follow-up date |
| --- | --- | --- | --- | --- | --- |
| [GitHub repository](https://github.com/gpartin/LeanDiracDerivation) | Full 15-gate release | Published | Release `v0.15.0-research` is public; the Git tag and GitHub release remain mutable references, while the linked Zenodo record and Software Heritage snapshot provide immutable preservation | Maintain reproducibility and respond to issues | On change |
| [Zenodo](https://doi.org/10.5281/zenodo.24150001) | Full archived release | Published | DOI minted | Publish a new version only for a material release | On release |
| [Software Heritage](https://archive.softwareheritage.org/swh:1:snp:4841f690581165c2ad54f55685790d592da64e05) | Full Git history | Archived | Snapshot `swh:1:snp:4841f690581165c2ad54f55685790d592da64e05` is `full`; its `main` branch points to release commit `2c7774d41166a3b133a81a7b1ef2f816047f095e` | Request a new snapshot only for a material release | On release |
| Palomar (`dzgf72w4m3wl`) | Narrow cubical C19 factorization | Mechanical verification passed; automated review identified blocking problems | [Independent run passed](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/43217433664); the private review on 2026-11-12 found that the originality claim lacked a documented literature basis and the selected identity was too narrow to warrant the broader research claim | Do not register this intake. Preserve it as failed provenance and use the stronger corrected candidate below | Closed |
| Palomar corrected candidate (`gx041dl90m63`) | Six connected native-action, C19-symbol, cubical-factorization, residual-control, and current theorems | Superseded and withdrawn after editorial review | Exact public commit `ae7e6c1b5b6cfe73be95c1bd27769224ad383bac` passed the official [preflight](https://github.com/gpartin/LeanDiracDerivation/actions/runs/43620905005), [intake verification](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/43744997341), and renderability checks. The private review on 2026-11-15 identified two missing links. On 2026-11-17 this intake was withdrawn so the repaired replacement below could enter verification. | None; preserve the review as provenance for the repaired replacement | Closed |
| Palomar repaired replacement (ID pending) | Full native-action restriction and first-variation bridge, C19 Klein--Gordon tangent and controlled remainder, cubical square root, recurrence-derived first-order branch, and native continuity identity | Submitted; mechanical verification queued | Exact public commit `522dff1b6504e74b591992d1d4210d189ffe9956` passed the official full [Palomar preflight](https://github.com/gpartin/LeanDiracDerivation/actions/runs/44005024848) and was submitted on 2026-11-17 with `comparator.json`. Palomar queued [verification run 44057900349](https://github.com/PalomarRegistry/PalomarSubmission/actions/runs/44057900349). The submission remains conditional and makes no localized-electron, physical-spin, nonlinear live-chi, uniqueness, or exact finite-spacing Dirac claim. | Wait for mechanical verification and private automated review; read the review before requesting any permanent registration | 2026-11-18 |
| [Lax `lax-423666`](https://laxarchive.org/lax-423666/) | Narrow cubical C19 factorization | Permanently registered; immutable and citable | G. D. Partin explicitly approved registration on 2026-11-16. Lax permanently registered the validated draft; [registration workflow 43904456735](https://github.com/lax-archive/lax/actions/runs/43904456735) completed, while [issue #224](https://github.com/lax-archive/lax/issues/224) preserves submission provenance | None; cite the permanent Lax record and publish a new concept only for a materially different formal result | Closed |
| [Lemmata PR 18](https://github.com/ZIB-IOL/lemmata/pull/18) | Narrow cubical C19 factorization | Ready for review; open, clean, and mergeable | Rebased commit `0adbbeec477ba5181e6447ed6e660d16f08963b0` passed validation plus 102 adversarial cases, the full 4,562-job build, independent kernel replay, a 22,139-declaration axiom/export audit using only standard Lean foundations, and strict cluster-style and offline renders; GitHub recalculated the PR as clean and mergeable | Wait for mathematical review and respond to maintainer feedback | On response |
| [VibeRegistry PR 13](https://github.com/GasStationManager/VibeRegistry/pull/13) | Narrow cubical C19 factorization | Draft; workflow approval required | Workflow `43205781459` concluded `action_required` at `5d15c382d29366855e1154503586c9b5b35c5da4`; [maintainer approval was requested](https://github.com/GasStationManager/VibeRegistry/pull/13#issuecomment-6824860432) | Wait for approval, then inspect Comparator/Nanoda output before marking ready | On response |
| [Lean Pool issue 800](https://github.com/LeanPool/lean-pool/issues/800) | Full project proposal | Open; no response | Proposal filed | Wait for fit/licensing guidance; prepare a content PR only if invited | 2026-11-19 |
| [Best-of-Lean4 PR 297](https://github.com/34j/best-of-lean4/pull/297) | Project listing | Open, non-draft, mergeable | Listing PR filed at `9ed687d1caf405907b4419a42dd41c0c67c8feb5` | Respond to review; send one polite ping only if inactive | 2026-11-19 |
| [Formalization Corpus issue 10](https://github.com/dzackgarza/formalization-corpus/issues/10) | Full repository indexing | Open; no response | Inclusion request filed | Ask whether a direct source-list PR is preferred if no response | 2026-11-26 |
| [Project Intentions issue 55](https://github.com/leanprover-community/project-intentions/issues/55) | Community coordination | Open; topic labels applied; protected `intention` label pending | Maintainer label request posted | Wait for a maintainer to apply the protected label | 2026-11-19 |
| Lean Zulip announcement | Community announcement | Deleted; outreach closed | On 2026-11-12 the moderators deleted the AI-written announcement and imposed a one-day account suspension under the community rule prohibiting AI-written Zulip or GitHub comments | Do not repost it or use AI-written comments in Lean community channels; any future communication must be written personally by G. D. Partin and comply with the venue's rules | Closed |
| [Constraint Systems Institute](https://constraintsystemsinstitute.org/lean/?framework=beyond-aasc) | Archive/listing request | Accepted and listed under Beyond AASC | On 2026-11-12, the request for `v0.15.0-research` (DIRAC-001 through DIRAC-015, Zenodo DOI `10.5281/zenodo.24150001`) was sent to `amos@constraintsystemsinstitute.org`; Amos confirmed the public listing the same day. Catalogue inclusion carries no independent mathematical review or endorsement of LFM's physical claims. | None. Listing confirmed. | Closed |

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
