# Publication-quality audit through DIRAC-019

## Scope and method

This audit reviews the exact production dependency chain rooted at
`LeanDirac.current_highest_gate_is_dirac019`, its independent evidence,
machine-readable status, release automation, archival metadata, and public
claim language. It distinguishes three questions that must not be collapsed:

1. Does the Lean theorem compile from the declared two-field action using only
   the allowed standard foundations?
2. Do independent programs reconstruct the finite-dimensional identities that
   the Lean theorem advertises?
3. Do the release files describe exactly those results without converting a
   conditional representation into a physical-electron claim?

The native action and governing equations are unchanged by this repair.

## Findings and resolutions

| Severity | Finding | Risk | Resolution |
|---|---|---|---|
| High | DIRAC-017 documentation described full rotation covariance, while Lean directly proved only the quarter-turn identity. | The public claim was broader than the formal certificate. | Added `cube_continuous_turn_hamiltonian_covariance`, covering every real angle, and made the certificate retain discrete and continuous covariance as separate fields. |
| High | The release workflow still referred to DIRAC-015 artifacts after the chain advanced to DIRAC-017. | A successful workflow could publish no current archive. | Corrected the name and paths to DIRAC-017, made missing artifacts fatal, and added a 90-day artifact retention policy. |
| High | `project/status.json` flattened DIRAC-017 to a generic `PASS`. | Machine consumers could erase the theorem's conditional boundary. | Replaced it with `PASS_CONDITIONAL_CUBICAL_SPIN_HALF_REPRESENTATION`; all audits now require that exact value. |
| High | Independent admitted evidence stopped at DIRAC-013. | Copy-exchange and spin claims had no separately implemented numerical reconstruction. | Added deterministic NumPy reconstructions for DIRAC-016 and DIRAC-017, including continuous Hamiltonian covariance, and fail-closed schema validators plus adversarial mutations. |
| High | Current packaging lacked an embedded commit-to-archive provenance record. | A detached archive could not identify the exact source theorem it contained. | Added `PROVENANCE.json` with the Git commit, toolchain, action ID, highest gate/status, archive name, and publication theorem. The package still contains a full SHA-256 manifest and external checksum. |
| Medium | DIRAC-015's historical package status was being reused as if it described the current release. | Old and current release guarantees were conflated. | Marked the DIRAC-015 record historical and added a separate DIRAC-017 release-readiness contract with current artifact names. |
| Medium | The proof root imported the entire Mathlib umbrella and nine modules raised global heartbeat and recursion limits. | Build cost and fragility were unnecessarily high and concealed which proofs were expensive. | Replaced the umbrella import with targeted modules, removed all recursion-depth overrides, and retained only the smaller heartbeat budgets demonstrated necessary by the build. The expensive cube-face budget is theorem-local. |
| Medium | GitHub actions used mutable major-version tags. | Re-running an old commit could execute different third-party action code. | Pinned checkout, Lean setup, Python setup, and artifact upload actions to immutable commit SHAs. |
| Medium | Reproduction instructions ran `lake update`. | Reproduction could silently change the dependency graph. | Reproduction now uses the committed `lake-manifest.json`; dependency upgrades are a separate reviewed task. |
| Medium | Citation metadata still identified the archived 0.15 release while describing DIRAC-017. | Citation version and described source did not match. | Marked the current tree `0.17.0-research-dev`, removed the false release date, and explicitly distinguished it from the archived v0.15 record. |
| Medium | The publication dependencies included unused PyTorch. | Installation was much larger and exposed irrelevant supply-chain surface. | Removed PyTorch; the independent suite uses pinned NumPy only. |
| Low | `docs/FRAMEWORK.md` used literal bracket lines rather than Markdown math delimiters. | Core equations rendered poorly on common hosts. | Converted every display to `\[` and `\]`. |
| Low | Submission tracking described the GitHub tree as a 15-gate release even though `main` had advanced. | Readers could confuse current source with the immutable archive. | Separated the current 17-gate development chain from the immutable 15-gate Zenodo release. External-review statuses were otherwise preserved verbatim. |
| High | The first DIRAC-018 draft required each copy candidate to square correctly on the full eight-corner space. | That overconstrained the problem and silently imposed a relationship between the two copies instead of proving it. | Replaced the global condition with independent sector-local conditions on the exact plus/minus embeddings. A generic probe theorem fixes each copy's coefficients separately; no copy symmetry is assumed. |
| High | A uniqueness headline could be misread as classifying every possible lattice square root. | Nonlocal or flavor-mixing roots and exact finite-spacing factorizations were outside the proof. | Named the admitted class in the theorem, status, README, claims, related-work discussion, and release guards. Added explicit exclusions for arbitrary roots, exact finite-spacing Dirac, universal doubling, and electron claims. |
| High | Cubic covariance and positive orientation could have functioned as untested answer-selecting assumptions. | The theorem would show sufficiency without demonstrating why the hypotheses eliminate actual alternatives. | Added two formal controls: an overall negative root still squares and is covariant but fails orientation; a one-axis-reflected root still squares but fails cubic covariance. Independent reconstruction repeats both controls. |
| Medium | The existing two-copy declaration did not by itself prove that the two embeddings exhaust the native eight-corner space. | A hidden third sector could invalidate the copy census. | Proved both embedding ranges equal the corresponding flavor eigenspaces, their intersection is zero, and their projectors decompose every amplitude. Independent matrices confirm rank four plus rank four. |
| Medium | The continuum classification did not say whether the exact scalar C19 symbol had extra Brillouin-zone zeros. | Readers could conflate continuum root classification with the exact lattice zero structure. | Proved analytically that the exact scalar C19 transport symbol has only the origin zero in the first Brillouin zone; periodic grids 2 through 32 independently reproduce one zero. The documentation explicitly does not convert this into an exact first-order factorization or universal no-doubling claim. |
| High | The initial movement extension conserved one update only. | A one-step identity could not support sustained propagation. | Proved the recurrence-derived phase advances every positive-frequency eigenmode and preserves both density and density-weighted current for every integer update count; lifted this to arbitrary finite spectral families. |
| High | A finite list of modes could be mislabeled as a localized electron. | The proof does not yet include a spatial Fourier basis, Parseval theorem, centroid, nonlinear live chi or localization. | Renamed and scoped the claim as finite spectral transport. Added a separately labeled Fourier diagnostic and left the coherent real-space Lean theorem and nonlinear electron movement open. |
| High | Movement might have been implemented by a manual shift. | That would be a runtime bolt-on unrelated to the native equations. | The proof uses only the existing recurrence phase, Hamiltonian momentum response and derived current. No coordinate translation, action term, force, track or projector was added. |
| Medium | Stability and frequency-branch assumptions could be decorative. | Conservation fails outside the stable recurrence interval, and the negative branch is not the stated sector. | Made both assumptions explicit in Lean and added independent unstable-timestep and negative-frequency rejection controls. |
| High | Quantification over positive modes could pass vacuously if none were constructed. | A formally empty sector would say nothing about movement. | Constructed an explicit nonzero positive-frequency eigenmode at every continuum momentum from the previously proved Hamiltonian-square identity. |

## Formal claim after repair

The strongest theorem remains conditional. Given the declared noncanonical
positive two-field C19 action and the prepared invariant sector, Lean proves a
controlled free-Dirac continuum factorization, density/current limit, scoped
copy-translation algebra, and a cubical spin-one-half representation. For the
spin representation it now proves both the discrete quarter-turn covariance
and all-angle continuous covariance of the interpreted mass-plus-transport
Hamiltonian.

The theorem does **not** prove that unrestricted native dynamics uniquely
chooses the cubical square-root orientation, produces or stabilizes a localized
electron, applies the derived rotations to such a carrier, or yields
electromagnetic coupling and a magnetic `g` factor. Those are scientific
research gates, not release-engineering defects, and this audit does not hide
them behind a generic pass label.

DIRAC-018 now proves that the cubical root is forced inside the explicitly
declared real, local, first-order, flavor-preserving incidence-coefficient
class. This closes the earlier orientation gap only at that scope. It does not
classify arbitrary nonlocal or flavor-mixing operators, derive an exact
finite-spacing square root, or settle general lattice-fermion doubling.

DIRAC-019 adds sustained positive-frequency spectral transport. It proves
all-update density conservation, the finite-family current/group-velocity
identity, strict massive subluminality and recurrence-generator convergence.
It does not yet prove a coherent localized real-space packet in Lean, a
uniform finite-time continuum estimate, or nonlinear live-chi electron motion.

## Acceptance tests

The repair is accepted only if all of the following pass from the repaired
tree:

- full `lake build`;
- compiled environment enumeration of every production Lean module;
- explicit axiom reports for the cumulative theorem and the new continuous
  covariance theorem, with no foundation outside `propext`,
  `Classical.choice`, and `Quot.sound`;
- independent DIRAC-010, 011, 012, 013, 016, 017, 018, and 019 reconstructions;
- adversarial mutations of evidence status, completeness, finiteness, and
  tolerances;
- derivation-contract, final-chain, and publication-quality audits;
- synchronized framework, proof, evidence, and release hashes;
- two publication-package builds from the same clean commit with identical
  SHA-256 checksums and verified embedded provenance;
- a final clean Git status.

The exact executed results are reported in the release handoff and are not
inferred merely from this document.
