# Lean LFM-to-Dirac Derivation

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository is the production Lean 4 environment for a noncanonical
attempt to derive the free Dirac equation from a local two-field cubic-lattice
substrate.

All **15 of 15** derivation gates pass. The chain begins with
one positive complex-Psi/complex-chi C19 action, derives a cubical
low-momentum square root with an explicit fourth-order lattice remainder, and
proves that both native rank-four copies are equivalent to the conventional
Dirac alpha/beta representation, organizes both frequency signs, and uniquely
selects the nonnegative rest coefficient from the native running frequency,
and proves the prepared-vacuum low-momentum continuum equation with explicit
C19 spatial and centered-time errors. It then derives the native bond flow,
exact local phase continuity, and the positive-frequency continuum Dirac
density/current limit from the same action and embedding. The final theorem
makes the Klein--Gordon/Dirac relationship explicit: the action-derived
massive C19 tangent has a Klein--Gordon continuum symbol, and the derived
four-component Dirac Hamiltonian squares to that symbol. This is a conditional
free-field continuum result, not a localized physical electron proof.

## Current status

| Range | Status | Result |
|---|---|---|
| DIRAC-001--003 | PASS | Native framework, one positive action, and reciprocal action variations. |
| DIRAC-004--005 | PASS | Reversible centered evolution and the conditional prepared background. |
| DIRAC-006--008 | PASS | Native C19 transport, lossless readout, and exact first-order map. |
| DIRAC-009 | PASS | Cubical Clifford square root of the C19 continuum symbol, with an explicit lattice remainder and both rank-four copies retained. |
| DIRAC-010 | PASS | Exact change of basis from both native copies to the standard Dirac alpha/beta representation. |
| DIRAC-011 | PASS | Common positive/negative-frequency graphs and native nonnegative rest-mass selection. |
| DIRAC-012 | PASS | Action-derived linearized scalar channel and controlled low-momentum/finite-timestep Dirac evolution. |
| DIRAC-013 | PASS | Action-derived native bond flow, exact local continuity, and positive-frequency continuum density/current. |
| DIRAC-014 | PASS | Final adversarial audit reports zero critical/high findings. |
| DIRAC-015 | PASS | One cumulative action-to-Dirac theorem and deterministic clean-checkout package. |

See [the complete proof chain](docs/PROOF_CHAIN.md) for every gate and its
claim boundary.

## Reproduce the formal result

Install Lean with `elan`, then run:

```bash
lake update
python scripts/verify_release.py
```

The verifier builds every production module, audits the compiled Lean
environment, checks role-specific manifests, runs adversarial audit controls,
and independently reconstructs the DIRAC-010--013 identities, audits the final
theorem, and verifies deterministic packaging. It is read-only: it
does not install Git hooks or alter the index.

## Repository guide

- [Native framework](docs/FRAMEWORK.md): fields, constants, geometry, action,
  equations, and update.
- [Formal proof chain](docs/PROOF_CHAIN.md): all 15 gates and theorem boundary.
- [Claims and non-claims](docs/CLAIMS.md): what DIRAC-015 establishes.
- [Reproducibility](docs/REPRODUCIBILITY.md): rebuild instructions.
- [Palomar literature review](docs/PALOMAR_LITERATURE_REVIEW.md): prior-work
  search, closest established lines, and explicit priority limitations.
- [Changelog](CHANGELOG.md): current cleanup and historical milestones.
- `LeanDirac/Publication.lean`: sole cumulative publication theorem.
- `LeanDiracAudit.lean`: explicit theorem axiom reports.
- `project/`: framework, proof, evidence, release, status, and review manifests.
- `experiments/dirac010_standard_map/`: independent matrix reconstruction.
- `experiments/dirac011_frequency_mass/`: independent frequency/remainder and
  rest-mass reconstruction.
- `experiments/dirac012_continuum_evolution/`: independent action-expansion,
  centered-recurrence, and continuum-error reconstruction.
- `experiments/dirac013_density_current/`: independent native flow,
  continuity, positive-frequency density, and continuum-current reconstruction.

## Proof-security boundary

The production audit imports every Lean source and inspects the compiled
environment. The publication theorem may use only Lean's standard foundations:
`propext`, `Classical.choice`, and `Quot.sound`. A custom axiom,
`sorry`, missing module, changed locked file, incomplete/non-finite evidence,
or stale claim status fails verification.

The native action is locked separately from the downstream continuum readout.
The cubical alpha/beta operators are not terms in the action.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json). This repository is released under
[CC0-1.0](LICENSE).
