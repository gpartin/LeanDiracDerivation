# Lean LFM-to-Dirac Derivation

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository is the production Lean 4 environment for a noncanonical
attempt to derive the free Dirac equation from a local two-field cubic-lattice
substrate.

The highest admitted gate is **DIRAC-009 of 15**. The current theorem connects
one declared positive complex-Psi/complex-chi action to three native
equal-square, mutually anticommuting cube-incidence directions while retaining
both rank-four copies. It does **not** yet prove the free Dirac continuum
equation, physical mass, spin one-half, or an electron.

## Current status

| Range | Status | Result |
|---|---|---|
| DIRAC-001--003 | PASS | Native framework, one positive action, and reciprocal action variations. |
| DIRAC-004--005 | PASS | Reversible centered evolution and the conditional prepared background. |
| DIRAC-006--008 | PASS | Nondegenerate native transport, lossless readout, and exact first-order map. |
| DIRAC-009 | PASS | Action-derived cubical incidence algebra with two retained rank-four copies. |
| DIRAC-010--015 | OPEN | Standard Dirac identification, frequency readouts, continuum theorem, current, final audit, and clean publication. |

See [the complete proof chain](docs/PROOF_CHAIN.md) for every gate and its
claim boundary.

## Reproduce the formal result

Install Lean with `elan`, then run:

```bash
lake update
python scripts/verify_release.py
```

The verifier builds every production module, audits the compiled Lean
environment, checks all role-specific manifests, runs adversarial audit tests,
and verifies the derivation contract. It is read-only: it does not install Git
hooks or alter the working tree or index.

## Repository guide

- [Native framework](docs/FRAMEWORK.md): exact fields, constants, geometry,
  action, equations, and update.
- [Formal proof chain](docs/PROOF_CHAIN.md): all 15 gates and theorem boundary.
- [Claims and non-claims](docs/CLAIMS.md): what DIRAC-009 does and does not show.
- [Reproducibility](docs/REPRODUCIBILITY.md): formal and numerical rebuilds.
- [Changelog](CHANGELOG.md): current cleanup and historical milestones.
- `LeanDirac/Publication.lean`: sole cumulative publication theorem.
- `LeanDiracAudit.lean`: explicit theorem axiom reports.
- `project/`: framework, proof, evidence, release, status, and review manifests.
- `experiments/dirac009_native_coupling/`: supplementary numerical evidence.

## Proof-security boundary

The production audit imports every Lean source in the repository and inspects
the compiled environment. The publication theorem may use only Lean's standard
foundations: `propext`, `Classical.choice`, and `Quot.sound`. A custom axiom,
`sorry`, missing module, changed locked file, incomplete/non-finite evidence,
or stale claim status fails verification.

The native action and formal proof are locked separately from numerical
evidence and publication files. Consequently, a documentation or audit change
cannot be confused with a change to the physical action.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json). This repository is released under
[CC0-1.0](LICENSE).
