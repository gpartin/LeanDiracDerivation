# Lean LFM-to-Dirac Derivation V2

[![Lean proof](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml/badge.svg)](https://github.com/gpartin/LeanDiracDerivation/actions/workflows/lean.yml)

This repository contains one production proof line: the V2 top-down Lean 4
derivation of the complete free linearized Dirac equation from the declared
two-field C19 lattice action.

The default import is unambiguous:

```lean
import LeanDirac

#check LeanDirac.V2.complete_first_principles_free_dirac
```

The checked-out tree does not contain the historical numbered V1 proof or its
downstream consequence stages. That development remains available in Git
history at tag `v1-legacy-final`.

## Starting point

At each lattice site the native fields are `Psi_i, chi_i : C^3`. The declared
geometry is C19: six face bonds of weight 2 and twelve edge bonds of weight 1.
The constants are `chi0 = 19`, `kappa = 1/63`, and `B = 1197`.

The proof starts from one real reciprocal action `L = T - U`, derives both
Euler--Lagrange equations, and derives the centered two-layer update. The full
action is stated in [docs/FRAMEWORK.md](docs/FRAMEWORK.md).

## Result

The cumulative theorem is
`LeanDirac.V2.complete_first_principles_free_dirac`. Its dependency chain
derives:

- reciprocal native field variation from the action;
- an exact native zero-energy vacuum and the full finite-lattice tangent;
- the closed four-complex-component phase space;
- the cubical Clifford root and both rank-four copies;
- the standard alpha/beta form, nonnegative mass, and three transport terms;
- centered/Verlet time evolution with an explicit vanishing time defect;
- the position-space free Dirac equation with the explicit C19 remainder;
- the positive-frequency density/current bridge and its controlled remainder.

There is no external, prepared, period-two, maintained, or Dirac-shaped
background. The expansion point is one exact zero-energy vacuum solution
proved from the native action. Selecting a representative of the degenerate
vacuum family is not claimed to be a cosmological origin mechanism.

The theorem is a free linear tangent result. It does not claim a nonlinear
localized electron or any later physical consequence.

## Reproduce

Install Lean through `elan`, then run:

```bash
lake build LeanDirac LeanDiracAudit
python scripts/verify_release.py
```

The verifier compiles the complete dependency closure, checks the sealed
sources, rejects `sorry` and custom axioms, and admits only `propext`,
`Classical.choice`, and `Quot.sound`.

See [the proof chain](docs/PROOF_CHAIN.md), [claims](docs/CLAIMS.md), and
[reproducibility instructions](docs/REPRODUCIBILITY.md).
