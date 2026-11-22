# Press release submission copy

## Headline

Lean 4 Formalization Derives Free Dirac Structure From a Two-Field Cubic Lattice Action

## Summary

Open-source V2 release removes the prepared-background assumption and formally derives two four-component free-Dirac sectors with explicit lattice and timestep corrections.

## Release

An independent open-source research project has published V2 of the Lean LFM-to-Dirac Derivation, a machine-checked Lean 4 development connecting a reciprocal two-field cubic-lattice action to the free linearized Dirac equation.

The formal chain starts from two native complex three-component fields on a weighted 19-point cubic neighborhood. It derives the action variation, an exact zero-energy uniform vacuum, the finite-lattice tangent operator, a cubical Clifford square root, a four-component phase-space readout, the rest-mass term, three transport generators, centered-time evolution, and the density/current bridge. Spatial and timestep corrections are stated explicitly rather than hidden inside a continuum assumption.

V2 also proves that the native eight-corner fiber decomposes exhaustively into exactly two disjoint rank-four sectors. Both sectors carry the same derived free-Dirac generator. The result is retained as a prediction of the formal lattice model rather than deleting one sector by assumption.

The principal improvement over V1 is removal of the maintained prepared-background assumption. The V2 theorem is instead based on the native reciprocal action and an exact uniform zero-energy vacuum. Lean's kernel checks the proof chain, and the release verifier reports only standard Lean foundations.

The scope is deliberately precise. The release establishes a free linearized mathematical result inside the declared lattice model. It does not yet prove that nature uses this substrate, derive a nonlinear localized electron, establish experimental agreement, or provide an origin story for the vacuum state.

The source code, reproducible release, formal assumptions, verification scripts, and theorem boundary are publicly available. The project welcomes review from mathematical physicists, lattice-field researchers, and Lean formalization contributors.

## Links

- Source repository: https://github.com/gpartin/LeanDiracDerivation
- V2 GitHub release: https://github.com/gpartin/LeanDiracDerivation/releases/tag/v2.0.0-research
- Independent V2 Zenodo record: https://doi.org/10.5281/zenodo.24432641
- Historical V1 record: https://doi.org/10.5281/zenodo.24150001

## Boilerplate

The Lean LFM-to-Dirac Derivation is an open research formalization studying which continuum operator structures can be derived from a local reciprocal cubic-lattice action. Claims are separated into machine-checked algebraic results, controlled continuum limits, and independently open physical-validation gates.

## Suggested classification

- Industry: Science and technology
- Subject: Mathematical physics / formal methods
- Location: Online
- Tags: Lean 4, Dirac equation, formal verification, lattice field theory, cubical Clifford algebra, C19 lattice
