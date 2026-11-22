# V3 Lorentz and spin-one-half derivation

This local research chain begins at the proved V2 free-Dirac boundary.  It
does not change the native fields, C19 geometry, action, equations, centered
update, vacuum, four-component readout, or either cubical copy.

It also does not use `B = 1197` to manufacture the Lorentz or spin algebra.
Those results extend the same transverse free-Dirac sector, whose generator is
independent of the numerical `chi` kinetic normalization. `B` remains a
parameter of the full nonlinear live-`chi` dynamics.

All mathematical gates below compile locally. The V3 release seal independently
checks the cumulative theorem and its complete imported environment.

## Target theorem

Starting from the gamma operators already derived from the native C19 action,
derive the rotation and boost representation carried by the same two
four-component sectors.  The final certificate must establish the Lorentz Lie
algebra, spin Casimir `3/4`, spin projections `+/- 1/2`, the `2*pi` sign
reversal and `4*pi` return, and preservation of both cubical copies.

## Derivation gates

| Gate | Required derivation | Pass condition | Failure action |
|---|---|---|---|
| LS-001 | Native oriented spatial-plane bivectors | Built only from `cubicalGamma`; each squares to `-1` | Report the first failed cube-incidence identity |
| LS-002 | Rotation normalization | The nonzero scale satisfying the native three-axis commutator is uniquely `1/2` | Reject any manually selected half-angle |
| LS-003 | Spin representation | Self-contained generators obey the three rotation commutators, each squares to `1/4`, and their Casimir is `3/4` | Stop if any relation needs an assumed Pauli/Dirac spin matrix |
| LS-004 | Two-copy transport | Both V2 rank-four embeddings intertwine the native spin operators and remain invariant | Record which copy/component fails |
| LS-005 | Finite rotations | The derived Hermitian generator exponentiates to a density-preserving angle-group representation, differentiates back to the generator, has `2*pi -> -I`, and `4*pi -> I` | Do not replace it with an imposed angle-halving rule |
| LS-006 | Lorentz generators | Boosts derived from the existing alpha operators and rotations satisfy `[J,J]`, `[J,K]`, and `[K,K]` with the Minkowski signs | Record the exact generator pair and sign defect |
| LS-007 | Free-operator covariance | The mass grading is invariant and the complete continuum Dirac symbol transforms consistently when its spinor and four-momentum arguments are varied together | Do not call gamma algebra alone a covariance proof |
| LS-008 | Pauli--Lubanski invariant | Construct `W` from arbitrary four-momentum and the derived rotations/boosts, prove `P.W = 0`, then specialize its square in the rest frame to `-3 m^2 / 4` | Reject a rest Casimir defined directly in the desired answer form |
| LS-009 | Both-copy audit | Native rotations and boosts intertwine with each of the two exhaustive rank-four embeddings | Do not silently select or delete one copy |
| LS-010 | Cumulative certificate and audit | One theorem packages LS-001--009; `lake build` and the release verifier pass with only standard Lean foundations | Keep the branch local and report the first unresolved obligation |

## Claim boundary

A pass identifies the V2 continuum sector as a massive spin-one-half
representation of the Lorentz Lie algebra and proves the result for both
geometry-derived copies. The full continuum Dirac symbol is infinitesimally
covariant and the Pauli--Lubanski rest invariant follows from a general
four-momentum construction. The result does not claim exact finite-lattice
Lorentz invariance. It is not a nonlinear
localized-electron theorem, an electromagnetic coupling, a magnetic-moment
calculation, or a selection rule for one copy.

## Proposed next chain

After review of this result, the next independent research chain is native
phase transport to a local `U(1)` connection, Maxwell--Dirac coupling, and the
Pauli limit with the tree-level `g = 2` prediction.  No electromagnetic term
is introduced in the present branch.
