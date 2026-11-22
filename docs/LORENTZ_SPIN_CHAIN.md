# V2 Lorentz and spin-one-half derivation

This local research chain begins at the proved V2 free-Dirac boundary.  It
does not change the native fields, C19 geometry, action, equations, centered
update, vacuum, four-component readout, or either cubical copy.

It also does not use `B = 1197` to manufacture the Lorentz or spin algebra.
Those results extend the same transverse free-Dirac sector, whose generator is
independent of the numerical `chi` kinetic normalization. `B` remains a
parameter of the full nonlinear live-`chi` dynamics.

All mathematical gates below compile locally.  The cumulative release seal is
intentionally deferred until review; no commit, push, or release-manifest
update has been made on this branch.

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
| LS-005 | Finite rotations | A unitary rotor derived from the bivector has `2*pi -> -I` and `4*pi -> I` | Do not replace it with an imposed angle-halving rule |
| LS-006 | Lorentz generators | Boosts derived from the existing alpha operators and rotations satisfy `[J,J]`, `[J,K]`, and `[K,K]` with the Minkowski signs | Record the exact generator pair and sign defect |
| LS-007 | Free-operator covariance | The mass grading is rotation invariant; rotations transform transport; boosts mix the oriented time/spatial gamma operators as a four-vector | Do not call generator algebra alone a covariance proof |
| LS-008 | Cumulative certificate and audit | One theorem packages LS-001--007; `lake build` and the release verifier pass with only standard Lean foundations | Keep the branch local and report the first unresolved obligation |

## Claim boundary

A pass identifies the V2 continuum sector as a massive spin-one-half
representation of the Lorentz Lie algebra and proves the result for both
geometry-derived copies.  The covariance statement is infinitesimal; it does
not claim exact finite-lattice Lorentz invariance.  It is not a nonlinear
localized-electron theorem, an electromagnetic coupling, a magnetic-moment
calculation, or a selection rule for one copy.

## Proposed next chain

After review of this result, the next independent research chain is native
phase transport to a local `U(1)` connection, Maxwell--Dirac coupling, and the
Pauli limit with the tree-level `g = 2` prediction.  No electromagnetic term
is introduced in the present branch.
