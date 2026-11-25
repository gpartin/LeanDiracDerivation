# V4 coherent localized-electron release plan

## Objective

V4 must prove that one freely evolving localized `Psi`--`chi` state realizes
the already-derived free Dirac behavior at rest and in motion.  Every passing
stage must use the same declared action, carrier preparation, phase-space
readout, C19 geometry, and native time update.  Evidence produced by different
actions may guide research but may not be combined into a release claim.

This plan concerns an already-running localized carrier.  It does not claim an
origin story, unique vacuum selection, Maxwell reconstruction, or selection of
one of the two cubical Clifford copies.

## Non-negotiable controls

1. Both native fields are complex three-component fields on the C19 lattice.
2. The action is real, local, reciprocal, and cubic symmetric.  Both native
   equations are obtained by variation of that one action.
3. No Dirac matrix, prepared corner label, projector, target trajectory,
   receiver preload, reset, or runtime holding force may enter the action or
   native evolution.
4. The independently evolved discrete Dirac packet is a comparator only.  It
   supplies no force or state correction.
5. Every artifact records the action hash, state hash, readout hash, integrator,
   timestep, box, and exact claim boundary.
6. A passing state must remain the same identifiable eight-site carrier.  A
   smaller residual for a different shape is not a pass.
7. A stage is committed and pushed only after its stated gate and red-team
   checks pass.  Failed attempts remain research evidence on the feature
   branch and never silently alter an earlier pass.

## Stage chain

| Stage | Required theorem or evidence | Pass criterion | Failure route |
|---|---|---|---|
| V4-001 | Exact localized rest solution | Existing Lean theorem rebuilds with standard foundations and no runtime support | Repair only the formal statement or proof; do not change the action |
| V4-002 | Nonlinear rest stability under the final action | Same carrier survives 100 internal cycles with the declared perturbation, containment, charge, energy, timestep, and box checks | RCA the first unstable native mode; repair the action or preparation, then reopen every action-dependent stage |
| V4-003 | Lossless localized four-component readout | One fixed phase-space map recovers four components from both committed layers, live `chi`, and required surroundings, with explicit complementary-sector residual | Enlarge the native readout only when the omitted native coordinate is independently identified; no fitted evolving frame |
| V4-004 | Operational spin one-half on the same carrier | Native rotation protocol gives the `2pi` sign reversal and `4pi` return while preserving identity and the V4-003 readout | RCA rotation versus readout failure; do not insert a spinor rule |
| V4-005 | Short-time native Dirac transport and current | Both copies, both directions of three axes, native density/current, and the fixed readout agree with the exact C19/centered-time Dirac comparator through the declared local interval | Decompose the first residual by field, site, component, action term, and copy |
| V4-006 | One complete cell crossing | The same unreset joint state crosses one cell and matches the independently evolved Dirac packet at the crossing time; spreading is allowed when Dirac predicts it | Solve the moving branch and crossing time; an exact rigid hop is a stronger optional control, not the only Dirac motion definition |
| V4-007 | Repeatable motion | The V4-006 state completes a second crossing without reload, projection, or reset in both signs | RCA receiver-to-sender conversion at the first failed reversal |
| V4-008 | Complete same-state Dirac map | Rest mass, time, all three transport generators, density, current, and phase evolution use the same embedding and include explicit lattice/time remainders | Reopen the first inconsistent mapping; do not average incompatible actions or embeddings |
| V4-009 | Robustness and convergence | All axes, a held-out diagonal, both copies, perturbations, `dt/dt/2/dt/4`, and box refinement pass without worse containment | Scope and repair the first failed control; preserve the incumbent |
| V4-010 | Cumulative Lean theorem and release | Lean derives the localized low-momentum Dirac limit from the declared action and same carrier; clean checkout, axiom audit, artifact audit, and full build pass | Release remains blocked at the earliest unmet dependency |

## Research loop and stop rule

For an open stage, each RCA uses exactly these fields:

1. **Issue**
2. **Why it matters**
3. **Symptoms**
4. **Expected symptoms**
5. **Root cause**
6. **Corrective action**

The corrective action must be falsifiable and must name the action,
preparation, readout, or numerical variable it changes.  After five
consecutive RCAs that produce no stage-closing improvement, work stops with a
scoped obstruction and the last known-good stage remains the release boundary.
Solver exhaustion alone is not a mathematical no-go.

## Promotion rule

V4 may merge to `main` only when V4-001 through V4-010 all pass on one coherent
dependency chain.  Until then all work remains on
`feature/v4-coherent-chain`; the existing public `main` is unchanged.
