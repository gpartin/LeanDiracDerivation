# V4-006 movement-repair candidate matrix

## Scope

This is a matched audit of the proposed repairs for the localized carrier's
movement gate.  The comparator is the centered-time, exact-C19 discrete Dirac
packet on its negative-frequency branch.  The Dirac packet supplies no force,
projection, reset, or fitted frame to the native evolution.  The canonical
V40 action is unchanged.  Every additional interaction below remains an
unadmitted research action.

No candidate reaches the V4-006 acceptance gate.  Tests that fail an earlier
gate are not promoted to the expensive 100-cycle or two-cell nonlinear gate.

## Results

| Candidate | Rest/action gate | Movement result | Verdict |
|---|---|---|---|
| Canonical phase-space/current audit | Exact discrete continuity passes: maximum global common-charge drift `1.5163e-10`; maximum local defect `2.2115e-10` | Longitudinal current has the correct sign, but transverse center drift remains | The remaining mismatch is not caused by omitting canonical momentum from the readout |
| Phase-shifted relative-periodic orbit | Translation time and both field phases were solved, not prescribed | Boundary residual `0.04760`; unreset second-cell residual `2.4707`; carrier fidelity `0.567`; containment `0.624` | No freely translating same-carrier orbit was found under the current action |
| Self-generated wake/tail | Reciprocal wake actions and state/frequency continuation were tested | Best field residual `5.75e-4` (state-only) and `8.02e-4` (state/frequency), neither converged | A precursor exists as a response direction, but no autonomous traveling wake was solved |
| Complete square-face factorization loop | Positive, cubic and local; rest energy `6.06e-28`, first-variation norm `5.94e-13`; only rank two at fixed charge | Best all-eight linear one-cycle shape error `1.3235` at scale 2 versus `1.3598` without the loop, but leakage rises from `0.01999` to `0.02745`. Four-update nonlinear error worsens from `0.002349` to `0.023015` and leakage from `0.0001057` to `0.0006285`. One case stopped at `1.022e-10` against a `1e-10` solver tolerance and passes when rechecked at `2e-10`; this marginal solver threshold does not affect the physical rejection | Best broad *linear* clue, but not a winner; it changes accumulated timing without reproducing the required short native response |
| Local/bond symplectic overlap connection | Real reciprocal velocity-linear actions, rest-screened | Neither action supplies rank-three fixed-charge transport on the admitted rest families | Tested overlap connection fails |
| Ablowitz-Ladik-like density-modulated exchange | 32 real reciprocal C19 position invariants, including coherent hopping; rest-null subspace dimension 18 | Best admissible response residual `0.24966` | Tested position-only action space cannot generate the missing response |
| Special sliding velocity | No action change; the Dirac transport speed was swept | One orientation improves to `0.4684` shape error near scale 8, but all-eight error is `1.9692`, leakage `0.02253`, norm drift `1.3686` | Direction-specific coincidence, not a cubic moving branch |
| Saturating/local clock repair | Onsite response included in the exact inverse-Hessian basis | Required unconstrained local-potential coefficient is `1.78e-15`; the positive-square basis leaves residual `0.95472`. Exact cancellation uses negative coefficients, including a negative sextic (`-26.14`) | A local clock change cannot by itself transport orientation between cubes; the exact algebraic cancellation is not an admissible bounded action |
| Two-copy relay | Both four-component copies retained; exact Hermitian exchange involution tested | Best one-cycle phase-aligned error `0.5741`; longer mixed-copy trials grow rather than translate | The second copy is a real algebraic channel but the tested exchange is not the native handoff mechanism |
| Peierls--Nabarro/pinning flattening | Full interpolation energy measured | Positive endpoint barrier is numerical zero (`1.46e-10`); the path instead contains a deep intermediate trough | A conventional positive pinning barrier is not the leading obstruction, so flattening it does not repair the clock/orientation handoff |

## Incumbent and conclusion

The strongest current incumbent remains the face-only native oscillator at
the geometry-derived `6/19` scale.  It passes all eight four-update nonlinear
cases with worst raw Dirac error `0.0023493` and leakage `0.00010572`, and its
Noether continuity identity passes.  It does not survive a complete internal
cycle: worst all-eight shape error is `1.35980`, cross-block leakage `0.01999`,
and norm drift `0.81863`.

Therefore there is **no winner**.  The experiments do narrow the missing
mechanism: it is not charge accounting, a simple speed choice, a conventional
pinning barrier, a local clock saturation, or a simple exchange of the two
copies.  A complete-face loop contains useful orientation information, but
its position-only curvature acts at the wrong time.  The next admissible
action search should combine that rest-null face-loop geometry with a
reciprocal velocity/symplectic response, then derive its coefficient from a
native degeneracy or normalization condition rather than fitting the Dirac
trajectory.
