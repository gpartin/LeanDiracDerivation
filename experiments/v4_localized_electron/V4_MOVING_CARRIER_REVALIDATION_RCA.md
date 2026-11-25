# V4 recovered moving-carrier revalidation — 2026-11-25

## Issue

V4-006 is not closed. The fixed companion-frame lift in the previous local
V4-003/005 approvals was not sufficient to describe literal translation of
the whole localized two-field state. The previous resting orbit and
conditional mathematical results are not erased; the cumulative mobile
carrier admission is reopened. See `v4_coherent_stage_tracker.json`.

## Why it matters

Transport of Ψ alone or a fitted spinor readout does not translate the
localized electron if the χ carrier/vacuum interface is left behind. The
same physical two-layer state must give the C19 Dirac rates and current
without external preparation of each receiver or a runtime projection.
Agreement of a small one-step state error as `dt → 0` is not sufficient:
the rate or acceleration defect must be controlled as well.

## Symptoms

All new incumbent experiments use `live_clifford_bond_action.py` with
SHA-256 `8868f69e48f8759ef58ca34a871069acfd61947b67143a36fd7366205919c4d3`,
complex Ψ and complex χ, periodic side-five C19 geometry, face/edge
weights 2/1, B=1197, joint face Laplacian plus frame oscillator,
spacetime scale 6/361, parallel matched and potential weights 1, and all
optional additional action scales zero. The update is the native discrete
Euler–Lagrange (DEL) recurrence, not ordinary Verlet applied to a
velocity-dependent action. The canonical action is unchanged. The input
cache records the previous/current/following DEL Jacobian blocks, not a
Dirac evolution operator. The physical rest state is embedded in
`v4_moving_rest_seed.npz` (raw little-endian array hash
`5df770deed55eda30942d9a5db50e19d0a5087b698c865f5600a37193c693b8e`).
Its original archive hash, recorded by the experiments, is
`142af61ef28a8a727da343742792bedc1c83541f1dcea825c2bcbe415024d80d`.

| Recovery or test | Result | Evidence |
|---|---|---|
| Historical exact overlap contrast, both copies | Companion-only lift misses 78.6–79.2% of the three centered full-field translation differences. The recovered contrast spans each to better than 2.4e-15. The two full all-center images no longer coincide (relative outside fraction 0.4672). This is a span result, not a dynamical closure theorem. | `v4_003_moving_native_graph_dt.json`, `_dt_half.json` |
| Native two-layer graph, no Dirac velocity imposed | With full contrast, raw one-update graph leakage is 9.27e-6 at dt=1/256 and 2.01e-6 at dt=1/512. But acceleration-normalized cross-block remains 0.859 and 0.745 and native centered-rate comparison remains large (31.07 and 26.50). Spectral half selection is a diagnostic hypothesis, not a proved native sector. | same |
| Eliminate every live χ coordinate from its own DEL row at the nonzero C19 probe frequency | χ row can be solved to numerical accuracy; the remaining full acceleration deficit is still about 0.975 (worst case, normalized by prescribed Ψ norm) at both steps. The zero-frequency solve is rank deficient and poorly conditioned. The test supplies an inverse requirement, not a force. | `v4_003_native_chi_response_dt.json`, `_dt_half.json` |
| 100-cycle tangent replay using the full contrast graph | Worst velocity-aware amplification 33.11 and 33.09 (two timesteps), late secular response approximately 99.5% χ, 82% exterior χ. The tested graph is not a demonstrated physically correct launch. | `v4_native_graph_revalidation_dt.json`, `_dt_half.json` |
| Remove incompatible approximate-stationary Noether-like linear invariants in the previous layer once | Incompatible initial charge reduced to numerical zero; worst velocity-aware amplification falls to 22.61, not the earlier gate's bound of 2. No recurrent projection used. Approximate-nullspace diagnostic, not a proof of exact nonlinear charge reduction. | `v4_native_graph_revalidation_charge_dt.json` |
| Recover the prior vacuum orientation action | The existing rest-null χ dyad C19 bond potential and its full complex-field alternative were differentiated as positive squares. Both exact rest gradients vanish. At strength 19 the complex-field alternative cuts the local acceleration-normalized leakage from 0.859 to 0.530 but the 10-cycle amplification only falls from 3.47 to 3.27. Stronger terms do not resolve the response. No candidate admitted and no coefficient fitted to Dirac. | `v4_vacuum_orientation_response.json` |
| Is secular growth actually translation? | Its overlap with the three full-field translation directions is under 0.064 in every tested case. The centered differences have nonzero stationary operator defects of 0.238–0.497. This does not forbid discrete cell-to-cell motion. | `v4_translation_marginal_mode.json` |
| Joint collective constraints instead of fixed microscopic amplitudes | Free all 1244 coordinates complementary to the 256-dimensional joint readout. At one nonzero C19 momentum magnitude (all signed axes, both copies), the minimum acceleration defect is 0.07123 and 0.07120 under timestep halving, but the correction requires up to 152 times the retained coordinate norm. With Ψ-only collective constraints the defect is 0.598 and the correction exceeds 2600 times the retained norm. The ranks and residuals agree across the three reported SVD tolerances. This is a fixed-frequency, finite-box least-squares result, not a theorem about finite hops. | `v4_collective_native_response_dt.json`, `_dt_half.json` |
| One complete crossing on the previous linearized embedding | The reference moves -1.0001 cells along x; the readout of the native tangent state moves +0.014 along x, +0.019 at half step. Not a nonlinear trajectory. | `v4_006_one_cell_crossing.json`, `_dt_half.json` |
| Actual nonlinear relative-periodic boundary condition | Five independent temporal nodes, exact translated endpoint, fixed exact initial rest snapshot, freely varied intermediate nodes and clock phase. At fixed period 17564 updates, native DEL residual decreases from 0.80463 to 0.40878 after 15 LBFGS iterations. It is not a root; no claim of a freely evolved solution, a solved period, or a universal no-go. | `v4_006_current_action_relative_orbit.json` |

Prior results were recovered from `localized_overlapping_cube_embedding.py`,
`localized_carrier_contrast_translation.py`, `localized_carrier_contrast_two_layer.py`,
`personal_touch_vacuum_activated_chi_orientation.py`, the earlier timing
and direct-prong reports, and the previous V4 ledger. In particular, the
old exact full-contrast spatial calculation was not the same thing as a
complete free-moving orbit. The earlier clock correction and carrier span
improvements have therefore been retained, not treated as a solution to
V4-006. The old companion-launch stability evidence recorded a worst ratio
of 1.23387, which remains a valid scoped result.

## Expected symptoms

The admitted native launch must contain the actual full Ψ–χ carrier
contrast and independent velocity response, not merely a normalized
companion direction. Its rate-level residual must converge; its same-state
spin and native Noether transport must follow. For V4-006 the independent
comparator and native state must complete the same integer-cell crossing,
with the next layer recoverable and no runtime forcing, resetting or
projecting. A continuous infinitesimal translation zero mode is not a
requirement for an exact discrete cell handoff.

## Root cause

The first confirmed integration mistake is the old readout's omission of
the physical χ-minus-vacuum contrast. The direct replay reveals a second,
separate defect: imposing that full contrast as a common spinor envelope
does not determine a closed native two-layer graph. The omitted χ normal
and surrounding modes carry most of the remaining response. Allowing χ
to solve its own frequency-domain equation removes its direct DEL defect
but leaves a large Ψ defect, demonstrating that χ orientation stiffness
alone is not enough. Freeing all complementary Ψ and χ in the collective
frequency response reduces the deficit substantially, but only with a
very large response outside the retained carrier and a nonvanishing
acceleration residual under timestep refinement. We do not yet know whether the incumbent action has
some different valid mobile preparation. The numerical tests do not
establish a theorem against finite-cell hopping.

For the local drift check, with the native recurrence
`x[n+1]=A x[n]+B x[n-1]`, a linear drift `x[n]=r+n dt v` obeys
`K v=0` and `K r=-dt (I+B)v`, where `K=I-A-B`. The tested centered
translation directions do not satisfy the first equation. The literature
supports using marginal Floquet response as a *diagnostic*, not demanding
it of every discrete hop: Aubry and Cretegny, [Mobility and Reactivity of
Discrete Breathers](https://arxiv.org/abs/cond-mat/9712046). For eliminating
symmetry drifts, Ortega and Ratiu, [Stability of Poisson Equilibria and
Hamiltonian Relative Equilibria by Energy Methods](https://arxiv.org/abs/math/0201239)
distinguishes momentum-constrained stability from the unreduced phase space.
Neither source proves mobility for this V4 action.

## Corrective action

1. Keep the frozen incumbent as a control. Construct the moving embedding
   using the full two-field response rather than attaching the resting χ
   profile to every Dirac component. Solve the native joint response while
   holding only a justified collective readout fixed; do not prescribe each
   microscopic Ψ or χ coordinate independently. The first collective
   frequency-response test already frees all complementary coordinates;
   it remains far too ill-scaled for admission. Resolve its near-resonant
   response into independent native action terms and derive a reciprocal
   coupling which transfers the needed orientation without the large
   accompanying complementary response. The Schur residual is the
   force/response criterion, not an inserted source.
2. If this cannot produce a native graph with the required rates, derive
   an admissible action change from the missing joint response. A vacuum
   stiffness increase alone has not closed the defect. In any repaired
   action, reproduce V4-001's rest solution, then revalidate the existing
   V4-002 criterion on the actual moving preparation and the same carrier
   V4-003/004/005 dependencies. Do not recreate the compiled theorems unless
   their hypotheses actually change; retain the old tests as regression
   controls.
3. Solve the full native relative-periodic boundary problem, including
   translation period, both initial layers and valid endpoint phase, and
   use independent holdout intervals and unreset free release. The sampled
   fixed-period optimization in this bundle is only a screen. Then test a
   second handoff both signs and compare mass, rate, density and current.

The prior five-stage approvals cannot be reused as a cumulative moving
electron pass. No new stage closed in this cycle; the failed repairs are
preserved. Full Lean build: 3890 jobs successful, no Lean theorem removed
or altered. No release to main is authorized by the evidence. The exact
rest theorem and conditional cubical spin/current theorems remain intact.

### Reproduction

Run from the `experiments/v4_localized_electron` directory with NumPy,
SciPy and PyTorch installed. Use one or two BLAS threads to avoid
oversubscription. The portable state seed is tracked; the large DEL
cache arrays are local derived data, not checked-in governing equations.
Rebuild them or check an existing cache using the same script:

```text
python v4_rebuild_moving_tangent.py --dt 0.00390625 --rest-cache v4_moving_rest_seed.npz --output v4_002_exact_1197/tangent_blocks.npz
python v4_rebuild_moving_tangent.py --dt 0.001953125 --rest-cache v4_moving_rest_seed.npz --output v4_002_exact_1197_dt_half/tangent_blocks.npz
```

With existing caches use `--existing <tangent-cache>` and name a JSON
report for `--output`. The independent JVP verification reports are
`v4_moving_cache_check_dt.json` and `v4_moving_cache_check_dt_half.json`.
Full cache regeneration is provided but was not repeated this turn; the
checks independently differentiate the actual action at both timesteps.
Each new diagnostic script takes `--dt`, `--rest-cache`, `--tangent-cache`
and `--output` (see `--help` for additional documented controls). The
tracked seed is exactly the raw `rest` array from the original archive;
archive SHA-256s will differ if regenerated, even with identical arrays.
