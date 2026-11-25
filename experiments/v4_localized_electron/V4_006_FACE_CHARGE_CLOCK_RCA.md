# V4-006: the moving six-face oscillator and its native phase charge

This is movement research on the frozen V4-002–005 action only. It does
not supersede those gates. The direct-prong communication and parallel
clock correction are retained; the native action, C19, both copies,
eight-site carrier and standard Lean proofs are unchanged.

## Issue

The exact same-assignment one-cell handoff remains open. The proposed
weak receiver and freely deformed relative-periodic paths can communicate
frame information, but none is an unrestricted moving solution yet.

## Why it matters

The same localized object must carry its conserved phase charge across the
face, not only reproduce a readout or overlap at an endpoint. A small
preconditioned residual is misleading if the candidate path changes its
native Noether charges or the internal oscillator is doing enormous work
that the Dirac comparator does not describe.

## Symptoms

**Baseline:** `live_clifford_bond_action.py` SHA-256
`8868f69e48f8759ef58ca34a871069acfd61947b67143a36fd7366205919c4d3`.
Both complex C3 fields, periodic side five, C19 face/edge weights 2/1,
six-face frame scale B=1197, spacetime face 6/361, matched parallel
material and potential scales 1. Timestep 1/256. Native implicit DEL;
the known 0.35-momentum independent centered-time comparator crosses one
cell at 17564 updates (`68.609375` native time). The rest array is the
tracked `v4_moving_rest_seed.npz`; the DEL tangent cache is the separately
verified V4-002 cache `0c953ff819727c38359afc7020801dee0851dbf438a36f62aac2d0fc9f8f4ef4`.
The action and comparator never supply each other's forces.

| Calculation | Result and boundary | Evidence |
|---|---|---|
| Actual action, free initial exterior/interior and crossing period | Five-node residual 0.50156 → 0.46336; holdout 0.50721. The path has up to 57.2% of the reference Ψ charge missing and 56.9% relative interval energy variation. Inconclusive. | `v4_006_phase_corrected_handoff_5.json` |
| Native phase charge at the exact rest clock | Phase momenta are Ψ 297227477.7057 and χ 113638802.2826. The six-face oscillator supplies 99.872% of Ψ's momentum and essentially all χ's, even with static χ. The older direct norm-preserving midpoint retains only 65.739% of Ψ's phase momentum at the old clock. Holding Ψ charge alone requires 1.38225 times the clock. | `v4_006_charge_clock_profile.json` |
| Eliminate both global phase impulses exactly from ordinary overlap trial | Both charge equations close to numerical precision at the sampled intervals, but maximal native acceleration defect rises from 0.76340 to 84.35453. Changing clock alone therefore does not repair that preparation. | `v4_006_charge_reduced_handoff.json` |
| Geometry-derived two-charge face-capacity seed | On sampled shape nodes, both static charges agree to less than 5e-16. The calculation does not require new action terms. Exact native phase elimination after the seed is still not a full DEL root (1.33361 maximum). | `v4_006_face_charge_transport_seed.json`, `v4_006_charge_reduced_face_seed.json` |
| Matched midpoint, both exact charges | Direct norm-preserving seed needs 1.382 times the Ψ clock and changes the fixed-charge Routhian by 370053000. Face-capacity seed retains the rest clock and changes it by 1345239, about 275 times less. Both charges close to roundoff. This measures two candidate configurations, not a minimum barrier or the Dirac energy. | `v4_006_midpoint_charge_energy.json` |
| Actual nonlinear force attribution | On that seed six-face oscillator contributes 84–98% of the signed dynamic residual. Its static DEL vanishes at the original clock. At fraction 0.4 the total dynamic defect is 1.303 versus 0.223 static and 1.279 dynamic-minus-static. Exact term sums agree to 7e-13. | `v4_006_handoff_force_decomposition.json` |
| Clock-neutral oscillator replacements | Exact Hessian deltas verified by independent JVP to ~2.3e-14. Removing Ψ's phase load worsens 10-cycle aligned Dirac error (0.0653 → 0.2742 on the existing V4-002 test family); removing W's load gives ≥7× norm growth, removing both ≥14.8×. Rejected. | `v4_006_clock_covariant_face_probe_10cycles.json` |
| Material derivative replacements | Replacing the ordinary face-frame kinetic rate using the existing native common velocity either breaks the rest orbit (Ψ/W variants) or worsens the path and short-time residual (χ-only variant). Using the exact covariant shape-rate split preserves rest but fails to improve the response. All rejected. | `v4_006_material_face_probe.json`, `v4_006_material_face_covariant_probe.json` |
| Charge-capacity chart + free period | Residual 0.83100 → 0.21079 and holdout 0.25801. Ψ charge deviation drops to 1.30%, interval energy variation to 1.44%. But the best period drifts to 35101 updates, nearly twice the Dirac comparison time. This is an improved screen, not movement agreement. | `v4_006_face_capacity_handoff.json` |
| Same capacity chart at the independent Dirac crossing time | Residual 0.43789 → 0.32038; holdout 0.50642; charge mismatch persists and tail reduces whole-carrier initial fidelity to 0.95638. Cannot be admitted. | `v4_006_face_capacity_matched_speed.json` |
| Independently vary individual three-prong phase offsets at matched speed | Collocation score 0.32038 → 0.30857 but holdout worsens 0.50642 → 0.55196. Not a stage improvement. | `v4_006_face_capacity_prong_phase.json` |
| Derive and integrate all native phase rates on matched-speed approximate orbit | Sampled phases solve both exact interval charges. Integrated periodic phase correction reduces sampled charge deviation 0.17008 → 3.945e-5 but increases max acceleration residual 0.36074 → 6.85585. Holdout charge also remains imperfect. Global phase alone cannot repair the shape dynamics. | `v4_006_charge_reduced_optimized.json`, `v4_006_integrated_native_clock.json` |
| Charge-derived first receiver phase and chi orientation | With exact original core, two possible signed Ψ precursor phases arise near ±pi/2 as amplitude tends to zero. χ tilt derived from the native variation of the W bond norm, not assigned target tilt. At 1% receiving Ψ amplitude, tilt norm is 3.63% of a complete receiving χ contrast; both native static charges close to roundoff. Orientation has only 0.568 absolute overlap with the fully translated χ contrast. This is an admissible algebraic *seed*, not evidence of spontaneous formation or transfer. | `v4_006_charge_preserving_receiver_seed.json` |
| Unrestricted release attempt from charge-derived layers | Incoming native phase-charge roots agree to ~1.8e-13. The first candidate following layer did not solve to the stated numerical tolerance for either phase branch. Rest-preconditioned implicit solver defects 0.04456/0.06821 improved to 0.02000/0.03854 with exact candidate-layer matrix-free Newton. No native update was accepted; therefore no claim about whether the precursor continues or reverses in free evolution. This is numerical-solver nonconvergence, not a physical no-go. | `v4_006_receiver_seed_release.json`, `v4_006_receiver_seed_release_newton.json` |

## Expected symptoms

The moving solution must conserve both actual phase momenta, retain the
same eight-site assignments at translated endpoints, sustain both
committed layers and satisfy unrestricted DEL between endpoints. The
phase relation has to be consistent across the whole evolution, not
reset at each frame. The independent Dirac comparison must agree at the
same low-momentum crossing time. The already-passing rest, local
four-component readout, spin and current are not replaced by new criteria.

## Root cause

The present face-frame term is not merely an orientation spring. It
also carries the majority of the oscillator's global phase momentum.
For undirected cubic face differences `D`, `W=chi wedge Psi`, and
the positive rest phase increment theta, directly varying the two
independent global phase angles gives its exact rest contribution

```
Q_Psi_face = (B sin(theta)/dt) (sum|D Psi|² + sum|D W|²/chi0²)
Q_chi_face = (B sin(theta)/dt) sum|D W|²/chi0².
```

The direct automatic phase derivative independently reproduces this
identity. Thus the oscillator's effective phase inertia changes if a
trial redistributes the live frame between cubes while holding simply
sum|Psi|² fixed. The old clock is correct at rest but is not sufficient
to maintain charge on arbitrary intermediate shapes. Raising the clock
on such a shape restores the phase momentum at the cost of a much
larger native DEL mismatch. Conversely, removing the oscillator's
phase load entirely changes the gyroscopic response that was required
for the prior stability and local transport passes. The clock-neutral
replacement was therefore rejected instead of silently promoted.

The separate dynamic force attribution identifies the next mechanism
more precisely: **the B-normalized oscillator's response to changing
face-frame geometry dominates the error on the tested charge-consistent
moving preparation**. It is not a missing basic three-prong link, the
old parallel-overlap clock term, or a generic lack of χ stiffness. The
tested ordinary material and clock-covariant shape substitutions do not
remove this response while retaining the original rest and transport.
This does not rule out a different fully native moving solution or a
different reciprocal reduction of that term.

There is also a positive local algebraic result. For a receiver tail
`epsilon exp(i phi) U`, exact preservation of the face-Ψ phase capacity
with the source core unaltered requires

```
2 epsilon Re(exp(i phi) <D Psi, D U>) + epsilon² ||D U||² = 0.
```

The two solutions of this cosine equation are close to opposite
quarter-turns for a weak precursor. With that Ψ signal, the W-face norm
depends quadratically on the amplitude of a receiver-only χ tilt. Its
native gradient and the quadratic root supply a candidate initial χ
reorientation. Refining the two coordinates against the full phase
momenta gives an exact static charge seed, without a new equation or
runtime force. These two phases are **not** identified with the two
Clifford copies; that additional physical connection is unproved.

The phase-reduction method follows from the Abelian variational principle
used in Jalnapurkar, Leok, Marsden and West, [Discrete Routh Reduction](https://arxiv.org/abs/math/0508330).
For a moving relative-periodic lattice mode, the fixed point must be the
native flow composed with a lattice shift, as in the scope of
Flach and Kladko, [Moving discrete breathers?](https://arxiv.org/abs/cond-mat/9709049).
Neither paper establishes existence for the present LFM action. The
sampled midpoint reduced functional is not a proved Peierls–Nabarro
barrier or an exact discrete conserved Hamiltonian.

## Corrective action

1. Keep V4-001–005 and the exact frozen action as the control. Use the
   native phase-momentum equations to derive a self-consistent two-layer
   receiver precursor, test its first push and reversal, and retain only
   unreset native trajectories. A charge-constrained path is merely a seed.
2. Work on the finite transfer phase/shape equations together. The
   dominant rotating face energy must supply a reciprocal material
   response while retaining its first-order gyroscopic effect. Do not
   merely remove the face oscillator, spin clock, or re-add generic
   receiver coupling. The actual native local force and momentum
   derivative are the inverse requirements; Dirac remains the comparator.
3. If changing the action is required, derive the smallest replacement
   that preserves the native resting phase, full reciprocal variations,
   and the existing passing local transport. Revalidate V4-002–005 only
   on an admitted change, not on a discarded trial. Solve the full
   relative-periodic trajectory, then demonstrate independent free release
   at the matched C19-Dirac crossing time before calling V4-006 PASS.

The use of a five-node chart is a numerical screening limitation. No
solver optimum, Routhian sample, frozen readout or initial charge root is
presented as a Lean theorem or an exact moving electron. All action
variants remain research comparisons. Main is untouched.

### Reproduction and checkpoints

All scripts are in this directory. Supply `--rest-cache v4_moving_rest_seed.npz`
and, where required, `--tangent-cache v4_002_exact_1197/tangent_blocks.npz`.
The cache can be rebuilt using `v4_rebuild_moving_tangent.py`. Set BLAS and
OpenMP threads to one or two as appropriate. The default timestep is 1/256,
axis x-negative, comparator period 17564. Use a separate `--output` for any
replay so the captured evidence stays intact.

The finite trial states are tracked as small NPZ checkpoints (not action
coefficients or native update rules). For `v4_006_phase_corrected_handoff.py`:

| Captured run | Additional options |
|---|---|
| `phase_corrected_handoff_5` | `--nodes 5 --iterations 45 --seed local` |
| `face_capacity_handoff` | `--nodes 5 --iterations 22 --warm-start v4_006_face_charge_transport_seed.npz --project-face-capacity --fast-exact-reduction` |
| `face_capacity_matched_speed` | `--nodes 5 --iterations 20 --warm-start v4_006_face_capacity_handoff.npz --fixed-period-updates 17564 --project-face-capacity --fast-exact-reduction` |
| `face_capacity_prong_phase` | `--nodes 5 --iterations 16 --warm-start v4_006_face_capacity_matched_speed.npz --fixed-period-updates 17564 --project-face-capacity --phase-chart prong --fast-exact-reduction` |

Each run takes a separate `--state-output` checkpoint. The fast exact
interval reduction cancels repeated copies of the same incumbent action;
it was cross-checked at approximately 2.1e-13 relative DEL defect and the
final and independent holdout evaluations use the original action. For
the remaining trials use `--help`; the output records the mode, period,
action hash, and state/candidate hash where applicable. The phase-reduced
and integrated clock reports use the tracked warm-start states. The
receiver release was run with four requested updates and with two
requested updates plus `--matrix-free-newton`; neither accepted a first
update. `v4_006_face_charge_artifact_audit.py` documents all output and
checkpoint digests and confirms the stage statuses without granting
movement credit.
