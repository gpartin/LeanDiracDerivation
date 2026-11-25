# V4-002 complete-frame five-RCA stop report

## Issue

The actual complete-frame clock-corrected movement action preserves the exact
localized rest orbit and is far more stable than the preceding movement
incumbent, but it retains a small exponentially growing native mode. Five
corrective action families did not put its full two-layer Floquet spectrum on
the unit circle. V4-002 therefore remains open.

## Why it matters

V4 must use one action for the resting carrier and its Dirac movement. The
clock correction cannot be credited as the final movement law while the same
action destabilizes that carrier at rest. The nonlinear 100-cycle gate and
V4-003 onward remain dependency-blocked until this necessary linear gate
passes.

## Symptoms

The screen includes every `Psi` and `chi` component on the side-five lattice
and both committed layers. The exact clock-corrected action has largest
multiplier modulus `1.0002383211986974`, growth rate
`0.061002958006188275`, and a leading mode with `97.7410%` of its power in
`Psi`. The dense solver computed all 3,000 multipliers.

| RCA | Test | Best growth rate | Verdict |
|---|---|---:|---|
| 1 | Exact complete-frame action and geometric spacetime scan | 0.0610022 | Fail; spacetime coefficient has negligible leverage |
| 2 | Matched parallel-material coefficient scan | 0.0610154 at the natural unit coefficient | Fail; weaker tends toward the old instability and stronger becomes violently unstable |
| 3 | Matched rest-clock potential scan | 0.0609140 | Fail; only a negligible change |
| 4 | Both signs at `1/361` and `1/19` of the reciprocal symplectic response | 0.0553607 | Fail; negative sign helps but plateaus, positive sign worsens |
| 5 | Triangle and transverse material-frame completions | 0.0616401 | Fail; positive completions strongly worsen |

The corresponding JSON artifacts record the exact action-source and
tangent-cache hashes. The prior older-incumbent stop report is explicitly
superseded rather than deleted.

## Expected symptoms

Every multiplier must lie on the unit circle within `1e-7` before the same
action enters the nonlinear 100-cycle perturbation, containment, conservation,
timestep, and box-refinement tests.

## Root cause

The clock-face omission was real and its repair removes most of the old
instability. The remaining mode is predominantly an internal `Psi`
amplitude/phase response. It is not controlled by further scalar clock tuning,
the tested symplectic face response, triangle completion, or transverse
material completion. The negative symplectic response couples to the correct
mode but lacks sufficient independent response rank to remove it. This is a
scoped obstruction for the tested complete-frame action vocabulary, not a
universal no-go for reciprocal C19 dynamics.

## Corrective action

Per the declared five-RCA rule, stop this stage rather than continue coefficient
tuning. The next research attempt must derive a new independent in-plane
`Psi` phase/amplitude response that is exactly rest-null, real, reciprocal,
local, and cubic symmetric, then combine it with the retained complete-frame
clock mechanism. The already-tested clock term remains the incumbent because
it produced the major improvement; it must not be discarded or replaced by
the superseded action.

## Completion

`V4-002: OPEN; STOPPED AFTER FIVE CORRECTED RCAS.` V4-001 remains passed and
pushed. V4-003 through V4-010 were not attempted on an incoherent action.
