# V4-002 complete-frame five-RCA stop report

> **Gate correction (2026-11-25):** This report is retained as the history of
> the five action searches. Its demand that every arbitrary full-phase-space
> multiplier lie on the unit circle is not the controlling V4-002 criterion.
> The controlling test is now physical-sector reachability. That narrower test
> still fails: `v4_002_reachable_sector_audit.json` shows the actual two-layer
> Dirac packet launches growing to norm ratio `4.8200821871` and shape error
> `4.6536528032` over 100 cycles.

## Issue

The actual complete-frame clock-corrected movement action preserves the exact
localized rest orbit and is far more stable than the preceding movement
incumbent, but it retains a small exponentially growing native mode. Five
corrective action families did not put its full two-layer Floquet spectrum on
the unit circle. V4-002 therefore remains open.

## Why it matters

V4 must use one action for the resting carrier and its Dirac movement. The
clock correction cannot be credited as the final movement law while the same
action destabilizes that carrier at rest. The nonlinear 100-cycle gate was not
run at that point. V4-003 onward were treated as dependency-blocked by that
ambient linear gate. The controlling correction above now blocks them on the
demonstrated growth of the actual carrier launch, not on the ambient spectrum
alone.

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

This report originally required every multiplier to lie on the unit circle
within `1e-7`. That requirement is superseded. The current criterion requires
bounded evolution of the physical carrier/rest/movement perturbations with
containment, conservation, timestep, and box-refinement checks. The exact
complete-frame action fails that narrower criterion as recorded above.

## Root cause

The earlier “stable orbit” did not establish perturbative stability. Its
100-cycle zero-`chi`-charge run followed the exact unperturbed orbit; the later
full two-layer audit already showed that nearby native perturbations grow at
approximately `1.036`. The separately described Floquet-stable wake result
covered a reduced/common channel rather than the full native phase space. A
new dense audit of all retained wake caches confirms that none is stable; the
best tested wake variant grows at `0.14999`.

The movement additions did not destroy an already perturbatively stable
full-system orbit. They improve it. The historical wake base tested here
grows at `1.61850`; adding the live movement connection and six-face frame
oscillator lowers this to `0.85391`; adding the matched parallel clock
coordinate lowers it again to `0.06102`. The clock-face omission was therefore
real and its repair removes most of the inherited instability.

The remaining mode is predominantly an internal `Psi` amplitude/phase
response already traced to negative bare-V40 generalized curvature. It is not
controlled by further scalar clock tuning, the tested symplectic face
response, triangle completion, or transverse material completion. The
negative symplectic response couples to the correct mode but lacks sufficient
independent response rank to remove it. This is a scoped obstruction for the
tested complete-frame action vocabulary, not a universal no-go for reciprocal
C19 dynamics.

## Corrective action

Per the declared five-RCA rule, stop this stage rather than continue coefficient
tuning. The next research attempt must derive a new independent in-plane
`Psi` phase/amplitude response that exerts zero force on the exact rest orbit
but positive curvature on its perturbations, is real, reciprocal, local, and
cubic symmetric, then combine it with the retained complete-frame clock
mechanism. The already-tested clock term remains the incumbent because it
produced the major improvement; it must not be discarded or replaced by the
superseded action.

## Completion

`V4-002: OPEN; STOPPED AFTER FIVE CORRECTED RCAS.` V4-001 remains passed and
pushed. V4-003 through V4-010 were not attempted on an incoherent action.
