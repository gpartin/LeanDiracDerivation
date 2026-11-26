# V4-006 — Cross-face exchange and the missing net through-current

**OPEN — frozen V4-002–005 research action.** Complex Psi and complex chi,
C19 field geometry, six-face matched matter oscillator, dt = 1/256, 17564
updates per independently specified comparator crossing. Scope: exact
local algebra and finite nonlinear local tests of a mid-handoff
preparation; no unrestricted full-cell orbit or electron-motion claim.

## Issue

The one-face opposite-phase identity from the preceding RCA gives a
direction-correct flux, but the proposed full-cube transfer fails to
rebuild the momentum stored in the other face orientations.

## Why it matters

A native handoff must conserve the two phase charges locally and move
both with the same translated carrier. Correct phase current direction
or a prepared destination cannot by itself establish a crossing.

## Symptoms

The exact axis-resolved charge budgets of the preceding opposite-phase
x-negative BVP trial are below (millions of native phase-momentum units;
33-point quadrature). Each entry is Psi, chi.

| Bond axis | Storage gain | Net inflow into four receiver sites | Unbalanced |
|---|---:|---:|---:|
| Motion axis x | 47.43, 21.30 | 29.58, 5.61 | 17.85, 15.69 |
| y | 56.24, 21.30 | ~0, ~0 | 56.24, 21.30 |
| z | 40.33, 14.20 | ~0, ~0 | 40.33, 14.20 |

All three orientations require additional charge on the trial path.
There is no surplus from the motion axis for the other two. Internal
redistribution alone cannot fix the total deficit: it sums to zero
between stores. Full native missing integrated charge on this path is
114.61 million Psi and 51.40 million chi. The same-phase comparison has
still larger deficits; the phase-conditioned BVP still lacks both
charges.

At quarter crossing, three genuinely solved successive native DEL steps
from the opposite-phase trial's two committed layers do demonstrate
exchange between face orientations, without any imposed reset. On the
first, the x-oriented exchange is negative (-41140 Psi and -200133 chi
per lattice time), while the y and z exchanges are positive. The other
action terms compensate the remaining balance. Total receiver phase
storage growth equals native influx within numerical error. But it is
only 29.60% and 27.49% of the trial's required local rate. At midpoint it
is 14.85% and 15.04%; the third step gives 13.81% and 22.41%. All six
unrestricted solved steps have normalized DEL below 2e-8. The
three-quarter solve stalled at 6.46e-4 and is not accepted as a native
update.

## Expected symptoms

The two local phase conservation equations hold and their signed
receiver influx supplies the required total charge consistently over
the full crossing interval, in addition to the native field equations,
the fixed Dirac readout, and the full translated endpoint. There must be
no later reset or enforced pattern. Near-midpoint local current agreement
alone is not sufficient.

## Root cause

### Actual oriented balance and live companion curvature

For each Cartesian face orientation a, let L_a F = 2F - F_(+a) - F_(-a).
The actual matched clock defects on three consecutive native layers are

    R_Psi = Psi_- + Psi_+ - 2 c Psi
    R_chi = chi_- + chi_+ - 2 chi
    R_W   = W_- + W_+ - 2 c W
    W = (chi wedge Psi)/chi0, c=1-dt² chi0²/2.

The exact finite-layer product rule is

    chi0 R_W = chi wedge R_Psi + R_chi wedge Psi
             + (chi_- - chi) wedge (Psi_- - Psi)
             + (chi_+ - chi) wedge (Psi_+ - Psi).

The last two terms are the native mixed motion contribution; there is
no frozen-companion or small-velocity assumption. At a site, the
orientation a change of the phase momentum store minus the signed
inflow from that axis is

    E_Psi,a = (B/dt) Im[Psi† L_a R_Psi + W† L_a R_W]
    E_chi,a = (B/dt) Im[B chi† L_a R_chi + W† L_a R_W].

These are the negatives of the corresponding directional interval
action phase variations. Sum over axes and the remaining native action
terms to obtain the phase contraction of the full DEL. Only that total
must vanish for a native trajectory. The action is varied independently
for each term; these are diagnostic partitions, not independent fields.
The exact formulas, wedge curvature, and derivative of the two-interval
action are verified in Lean. The Python formulas agree with independent
automatic differentiation to under 6e-13 of rest phase-momentum scale.

For the opposite-phase trial the common mixed Psi–chi motion contributes
about 4.67 million to the integrated phase imbalance in each field, but
the direct Psi clock defect and the wedge's response to it dominate.
The independent chi centered acceleration term in the wedge is only
about -40 in these units. The trial's native forces do not support the
prescribed rates, not simply a static orientation defect.

### Two independent phase channels

The wedge is a sum-phase field: independent local Psi or chi phase
variation gives the same infinitesimal change in W. The existing parallel
component S = chi†Psi/chi0 is the difference-phase field; its two phase
variations have opposite signs. Thus an arbitrary wedge-only contribution
cancels identically from the difference of the two native charges. The
parallel-only contribution cancels from their sum. These are necessary
response constraints, not a law forcing two physical copies.

Term-by-term exact action differentiation on the opposite-phase trial
shows the matched oscillator produces nearly the entire phase deficit
(114.32 million Psi and 51.14 million chi in 17-point quadrature). Its
half-sum and half-difference deficits are 82.73 million and 31.59 million
respectively. The actual existing direct-prong and parallel-material
terms contribute negligible incoming net phase charge on that trial.
Simply scaling the wedge cannot independently correct both constraints.

## Corrective action

First screen the already implemented matched face oscillator for S at
the same B normalization as W. This is the natural parallel completion
using the complete local relative-frame norm, not a fitted coefficient.
It preserves the exact rest orbit (preconditioned residual 1.07e-11),
and its opposite phase contributions match the formula to 1.9e-18.
But its added net inflow is only -11.55 Psi, +11.55 chi units, compared
to the required gains in the hundreds of millions. Both delivered
fractions and dynamic residual are essentially unchanged. The
candidate action is rejected; the incumbent is unchanged.

Next, the inverse initial-data problem uses the same complete native
action to solve the two global rest charges and the two required
receiver through-currents simultaneously. It does not adjust any action
coefficient. Given the selected midpoint configuration, solve for the
next layer by differentiating these four quantities with respect to all
native component coordinates. Then solve the original backward DEL for
the preceding layer, using the previous-layer Jacobian rather than the
forward one. The resulting two committed layers are tested by autonomous
native evolution. No subsequent current or phase correction is applied.

This inverse solve reaches both charges and target currents at the
chosen midpoint to a relative residual 2.79e-12. The actual backward
DEL closes to 2.96e-11. It then advances autonomously for three further
updates, with full normalized DEL below 1.55e-11, both global charge
defects below 8.4e-13, and the following fractions of the trial's
midpoint-required currents:

| Update after the inverse pair | Psi | chi |
|---|---:|---:|
| Prepared current (not a prediction) | 100% | 100% |
| First free update | 89.70% | 99.44% |
| Second free update | 77.88% | 99.54% |
| Third free update | 64.86% | 100.50% |
| Uncorrected preparation, third free update | 13.81% | 22.41% |

Thus a different preparation, not a force change, produces substantially
larger and temporarily sustained native current for both fields.
Yet Psi is already declining, the corrected layer displacement is seven
times the original slow envelope step (0.122% of the rest contrast),
and its energy diagnostic is 2.175% above the resting reference;
the energy varies by 2.8e-6 relative to rest over these steps. The
extra energy may reflect the selected shape or the launch, not a proved
Dirac kinetic energy. A separate five-constraint control using the rest
energy itself reduces the mismatch to 0.06856 but fails to close it;
it is not released as a successful launch. These are mid-handoff
preparations, not a carrier launched from unprepared vacuum, and 3
updates are not 17,564. They do not establish full-cell translation.

### First departure of the improved native launch

The new free layers identify the direct Psi oscillator, not the chi
orientation exchange, as the main remaining error. Over the first
three unrestricted updates the total Psi cut current falls 2.362
million units/time. The direct Psi face term loses 1.847 million
(78.2% of the total loss), the shared wedge loses 0.490 million, and
the other terms lose 0.025 million. Chi's direct channel and the other
native terms gain enough to compensate the wedge loss; chi's total
current changes only +0.014 million.

The following exact two-interval advance identity (proved in Lean)
separates the direct clock rotation from its field-equation defect.
Set delta_i = F_i,+ - F_i,-, R_i = F_i,- + F_i,+ - 2c F_i, and k=B/dt:

    I_n - I_(n-1) = k Im[(1-c²)(F_j† delta_i + delta_j† F_i)
      - c/2 (delta_j† R_i + R_j† delta_i)].

For the direct Psi cut, the centered-clock term accounts for 1.306
million of the 1.847 million drop (70.7%); the nonzero clock-defect
term contributes 0.541 million (29.3%). The exact numerical split
agrees with the independent interval current to an absolute
2.3e-6 units/time at the tested cut. Thus the wave phase and velocity
correlation itself evolves in the wrong direction, even before
accounting for the additional coupled-force correction. The next
inverse calculation is to prepare the native phase momenta so both
current and its first advance agree with the required rate, including
global charges and the independently justified moving (not arbitrarily
rest) energy. Test the resulting freely evolving pair; a failure
identifies the residual to repair in the reciprocal wave clock action.

Artifacts: oriented_phase_exchange_{matched,antiphase,balanced}.json,
phase_sum_difference.json, parallel_face_charge_completion.json,
current_inverse_launch{,_energy}.json and saved initial/updated layers,
inverse_launch_current_decay.json.
All file names use the v4_006_ prefix. The exact local results are in the
research Lean module FacePhaseMomentum.lean; they do not change the
publication theorem.
