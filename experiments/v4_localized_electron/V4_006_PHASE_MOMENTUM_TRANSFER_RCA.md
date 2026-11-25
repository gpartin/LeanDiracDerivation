# V4-006 — Derived native face phase-momentum transfer

Status: **OPEN**. Frozen V4-002–005 research action, complex Psi and complex chi, C19 native stencil; the established extra matter frame term uses the six faces. No new action admitted. This work is a scoped mechanism theorem and a failed full-cell orbit screen, not an early-stage regression.

## Issue

The established direct-prong and clock-matched frame action preserves the
rest carrier, but a full finite cell crossing at the independent Dirac
crossing interval has not been obtained.

## Why it matters

Both independent native phase charges must reach the next cube with the
same carrier, not just the field amplitudes or the receiver's orientation.
A conserved global charge alone does not establish local delivery.

## Symptoms

The face oscillator carries nearly all the rest Psi phase momentum (99.87%
in the preceding study). The exact native receiver cut on the original two
transfer paths gives the following fractions of the required phase charge:

| Trial | Psi | chi |
|---|---:|---:|
| Original seed | -24.37% | -19.65% |
| Charge-capacity matched speed | -22.58% | -58.40% |
| Opposite-phase endpoint, nonlinear boundary trial | +20.50% | +9.88% |
| Same, with local phase-balance residual in the solver | +20.41% | +10.55% |

Negative means native current is leaving the receiving sites despite the
needed gain. Screening relative endpoint phases 0, pi/2, pi and -pi/2 on
each of the six signed axial paths identifies pi as the direction-correct
branch; the face-only charge delivered on that branch is approximately
19–21% for Psi and 10–17% for chi. This is an evaluated candidate
trajectory, not free evolution. The half-turn is a relation between donor
and recipient phases, **not** identification of the two Clifford copies.

The nonlinear trial used the independent crossing interval of 17,564
updates at dt=1/256. Its normalized acceleration defect went from 0.77472
to 0.77389 (holdout 0.78596). Adding the exact phase-balance contraction of
the existing equations as a solver diagnostic gives 0.71926 (holdout
0.82208). These are not small enough to certify a trajectory. No receiving
state or phase rule was inserted in the action.

## Expected symptoms

Unrestricted native evolution would deliver both required charges to the
same translated eight-site shape at the independent discrete Dirac crossing
time. The source would release what the receiver gains, with no artificial
phase reset. All six signed directions and a subsequent unreset crossing
must still be tested on a genuine orbit. The existing stationary and
short-time V4-001–005 gates remain unchanged.

## Root cause

### Exact one-face calculation

For either charged native face field F = Psi or W = (chi wedge Psi)/chi0,
let d_- = F_j^- - F_i^-, d_+ = F_j^+ - F_i^+, h = dt, k = B/h, and
c = cos(theta) = 1 - h^2 chi0^2/2. The frozen matched face oscillator is

    L_e = k/2 [c (|d_-|^2 + |d_+|^2) - 2 Re(d_-† d_+)].

Differentiating this action at each endpoint gives

    P_j^- = k(c d_- - d_+)
    P_j^+ = k(c d_+ - d_-)
    Q_j^± = Im(F_j^±† P_j^±).
    I_i→j = Q_j^- + Q_j^+.

In particular this is the antisymmetric bond inflow

    I_i→j = k Im[F_j^-† F_i^+ + F_j^+† F_i^-
                  - c(F_j^-† F_i^- + F_j^+† F_i^+)].

For independent phase variations of the native fields, W contributes to
both Psi and chi. The separate native chi face kinetic term uses c=1 and
its own mass coefficient. The remaining native action terms are included
in the numerical full-action cut by differentiating local phase shifts.
Their total local balance is the phase contraction of the full discrete
Euler–Lagrange equation. This is a derived current, not a Dirac force.

On a single shared face, take co-rotating amplitudes F_i=a and F_j=-b,
with a+b=A at every layer. Then the bond difference stays -A in the
rotating frame and its discrete matched-clock acceleration vanishes
exactly at the middle layer. Its endpoint phase momenta are

    Q_j^- = -k sin(theta) A b_n
    Q_j^+ =  k sin(theta) A b_(n+1)
    I_i→j = k sin(theta) A (b_(n+1) - b_n).

An advancing b therefore receives phase momentum while the shared face
clock remains intact. This is an exact one-bond result and is conditional
on those two amplitudes; the equation does not force the whole cube
to follow that partition. Lean proves the interval-polynomial variation,
its derivative, antisymmetry, the explicit inflow, the pair identity and
the zero-clock-DEL identity in FacePhaseMomentum.lean with standard
foundations only. The numerical current formula agrees with independent
automatic differentiation to under 3e-14 relative defect.

### Why the whole cube is different

In the rotating frame F_± = exp(±i theta)(F ± h u), set
K=B sin(theta)/h and L6 the positive six-face Laplacian. To linear order
in velocity the storage and signed inflow are

    R_i = Re(F_i† (L6 u)_i), S_i = Re(u_i† (L6 F)_i)
    storage rate = K (R_i + S_i)
    inflow rate  = K (S_i - R_i)
    imbalance    = 2 K R_i.

The link contribution is K Re(F_j† u_i - u_j† F_i).
The wedge velocity is (u_chi wedge Psi + chi wedge u_Psi)/chi0.
An independent odd finite-difference test of this identity gives worst
defects about 1.2e-8. The local inverse system shows that phase-only
velocities cannot generally retain the proposed redistribution and close
local phase balance. Allowing all joint native velocities can solve the
linear face balance but calls for large changes and worsens the full
equations on the tested path. It is not a complete moving preparation.

The opposite-phase path's net receiver influx in this face oscillator is
almost entirely along the motion axis. At the endpoint, about 61–70% of
Psi and 62.5–75% of chi static face charge on the receiving sites belongs
to the two other bond orientations. The single-face solution does not
account for rebuilding these transverse stores. All orientations share
the same local native fields; their individual phase momenta need not
be separately conserved. This establishes a concrete deficit on the
tested transfer path, not a no-go theorem for the action or proof that
an additional coupling is necessarily required.

## Corrective action

A research replacement of the matched kinetic rate by a skew-adjoint
material derivative of the actual face frame was screened. Its unaligned
version destroys rest and gives reverse current. Its overlap-aligned
version restores rest to 4.0e-12 preconditioned residual but supplies
essentially the same insufficient current (20.5015% Psi, 9.8643% chi
versus the baseline 20.5016%, 9.8778%). Both are rejected; the frozen
action is unchanged.

Next derive the exact individual-orientation phase balance, including
the common local Psi/chi contribution, and solve for a self-consistent
redistribution among the six face oscillators during a handoff. The
derived one-face identity is a useful control, not an imposed endpoint
clock. Verify that a joint solution simultaneously closes the complete
native equations and the two current budgets at the Dirac crossing
interval before any free-evolution claim or action promotion.

Evidence: v4_006_face_phase_momentum_{seed,matched,antiphase_bvp,antiphase_balanced}.json,
v4_006_relative_receiver_phase_transfer.json, v4_006_antiphase_face_routing.json,
v4_006_local_phase_transfer_rank{,_antiphase}.json,
v4_006_antiphase{,_phase_balanced}_handoff.json and their saved states,
v4_006_face_frame_transport_probe.json. The new Lean theorem remains a
research module, not a cumulative publication theorem. V4-006 stays OPEN.
