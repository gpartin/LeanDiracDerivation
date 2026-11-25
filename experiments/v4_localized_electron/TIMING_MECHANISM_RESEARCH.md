# V4-006 timing-mechanism research

## Controlling scope

- Framework: canonical V40 complex `Psi`/complex `chi`, C19/C19.
- Gate: movement of the same localized carrier against the centered-time,
  exact-C19 discrete Dirac comparator.
- Canonical action: unchanged.
- Research action admission: none.
- Origin story and massless light: out of scope.

## What “the clocks are off” means here

The mismatch is not primarily a common phase error.  After one internal
cycle, fitting one overall phase removes only zero to five percent of the
error, depending on copy and direction.  The plus and minus copies also
accumulate different errors.

For the central corrected-frequency control, the first four relative errors
are

`0.00023106, 0.00069153, 0.00137809, 0.00228582`.

Their second differences are

`0.00022941, 0.00022610, 0.00022116`.

The near-constant second difference diagnoses a persistent acceleration or
frequency-curvature mismatch in the co-rotating frame.  Exact common-charge
continuity passes at about `1e-10`, so this is not ordinary loss of charge.
The square-face position loop slightly reduces the long-cycle error but makes
the first four updates substantially worse.  Therefore the loop contains
useful spatial orientation information but lacks the required time-oriented
or conjugate-momentum response.

In operator language, the next audit should compare the native quadratic
matrix polynomial

`A(k) z^2 + B(k) z + C(k)`

with the discrete Dirac characteristic polynomial.  The even part under
`z <-> z^-1` measures stiffness/frequency error; the odd part measures the
missing symplectic or time-oriented response.  The same decomposition under
`k <-> -k` separates clock correction from directional transport.

## Ten mechanisms

### 1. Mixed spacetime plaquette square — highest priority

Use the factorized native face relation, but compare opposite face products
across consecutive committed layers rather than only within one layer.  A
representative defect is

`F_ab^(n+1/2) = W_i^n tensor W_(i+a+b)^(n+1)
                - W_(i+a)^n tensor W_(i+b)^(n+1)`.

Sum the positive squared norm over all three face planes and include its time
reverse so the discrete action is real and reciprocal.  It is cubic, local,
uses only `W=chi wedge Psi`, and is zero on the exact factorized rest orbit.
Unlike the tested position-only loop, it can distinguish “same orientation,
turning forward” from “same orientation, turning backward.”

Pass criterion: remove the approximately constant early second difference,
preserve the rest orbit, and improve all eight copy/direction/sign cases.

### 2. Face-loop symplectic one-form

Build a Berry/gyroscopic one-form from the two native opposite-corner face
products, for example the imaginary inner product between their difference
and their common time derivative.  This changes the symplectic form rather
than merely adding stiffness.  Full variation must generate reciprocal
`Psi` and `chi` terms, and the resulting two-form must be nondegenerate on the
physical sector.

Pass criterion: supply the odd-in-time residual left by mechanism 1 without
creating a growing or negative-energy mode.

### 3. Complete-frame face kinetic metric

Use a positive, field-dependent kinetic metric for the complete local frame
`(Psi, chi, W)` on each face.  The tested face oscillator communicated only
part of that frame.  A metric built from the factorized face Gram matrix can
make the receiving cube's acceleration depend on the sender's instantaneous
turning direction while remaining phase-neutral and reciprocal.

Pass criterion: preserve exact Noether continuity and reduce both shape error
and cross-block leakage at four updates and one cycle.

### 4. Exact/phase-fitted discrete Lagrangian control

Construct the exact discrete Lagrangian for the frozen local quadratic
carrier and compare it with the centered update.  Phase-fitted variational
integrators are designed to reproduce a declared oscillator frequency
exactly while preserving variational structure.  This is first a diagnostic:
if it fixes the mismatch without changing the continuous action, the defect
was execution-time phase error.  If the same compressed residual remains,
the action is missing a physical response.

Pass criterion: timestep refinement and exact-discrete-action results agree
on whether the residual is numerical or native.  A state-dependent fitted
Dirac frequency is forbidden.

### 5. Variational multirate fast-clock/slow-handoff update

Treat the `chi0` internal rotation as the fast variational subsystem and the
face-to-face envelope as the slow subsystem, with microsteps inside one
committed lattice interval.  This addresses fast/slow phase slippage without
breaking momentum conservation.  It changes the integrator, not the
continuous action, unless the audit proves a missing term.

Pass criterion: the result converges to the same continuous-action trajectory
and removes the handoff error uniformly under `dt`, `dt/2`, and `dt/4`.

### 6. Floquet translation-mode degeneracy

Use action coefficients only to impose a native mathematical condition: the
localized rest orbit must acquire a marginal translation Floquet multiplier
at one, with the correct Krein signature and no new unstable multiplier.
This is the lattice analogue of restoring a Goldstone-like translation mode.
The coefficient is selected by degeneracy, not by fitting Dirac velocity.

Pass criterion: a three-direction translation tangent exists and its
compressed generator matches the Dirac transport derivatives.

### 7. Derived local-rotation/face-shift factorization

Discrete Dirac quantum walks work because an internal rotation and a
conditional spatial shift occur in a phase-coherent order.  Test whether the
native discrete action's one-step map can be exactly factored into a local
`Psi`--`chi` rotation and a six-face shift, including the Baker--Campbell--
Hausdorff commutator.  The factorization must be derived from the two-field
action; an inserted coin or shift rule is prohibited.

Pass criterion: the commutator supplies the measured acceleration residual
and the product reconstructs the native update.

### 8. Two-copy quadrature relay

Retain both four-component copies and test a dynamically derived half-cycle
phase relation in which one copy prepares the receiving face while the other
completes the source release.  The earlier instantaneous copy-exchange rule
failed; the new test must use the actual Floquet eigenvectors and their
conjugate momenta, not a hand-selected exchange matrix.

Pass criterion: each copy remains measurable, total current is conserved,
and opposite directions exchange leading/trailing roles without selecting a
preferred axis.

### 9. Saturable face exchange at a radiationless point

The local onsite saturation screen cannot transmit orientation and is
retired.  A different possibility is a bounded nonlinear *bond/face*
exchange whose tangent is unchanged at rest but whose finite-amplitude clock
shift saturates during handoff.  Literature on saturable discrete NLS
lattices shows radiationless travelers only at isolated parameter/velocity
values, so the coefficient should be fixed by simultaneous zero-radiation
and translation-degeneracy conditions.

Pass criterion: bounded energy, exact rest orbit, zero resonant tail, and a
finite-velocity moving branch without fitting the Dirac trajectory.

### 10. Phase-matched self-consistent nanopteron tail

Permit a small surrounding tail as part of the freely solved state.  Its
phase must cancel the radiation emitted by the moving core and arrive at the
receiving face with the missing orientation.  The prior wake solve failed for
the current action basis, but mechanisms 1--3 can change the relevant phase
matching.  No maintained background or preloaded receiver is allowed.

Pass criterion: a relative-periodic orbit translates by one cell, repeats for
a second cell without reset, and the tail converges with box enlargement.

## Ranking and next calculation

The evidence favors a combination of **1 + 2 + 6**:

1. the cube-face factorization supplies the spatial orientation;
2. a temporal/symplectic completion supplies its missing direction in time;
3. the translation Floquet degeneracy selects the coefficient without using
   Dirac as a fitted force.

Before constructing a nonlinear candidate, calculate the full projected
matrix-polynomial residual and decompose it into even/odd time and momentum
blocks.  If the leading residual is odd in time, mechanisms 1 or 2 are the
minimal action class.  If it is even in time, mechanism 3 is sufficient.  If
it disappears under the exact discrete Lagrangian, mechanisms 4 or 5 repair
the execution rather than the governing action.

## Research basis

- Kosmas and Vlachos, *Phase-fitted Discrete Lagrangian Integrators*,
  <https://arxiv.org/abs/0903.3370>.
- Marrero et al., *On the exact discrete Lagrangian function for variational
  integrators*, <https://arxiv.org/abs/1608.01586>.
- Ober-Blöbaum et al., *Variational multirate integrators*,
  <https://arxiv.org/abs/2406.12991>.
- Di Molfetta and Debbasch, *Discrete-time quantum walks: continuous limit
  and symmetries*, <https://arxiv.org/abs/1111.2165>.
- Flach and Kladko, *Moving discrete breathers?*,
  <https://arxiv.org/abs/cond-mat/9709049>.
- Aubry and Cretegny, *Mobility and Reactivity of Discrete Breathers*,
  <https://arxiv.org/abs/cond-mat/9712046>.
- Melvin et al., *Radiationless Traveling Waves in Saturable Nonlinear
  Schrödinger Lattices*, <https://arxiv.org/abs/nlin/0603071>.
- Oxtoby and Barashenkov, *Moving solitons in the discrete nonlinear
  Schrödinger equation*, <https://arxiv.org/abs/nlin/0610059>.
- Kevrekidis and Weinstein, *Breathers on a Background*,
  <https://arxiv.org/abs/nlin/0204032>.
- Cuevas-Maraver, Kevrekidis and Pelinovsky, *Nonlinear instabilities of
  multi-site breathers in Klein--Gordon lattices*,
  <https://arxiv.org/abs/1505.02815>.
