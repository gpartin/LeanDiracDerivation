# C19 Bargmann connection report

Status: **SCOPED ACTION-CLASS FAILURE**.  Canonical V40 is unchanged.  No
research action is admitted, and no V4 transport theorem is promoted.

## Question

Can a local C19 triangle orient a quiet receiving `chi` before local `Psi`
arrives, preserve the exact V4 rest orbit, and supply the missing third
fixed-charge transport channel without inserting a Dirac matrix, selected
copy, corner label, target direction, or receiver preload?

## Candidate class

For a C19 face--edge triangle `(i,j,k)`, define

```text
B(i;j,k) = (chi_i^dagger Psi_j)
           (Psi_j^dagger Psi_k)
           (Psi_k^dagger chi_i).
```

The positive phase potential `(Im B)^2` is invariant under reversal of the
two sender vertices.  The smallest tested reversal-invariant joint kinetic
one-form is

```text
Im(B) Im(conjugate(B) dB).
```

The second expression is not a total derivative and uses both native fields.
It is a genuine field-dependent kinetic connection, not a force installed
from a target trajectory.

## Results

1. The positive phase potential gives a nonzero `chi` gradient at a receiver
   with exactly zero local `Psi`.  It therefore verifies the missing
   preorientation channel in principle.
2. The same potential has rest first-variation norm `295.919`, rest DEL RMS
   `2.9846e-2`, and fixed-charge transport rank two in all axes.  It torques
   the proved rest orbit and is rejected.
3. The joint kinetic connection has rest DEL norm `3.17e-13`, consistent with
   exact rest preservation: the Bargmann loop is invariant under the common
   rest phases, so its discrete increment vanishes on the rest orbit.
4. At a response-screen coefficient of `10^6`, the connection raises the
   numerical fixed-charge center rank to three in all axes.  The weakest
   response is only `7.87e-6`, `9.00e-6`, and `1.72e-5` of the leading
   singular response.
5. That apparent third direction is not the required four-component
   transport.  Direct compression of the exact connection Hessian leaves
   `99.45%`, `99.97%`, and `99.60%` outside the tested eight-component
   two-copy tangent in the three axes.  The off-diagonal copy blocks are also
   nonzero.  The candidate excites complementary/exterior modes and mixes the
   two copies instead of preserving one free-Dirac copy.

## Verdict

The calculation discovers a real structural fact: C19 triangle loops can
turn receiving `chi` before receiving `Psi` is populated.  The tested minimal
positive potential and minimal reversal-invariant kinetic connection do not,
however, turn that fact into the same-carrier Dirac transport operator.

Therefore:

```text
V4-005: OPEN
V4-006: OPEN
V4-007: OPEN
LEAN TRANSPORT CLOSURE: NOT PROVED
```

The next admissible calculation must derive a joint connection whose native
Hessian is closed on one geometry-derived four-component embedding (or prove
that the embedding itself must include the first shell), with a vanishing
cross-copy block and the three Clifford transport generators obtained from
that same compression.  Projecting the present response back into the sector
would be a bolt-on and is forbidden.

## Evidence

- `c19_bargmann_triangle_gate.json`
- `c19_bargmann_joint_connection_gate.json`
- `c19_bargmann_connection_transport_gate_1e+06.json`
- `c19_bargmann_connection_copy_gate.json`
