DIRAC-030 PASS: the free electron current splits into motion and spin

The same current can be viewed in two different ways. One part follows the
electron's overall motion; the other is an intrinsic circulation carried by
its spin. This is the Gordon decomposition.

Lean did not insert a separate spin-current rule. It built the four
space-and-time direction operators from the mass grading and C19 transport
operators already in the chain. Their commutator produced the spin tensor.
Lean then proved the full operator identity and applied the positive- and
negative-side on-shell equations to obtain

    current = [ordinary flow + intrinsic spin circulation] / (2m).

The exact `1/(2m)` appeared because the mass contributes once from each side
of the current matrix element. Removing the spin term, changing a metric sign
or replacing the coefficient by `1/m` fails the general identity.

No electromagnetic field was used, so this does not yet claim a magnetic
moment or `g = 2`. It closes another property of the isolated Dirac electron:
its intrinsic spin is present not only in rotations and the relativistic spin
invariant, but also inside its conserved current.
