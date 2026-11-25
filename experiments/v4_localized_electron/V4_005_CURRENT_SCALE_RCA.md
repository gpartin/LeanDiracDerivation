# V4-005 current-scale RCA

Issue: The first current comparison treated microscopic phase flow among the
eight carrier corners as though it were the center-to-center Dirac current.

Why it matters: V4-005 must connect transport to a current derived from the
same native action. A current at the wrong spatial scale cannot establish that
bridge even when charge is exactly conserved.

Symptoms: The raw site-phase bond current was dominated by the carrier's
internal circulation and did not approach the four-component packet current.

Expected symptoms: A center-resolved current should have the same sign as the
independent discrete Dirac packet, satisfy the local charge balance, and show
shrinking error under timestep refinement.

Root cause: The fixed four-component readout is a coarse carrier coordinate.
Its local phase transformation is spread through the native Psi and chi lift,
whereas a phase change at one microscopic corner also measures internal
circulation. These are different Noether probes of the same action.

Corrective action: Pull a local four-component phase variation through the
already frozen V4-003 carrier lift and contract it with the exact automatic
derivative of the unchanged action. The resulting center-resolved Noether
current has the correct sign in both copies and all signed axes; its aggregate
flux error falls from 0.00267008 at dt to 0.00145136 at dt/2. No force, readout
fit, action term, projection, or runtime support was added.
