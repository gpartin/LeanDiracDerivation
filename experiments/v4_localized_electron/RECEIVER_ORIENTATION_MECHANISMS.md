# Ten mechanisms for receiver orientation

Date: 2026-11-24

Scope: research mechanisms by which an incoming native `Psi` pattern could
cause complex `chi` at a receiving C19 center to acquire the compatible
four-component orientation. This is a research ranking, not an equation
change, action admission, or V4-006 pass.

## Acceptance filter

An admissible LFM mechanism must:

- use only native complex `Psi` and complex `chi`;
- arise from one real, local, reciprocal action;
- preserve cubic/C19 symmetry, both cubical copies, and the resting carrier;
- transfer phase, handedness, and component ordering, not merely energy;
- require no receiver preload, selected direction, target spinor, runtime
  reset, projector, or external clock; and
- reproduce the frozen discrete-Dirac comparator rather than supplying its
  forces.

## 1. Native non-Abelian parallel transport

**Physical analogue:** Polarization and other multi-component waves carry an
internal frame. A matrix-valued connection rotates that frame as the wave
crosses a link; the order of rotations can matter.

**Logical analogue:** A message carries both data and a coordinate-system
header, allowing the receiver to interpret the components in its own basis.

**LFM realization:** Derive a C19 link connection from the complete local
`Psi`--`chi` frame, using invariant data such as `chi^dagger Psi`,
`chi wedge Psi`, and their time derivatives. Put the connection in a real
kinetic or symplectic action, so reciprocal `Psi` and `chi` responses follow
by variation.

**Evidence/status:** Highest priority. The hand-selected spin-diagonal frame
reduced one-cycle error to `0.217340`, while the first step was already accurate
to `4.56e-6`. The simple overlap-derived frame was rank deficient, so the
connection must use complete phase-space information rather than configuration
overlap alone.

## 2. Self-generated coin-and-shift dynamics

**Physical analogue:** A discrete quantum walk alternates an internal-state
rotation with a conditional spatial shift. Dirac dynamics can arise from the
continuum limit of such walks.

**Logical analogue:** Update the direction register, then move the token in the
direction selected by that register.

**LFM realization:** `chi` motion supplies the local internal rotation, while
C19 `Psi`--`chi` bond current supplies the conditional shift. Both operations
must arise simultaneously from the autonomous continuous action; an imposed
split-step rule would be a bolt-on.

**Evidence/status:** High priority as a design identity, but likely the
discrete-time description of mechanism 1 rather than a separate fundamental
law. It directly explains why moving amplitude without rotating the receiver
fails.

## 3. Ablowitz--Ladik-type nonlinear coherent hopping

**Physical analogue:** In exceptional nonlinear lattices the hopping strength
depends on the wave amplitude. The localized waveform changes the link it is
crossing, allowing exact or unusually mobile discrete solitons.

**Logical analogue:** The packet creates its own temporary communication
channel instead of moving over a fixed network.

**LFM realization:** Replace scalar hopping by an invariant, matrix-valued
coherent bond response constructed from both fields. The same action term
would move `Psi` and turn `chi`, so the track travels with the carrier.

**Evidence/status:** Still viable only in a kinetic/symplectic or complete-
frame form. The 32 tested position-only coherent-hopping and redistribution
directions did not close the residual; their best rest-preserving residual was
`0.249661`.

## 4. Translation-zero-mode or exceptional discretization

**Physical analogue:** Ordinary nonlinear lattices pin localized waves because
site-centered and bond-centered configurations have different energies. An
exceptional discretization can restore a neutral translation mode and remove
the Peierls--Nabarro barrier.

**Logical analogue:** Every intermediate handoff state costs the same, so the
token has no preferred node at which to become stuck.

**LFM realization:** Arrange the reciprocal face/edge action so the carrier
possesses a zero-curvature center-displacement tangent while its internal
frame remains closed.

**Evidence/status:** Necessary for continuously slow localized motion, but not
sufficient for orientation. The current carrier has translation curvatures
about `0.218`--`0.220`. Simple geometry-derived face/edge cancellation ratios
were nearly blind to that channel. This mechanism must be combined with 1 or
3, not pursued as another scalar stiffness scan.

## 5. Self-consistent precursor and traveling tail

**Physical analogue:** Moving discrete breathers and lattice solitary waves
often deform and carry a small oscillatory or evanescent tail. The leading tail
prepares the medium before the core arrives.

**Logical analogue:** A packet sends a setup message ahead; the receiver
configures itself before the payload arrives.

**LFM realization:** A weak leading `Psi`--`chi` tail rotates the receiving
`chi`; the core then arrives in the prepared native frame. The tail must be
part of the freely solved state and cannot be maintained or preloaded.

**Evidence/status:** High priority. The finite-orbit RCA found that a rigidly
translated rest profile leaves a dominant reversal-even `chi` inertial
residual. This is direct evidence that the moving state needs a deformed `chi`
profile or tail. The unrestricted relative-periodic traveling branch is the
right test.

## 6. Excitable or autocatalytic `chi` front

**Physical analogue:** In excitable and reaction--diffusion media, a small
leading signal pushes the local medium across a threshold; the activated
medium then reconstructs the same front at the next location. Coupled hidden
variables provide memory and recovery.

**Logical analogue:** A domino does not carry its material forward; it triggers
the next domino into the corresponding state.

**LFM realization:** Incoming invariant orientation, not density alone, moves
`chi` between symmetry-related local branches. The reciprocal Hamiltonian
version would need a conservative partner motion rather than dissipative
resetting.

**Evidence/status:** Conceptually plausible but risky. Ordinary excitable
models rely on dissipation and thresholds, which conflict with the bare
Hamiltonian gate. A conservative local bifurcation derived from the existing
complex `chi` potential could be tested, but a hard threshold or state reset
would be a bolt-on.

## 7. Adiabatic dark-state transfer

**Physical analogue:** STIRAP/CTAP transports an amplitude between endpoints
through a changing coupled eigenmode, with little occupation of the
intermediate state. The state follows a rotating dark eigenspace.

**Logical analogue:** The receiver establishes a connection before the sender
lets go, creating a make-before-break handoff.

**LFM realization:** The overlap of the leading carrier tail with receiving
`chi` strengthens the receiver bond while the sender bond weakens, causing the
joint `Psi`--`chi` eigenmode to rotate from one center to the next.

**Evidence/status:** Worth a targeted eigenmode test. External pulse sequencing
is forbidden, so the counter-intuitive coupling sequence must be generated by
the carrier's own field. This may be a special realization of mechanism 5.

## 8. Local phase entrainment or injection locking

**Physical analogue:** Coupled oscillators exchange phase error through a sine
of their phase difference and can synchronize frequency and phase. A traveling
phase wave can entrain successive oscillators.

**Logical analogue:** The sender and receiver perform a clock-and-phase
handshake before transferring ownership.

**LFM realization:** C19 invariant phase differences built from
`chi^dagger Psi` and oriented-area phases exert equal and opposite torques on
sender and receiver. The coupling must resolve all four internal directions,
not average them to one scalar phase.

**Evidence/status:** Scalar and position-only phase-locking versions have
already been insufficient. A complete-frame velocity-dependent entrainment
term remains part of mechanism 1's open action class.

## 9. Autonomous topological pump

**Physical analogue:** A cyclic change of internal parameters can transport a
wavepacket or charge by an integer number of cells per cycle. Higher-order
topological pumps can move states from corner to corner.

**Logical analogue:** The internal state walks around a closed state-machine
cycle, and completing the cycle advances the location counter by one.

**LFM realization:** The carrier's existing internal rotation could trace a
closed loop in native `Psi`--`chi` state space. If the C19 bond current has
nonzero geometric holonomy over that loop, one rotation cycle could translate
the carrier while returning its internal pattern.

**Evidence/status:** Attractive because it ties one internal cycle to one
handoff without equating a numerical timestep with a substrate tick. It must
be autonomous and direction-degenerate; imposed periodic modulation or a
preferred pump direction would be a bolt-on. The nontrivial holonomy measured
in the native frame makes this testable.

## 10. Two-copy relay or staggered handshake

**Physical analogue:** In quantum walks and staggered-fermion constructions,
internal components move conditionally on alternating sublattices. One sector
can lead while the other supplies the complementary update.

**Logical analogue:** Two buffers alternate roles: one transmits while the
other verifies and reconstructs, then they exchange roles.

**LFM realization:** The two native four-component cubical copies exchange
leading and trailing roles through a reciprocal local connection, preserving
both rather than deleting a doubler.

**Evidence/status:** Lower priority but not universally excluded. The simplest
exact copy-exchange involution produced one-cycle error `0.574072`, so the
second copy alone is not the missing mechanism. A more general dynamic
cross-copy connection would have to emerge from the action and improve both
the frame and current gates.

## Ranking for the next calculation

1. Native phase-space parallel transport (1).
2. Self-consistent precursor/traveling tail (5).
3. Their autonomous coin-and-shift interpretation (2).
4. Matrix-valued Ablowitz--Ladik-like kinetic hopping (3).
5. Autonomous geometric/topological pumping (9).
6. Translation-zero-mode completion combined with frame transport (4).
7. Adiabatic dark-state handoff (7).
8. Complete-frame phase entrainment (8).
9. Conservative excitable `chi` front (6).
10. Generalized two-copy relay (10).

The first falsifiable calculation should combine 1 and 5: derive the complete
phase-space link frame along the leading tail of an unrestricted traveling
ansatz, then ask whether a real reciprocal kinetic/symplectic action can make
that link frame parallel-transport the four retained components. This tests
the mechanism before fitting any coefficient.

## Research sources

- J. M. Speight, *A discrete phi^4 system without Peierls-Nabarro barrier*:
  https://arxiv.org/abs/patt-sol/9703005
- S. V. Dmitriev et al., *Discrete Nonlinear Schrodinger Equations Free of the
  Peierls-Nabarro Potential*: https://arxiv.org/abs/nlin/0603047
- M. Jenkinson and M. I. Weinstein, *Discrete Solitary Waves in Systems with
  Nonlocal Interactions and the Peierls-Nabarro Barrier*:
  https://arxiv.org/abs/1601.04598
- S. Flach and K. Kladko, *Moving discrete breathers?*:
  https://arxiv.org/abs/cond-mat/9709049
- D. Hennig, N. I. Karachalios, and J. Cuevas-Maraver, *The closeness of the
  Ablowitz-Ladik lattice to the Discrete Nonlinear Schrodinger equation*:
  https://arxiv.org/abs/2102.05332
- J. Cuevas-Maraver, P. G. Kevrekidis, and A. Saxena, *Solitary Waves in a
  Discrete Nonlinear Dirac equation*: https://arxiv.org/abs/1408.4171
- A. Mielke and S. Reichelt, *Traveling fronts in a reaction-diffusion equation
  with a memory term*: https://arxiv.org/abs/2104.12758
- W. A. Benalcazar et al., *Higher-order topological pumping and its
  observation in photonic lattices*: https://arxiv.org/abs/2006.13242
- A.-S. Walter et al., *Quantisation and its breakdown in a Hubbard-Thouless
  pump*: https://arxiv.org/abs/2204.06561

## Claim boundary

The literature supplies mechanism classes and mathematical tools, not evidence
that LFM realizes them. The only current LFM result is a ranked, falsifiable
research program. No candidate in this document is installed in the action,
and V4-006 remains open.
