# V3.4 Covariant Free-Spinor Closure Plan

## Requested finish line

Starting from the unchanged cumulative V3.3 theorem, close the standard
covariant free-spinor package that is still missing from the LFM-to-Dirac
chain.  V3.4 must prove exact two-state positive- and negative-energy sectors,
their covariant spin sums, arbitrary nonzero-momentum helicity projectors, the
non-forward Gordon identity, and the relation between the free resolvent,
spectral projectors, and pole residues.

V3.4 changes no LFM action, governing equation, C19 geometry, coefficient,
vacuum, readout, or physical scale.  Every new object must be defined from the
cube-derived alpha/beta operators, the V2 normalized frequency embeddings,
the V3.1 spin operators, or the V3.2 spectral projectors.

## Design red-team preflight

### Requested finish line

One cumulative Lean theorem must expose the complete covariant free-spinor
closure while retaining the exact finite-C19 remainder and all prior claim
boundaries.

### Required pass gates

1. The V3.3 cumulative certificate is an explicit dependency.
2. The two branch carriers are the existing normalized positive- and
   negative-frequency embeddings, not inserted textbook spinors.
3. Each fixed-energy branch is proved to have exactly two independent
   amplitudes through an explicit left inverse and two-channel resolution.
4. Positive and negative covariant spin sums are proved as operator identities
   with the repository's normalization convention stated exactly.
5. Arbitrary-momentum helicity uses the already-derived cubical spin/Pauli
   operator and excludes zero momentum.
6. The non-forward Gordon theorem follows from the derived Clifford operators
   and explicit left/right on-shell hypotheses.
7. The free resolvent is decomposed into the existing spectral projectors and
   its algebraic pole numerators agree with those projectors.
8. The cumulative build has no `sorry`, `admit`, new axiom, opaque constant,
   fitted coefficient, measured electron input, or physical-unit anchor.

### Disallowed live mechanisms

- standard Dirac matrices or standard spinors introduced independently of the
  cubical operators;
- a new action term, field, stencil, background, or fitted coefficient;
- an electromagnetic potential, minimal-coupling postulate, or unconditional
  magnetic-moment/`g = 2` claim;
- SI units, MeV calibration, or measured electron mass;
- packet transport, finite-time convergence, localization, or self-binding;
- CAR/Fock-space, occupation, exclusion, or charge-quantization assumptions;
- relabeling the continuum root as an exact finite-C19 first-order operator.

### Planned live mechanisms

- `normalizedPositiveFrequencyEmbedding` and
  `normalizedNegativeFrequencyEmbedding` from the V2 native spectral readout;
- `cubicalStandardBeta`, the three cubical alpha operators, and the
  action-derived Hamiltonian;
- `standardCubicalSpin` and `positivePauliSpatial`;
- V3.2 projectors and V3.3 free resolvent;
- finite-dimensional complex algebra and the already-proved mass-shell
  identities.

### Claim boundary

Passing V3.4 proves a covariant free linearized spin-one-half package derived
from the same LFM chain.  It does not prove a nonlinear localized electron,
electromagnetic emergence, physical charge, an unconditional `g = 2`, packet
transport, an absolute mass calibration, quantization, or an exact
finite-spacing Dirac root.

## Seven meaningful handoffs

1. `V34CovariantSpinors`: derive the covariant gamma/slash/adjoint language and
   the two positive plus two negative branch carriers from the existing
   normalized embeddings.
2. `V34SpinorSectorBasis`: prove explicit branch left inverses, exact
   two-channel injectivity, branch resolution, and positive/negative
   orthogonality.
3. `V34CovariantSpinSums`: prove the positive and antiparticle covariant spin
   sums, including the exact normalization factor.
4. `V34MomentumHelicity`: construct arbitrary nonzero-momentum helicity
   projectors and prove their half-valued eigenbranches.
5. `V34NonforwardGordon`: prove the full unequal-momentum Gordon identity from
   the derived Clifford algebra and on-shell equations.
6. `V34PropagatorResidues`: decompose the free resolvent into V3.2 projectors
   and identify the algebraic positive/negative pole numerators.
7. `V34CompleteCovariantFreeSpinor`: assemble the cumulative theorem and the
   explicit scope boundary.  Framework locking and scope auditing are verifier
   responsibilities rather than additional wrapper stages.

## Consolidation policy

Earlier public modules remain available for compatibility and provenance.
Documentation must distinguish scientific handoffs from release wrappers.
V3.4 does not add separate framework-lock, scope-audit, or packet-benchmark
modules whose only purpose is to repackage an existing certificate.

The modules follow the mathematical dependency graph rather than stage-number
order. Independent helicity, Gordon, and propagator branches import their
actual prerequisites directly, and the cumulative module assembles them
explicitly. Stage certificates must expose their strongest advertised result;
an unbundled theorem does not by itself satisfy a cumulative release claim.

## Verification and hostile review

After all seven modules compile, `scripts/verify_v34_research.py` must build
the cumulative facade, rerun the V3.2 and V3.3 verifiers, scan all V3.4 sources
for forbidden proof mechanisms, and inspect the cumulative theorem's axioms.
Two hostile red-team reviews follow.  Critical and high findings block the
release; straightforward medium and low findings are repaired before the
second review.
