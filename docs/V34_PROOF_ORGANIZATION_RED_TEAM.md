# V3.4 Proof Organization Red Team and Repair

## Scope

This audit reviewed redundancy, module boundaries, theorem naming,
certificate coverage, import direction, and verifier enforcement. It did not
change the LFM action, equations, physical assumptions, normalization, or
claim boundary.

## Findings

### HIGH-1: headline theorems were outside the cumulative certificates

Stage 2 proved unique two-plus-two coordinates and the two branch Gram
relations, but its certificate exposed only injectivity, span, and
cross-branch orthogonality. Stage 5 proved the native-to-native non-forward
Gordon theorem, but its certificate exposed only the generic theorem and a
left-shell bridge.

Repair: `V34SpinorSectorBasisCertificate` now contains both branch Gram
relations and the unique two-plus-two decomposition. The weaker branch-read
and injectivity layer was removed. `V34NonforwardGordonCertificate` now
contains the native-to-native Gordon theorem directly.

### MEDIUM-1: independent stages were wired as one serial import chain

Helicity imported spin sums, Gordon imported helicity, and propagator
residues imported Gordon even though those results are mathematically
independent.

Repair: helicity and Gordon now import the common covariant-spinor base.
Propagator residues imports spin sums and the prior propagator directly. The
cumulative module explicitly imports every independent V3.4 branch.

### MEDIUM-2: covariant gamma operators were recreated

The four-index V3.4 gamma wrapper repeated the underlying beta/alpha formulas
instead of reusing the established covariant definitions.

Repair: `v34Gamma` is now a four-index wrapper around `cubicalGammaTime` and
`cubicalGammaSpace` from `CovariantFreeDirac`.

### MEDIUM-3: spin-sum normalization algebra was duplicated

The positive and antiparticle spin-sum proofs independently reconstructed the
same scale-square, scale-ratio, lower-block, and cross-term facts.

Repair: the common facts now live in one private
`V34SpinSumNormalizationFacts` bundle proved once and consumed by both sums.

### MEDIUM-4: energy-normalized states were called orthonormal

The branch Gram theorems return `energy` on the diagonal, not one.

Repair: the theorem names now say `energy_orthogonal`, and the certificate
fields are named `positiveGram` and `negativeGram`.

### LOW-1: superseded and unused helpers remained public

The one-sided native-incoming Gordon wrapper, negative-branch forwarding
lemma, two-channel resolution helper, branch readers, and their injectivity
wrappers were unused after the stronger results were available.

Repair: these declarations were removed. The stronger public results remain.

### LOW-2: prior-version certificates were repeated

V3.2 was packaged directly even though the V3.3 certificate already contains
the V3.2 framework chain, and Stage 1 repeated the V3.3 certificate again.

Repair: the cumulative V3.4 certificate retains one explicit V3.3 dependency;
V3.2 remains available through it.

### LOW-3: pole-numerator formulas duplicated existing projectors

The named positive and negative pole numerators repeated the V3.2 projector
definitions byte-for-byte.

Repair: the names are now lightweight aliases of the projectors. The
nontrivial spectral-resolvent decomposition and spin-sum/projector theorem
remain proved independently.

## Verifier hardening

The V3.4 verifier now checks the strengthened theorem inventory, the direct
dependency graph, the absence of the old artificial serial imports, and
`opaque` declarations in addition to `axiom`, `constant`, `sorry`, and
`admit`.

## Result

The seven meaningful handoffs remain. The proof has a clearer dependency
graph, a stronger cumulative certificate, fewer redundant declarations, and
no change to its scientific scope.

Post-repair validation passed:

- the cumulative V3.4 build;
- the strengthened declaration and import-graph verifier checks;
- the inherited V3.2 and V3.3 release verifiers;
- the forbidden-source scan;
- the cumulative axiom audit, with only `propext`, `Classical.choice`, and
  `Quot.sound` reported.

Final organization-audit status: **PASS**, with zero open critical, high,
medium, or low findings from this audit.
