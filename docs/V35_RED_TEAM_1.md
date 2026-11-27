# V3.5 Hostile Review 1

## Verdict before repairs

The eight-stage plan was scientifically appropriate for a theorem-only
extension of V3.4, and it changed no action, governing equation, field,
stencil, coefficient, vacuum, readout, or physical scale. The first hostile
review nevertheless found three high-severity completeness/convention issues,
two medium proof-audit issues, and one low engineering issue.

## Findings and repairs

### HIGH-1: generated transformations exposed only one inverse direction

The initial generated-word certificate proved that applying an inverse word
after a word returns the original spinor. It did not package the opposite
composition, despite describing exact inverse laws.

Repair: Stage 3 now proves that inversion is involutive and proves both word
compositions equal the identity. The cumulative certificate includes both
directions.

### HIGH-2: arbitrary-word covariance did not explicitly close solutions

Stage 4 proved the exact residual intertwining law for every generated word,
so preservation of a zero residual followed mathematically. The advertised
solution-space closure was not itself a named theorem or certificate field.

Repair: `v35_lorentz_word_preserves_solutions` now states and proves the
arbitrary-word solution theorem directly, using a separate proof that every
generated transformation maps the zero spinor to zero.

### HIGH-3: standard coordinates were initially identified with generator axes

The inherited cubical generators use an oriented axis convention: generator
axis 0 is +z, axis 1 is -y, and axis 2 is -x. The first finite-covariance and
helicity attempts treated those axes as ordinary x, y, z. Lean rejected the
resulting identities.

Repair: the momentum transformations and helicity comparison now use the
orientation forced by the inherited generators. The corrected finite boost,
finite rotation, arbitrary-word covariance, and helicity identities compile.

### MEDIUM-1: helicity normalization admitted a nonphysical scale

The first V3.5 bridge required only a nonzero radius. Algebraically that was
enough to cancel a denominator, but V3.4 defines physical helicity with the
positive momentum norm.

Repair: the public bridge and certificate now require a positive radius whose
square is the momentum norm squared.

### MEDIUM-2: channel counting introduced a native-decider axiom

The first sixteen-channel count used `native_decide`, adding its generated
native-decider axiom to the cumulative audit.

Repair: the count now uses kernel-checked `decide`. The cumulative theorem
depends only on `propext`, `Classical.choice`, and `Quot.sound`.

### LOW-1: the Fierz proof was one oversized case expansion

The first completeness proof exhausted Lean's default heartbeat budget and
was unsuitable for CI review.

Repair: the same 256 component equalities are partitioned into sixteen private
verification blocks and recombined into one public theorem. The statement and
coefficients were not weakened.

## Post-repair boundary

The chain proves free momentum-space equation/solution covariance, bilinear
Clifford/Fierz closure, negative-branch residue closure, and the moving spin
invariant. It does not prove packet dynamics, a nonlinear v40.1 realization,
an interacting or localized electron, electromagnetism, `g = 2`,
quantization/statistics, or an MeV scale.

Critical findings: none.

High findings: repaired; second hostile review and cumulative verification
required.
