# V3.5 Hostile Review 2

## Review method

The second review restarted from the requested finish line and inspected the
actual theorem statements, definitions, dependency graph, normalization
sources, certificate fields, source hygiene, and public wording. It did not
treat successful compilation as sufficient evidence.

## Findings and repairs

### HIGH-1: Fierz normalizations appeared directly in the target

The full Fierz identity was machine-checked, but its `1/4` spinor coefficient
and `1/2` ordered-tensor coefficient were initially written directly in the
target statement. That established compatibility but left an avoidable
hardcoding criticism.

Repair: Stage 6 now defines the first normalization as the inverse of the four
derived spinor channels and the second as the inverse of the two orderings of
an antisymmetric pair. Lean proves that these definitions evaluate to `1/4`
and `1/2`, and the Fierz theorem uses only those derived definitions.

### MEDIUM-1: bilinear wording exceeded the proved transformation interface

V3.5 proves finite generated covariance of the free Dirac equation and
solution space. Its bilinear branch defines all five standard classes and
proves Clifford/Fierz closure, but it does not separately prove every
bilinear's finite transformation law.

Repair: the README, claims, plan, and release notes now make this separation
explicit and no longer describe the five definitions themselves as a proved
finite-covariant family.

### LOW-1: two orientation conversion helpers were unused

Two temporary momentum conversion definitions documented the axis convention
but were absent from every theorem.

Repair: the unused definitions were removed, and the convention is documented
at the live finite-energy transformation and helicity theorem.

### LOW-2: the inherited release seal did not recognize V3.5 extensions

The clean-checkout release verifier initially rejected the new V3.5 modules
and documents because its cumulative-extension allowlist ended at V3.4. It
did not report any modification to a sealed V3.1 proof source.

Repair: the verifier now treats only `V35` Lean modules and `V35` documents as
permitted cumulative extensions. The underlying V3.1 seal and its namespace
migration checks remain unchanged.

## Rechecked gates

- V3.4 is an explicit dependency and no earlier theorem was weakened.
- Both antiparticle residue identities retain their shell and normalization
  hypotheses.
- Finite transformation words contain only inherited cubical rotors and
  boosts and have two-sided inverse laws.
- The residual and zero-residual solution space are covariant under every
  generated word.
- Gamma-five is constructed from the inherited gamma operators.
- The five standard bilinear classes, sixteen-channel count, derived
  completeness normalizations, and full Fierz identity are machine-checked.
- The arbitrary-momentum Pauli--Lubanski square gives the on-shell
  spin-one-half invariant and agrees with positive-norm V3.4 helicity in the
  inherited cube orientation.
- No action, equation, field, fitted coefficient, measured electron input, or
  physical-unit anchor was added.
- No `sorry`, `admit`, custom axiom, opaque declaration, or native-decider
  axiom enters the cumulative theorem.

## Final severity table

| Severity | Open findings |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

## Final verification

The post-repair `scripts/verify_v35_research.py` run passed. It rebuilt the
V3.5 cumulative theorem, reran the V3.2--V3.4 regression suites, completed the
source audit, and confirmed that the final theorem depends only on
`propext`, `Classical.choice`, and `Quot.sound`.

Final hostile-review status: **PASS**. Commit and push remain release-delivery
steps, not proof gates.
