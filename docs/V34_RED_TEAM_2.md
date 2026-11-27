# V3.4 Hostile Review 2

## Review method

This second review restarted from the requested finish line and checked the
Lean statements rather than relying on the first review or on successful
compilation. It audited theorem hypotheses, normalization and sign
conventions, certificate coverage, dependency direction, source hygiene, and
the public claim boundary.

## Finding and repair

### MEDIUM-1: a signed radius could reverse helicity labels

The first repaired Stage 4 statements required `radius^2` to equal the squared
momentum norm and required `radius != 0`. Algebraically, that admitted both
roots. A negative root preserves the projector algebra but reverses which
projector is called positive helicity, contradicting the documentation's
description of `radius` as the momentum norm.

Repair: all public Stage 4 theorems and the cumulative helicity certificate
now require `0 < radius`. The private algebraic helper retains only the
minimal nonzero denominator premise. Thus the public interface fixes the
physical branch while the proof implementation remains appropriately general.

## Rechecked gates

- The V3.3 cumulative certificate is a direct dependency.
- Covariant spinors are constructed from the existing native embeddings.
- Each energy branch has an explicit two-channel inverse and resolution.
- Both covariant spin sums state their normalization explicitly.
- Helicity covers every nonzero momentum with a positive norm, complete
  projectors, idempotence, and the corresponding half-valued eigenbranches.
- The native non-forward Gordon result derives both left- and right-shell
  conditions from independently selected native spinors.
- Resolvent decomposition and pole numerators reuse the V3.2 projectors.
- No LFM action, equation, coefficient, field, physical scale, or measured
  electron property is added.
- The claims remain limited to the covariant free linearized spinor package.

## Final severity table

| Severity | Open findings |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |

## Final verification

The post-repair `scripts/verify_v34_research.py` run passed. It rebuilt the
V3.4 cumulative certificate, reran the V3.2 and V3.3 verifiers, completed the
forbidden-source scan, and confirmed that the cumulative theorem depends only
on Lean's standard `propext`, `Classical.choice`, and `Quot.sound` axioms.

Final release status: **PASS**.
