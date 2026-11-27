# V3.4 Dependency Matrix

| V3.4 result | Existing derived input | New assumption? |
|---|---|---|
| Covariant gamma/slash operators | Cube-derived alpha/beta operators | No |
| Covariant branch spinors | Native normalized frequency embeddings | No |
| Two-state branch coordinates | Existing two-channel native amplitude | No |
| Covariant spin sums | Derived branch spinors and Dirac adjoint | Only mass-shell/positivity domain hypotheses |
| Arbitrary-momentum helicity | `positivePauliSpatial` from the cubical lower block | Only positive momentum norm and radius identity |
| Non-forward Gordon identity | Derived gamma commutators and on-shell equations | Only explicit left/right on-shell hypotheses |
| Resolvent decomposition | V3.3 resolvent and V3.2 projectors | Only nonzero energy and off-pole hypotheses |
| Pole/projector agreement | V3.2 spectral projectors | No fitted input |
| Physical electron identity | Not proved by V3.4 | Not asserted |
| Electromagnetic `g = 2` | No interacting EM sector in this chain | Not asserted |
| MeV calibration | No absolute dimensionful anchor | Not asserted |
| Transport/localization | Separate work | Not asserted |

All hypotheses are algebraic domain conditions. No measured electron value,
physical-unit anchor, standard spinor, or independent Dirac matrix is used.

## Module dependency shape

- `V34CovariantSpinors` is the common covariant base.
- `V34SpinorSectorBasis` and `V34MomentumHelicity` depend on that base.
- `V34CovariantSpinSums` depends on the sector basis.
- `V34NonforwardGordon` depends directly on the covariant base.
- `V34PropagatorResidues` depends on the covariant spin sums and the prior
  V3.3 propagator.
- `V34CompleteCovariantFreeSpinor` explicitly assembles every branch.

Imports do not impose a scientific ordering between independent branches.
