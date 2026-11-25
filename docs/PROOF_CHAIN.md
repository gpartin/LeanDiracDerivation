# V2 proof chain

Only this chain is active on `main`.

| Gate | Derived result | Status |
|---|---|---|
| V2-001 | Native `Psi, chi : C^3`, constants, and C19 geometry | PASS |
| V2-002 | Positive reciprocal two-field action | PASS |
| V2-003 | Both field variations from that action | PASS |
| V2-004 | Centered/Verlet native update and stability identities | PASS |
| V2-005 | Exact uniform zero-energy native vacuum; no maintained prepared background | PASS |
| V2-006 | Full finite-lattice Hessian, invariant transverse sector, and C19 transport | PASS |
| V2-007 | Bijective four-complex-component phase-space readout | PASS |
| V2-008 | Exact first-order generator intertwining | PASS |
| V2-009 | Cubical incidence Clifford root, exhaustive direct sum of exactly two rank-four copies, and C19 remainder | PASS |
| V2-010 | Proved change of coordinates to standard alpha/beta form | PASS |
| V2-011 | Positive/negative frequency grading and `m = chi0` | PASS |
| V2-012 | Exact centered/Verlet dispersion and continuum time limit | PASS |
| V2-013 | Position-space equation, density, bond current, and controlled remainders | PASS |
| V2-014 | Compiled adversarial axiom/dependency audit | PASS |
| V2-015 | Cumulative action-to-free-Dirac theorem | PASS |

V3.1 adds a downstream, action-preserving consequence
chain.  Its detailed falsification gates are in
[`LORENTZ_SPIN_CHAIN.md`](LORENTZ_SPIN_CHAIN.md).

| Extension gate | Derived result | Local status |
|---|---|---|
| LS-001 | Oriented spatial-plane bivectors from the native cubical gamma operators | PASS |
| LS-002 | Unique nonzero rotation normalization selected by algebraic closure and, within the explicit local plane-bivector class, by native transport covariance | PASS |
| LS-003 | Spin commutators, projections `+/-1/2`, and Casimir `3/4` | PASS |
| LS-004 | The same rotation representation on both exhaustive cubical copies | PASS |
| LS-005 | Density-preserving finite rotation, `2*pi` sign reversal, and `4*pi` return | PASS |
| LS-006 | Rotation/boost Lorentz Lie algebra | PASS |
| LS-007 | Mass invariance and infinitesimal low-momentum Dirac-operator covariance | PASS |
| LS-008 | All-axis Pauli--Lubanski four-vector covariance, orthogonality, and rest Casimir | PASS |
| LS-009 | Cumulative Lorentz/spin certificate with standard Lean foundations only | PASS |

The dependency route is strictly top-down:

```text
native action
  -> action variation
  -> exact native vacuum
  -> finite-lattice Hessian
  -> closed transverse C19 phase space
  -> cubical Clifford root
  -> lossless four-component readout
  -> mass + time + three transport generators
  -> free Dirac equation + controlled C19/time remainders
  -> density/current bridge
```

No historical V1 module is imported by this route.
