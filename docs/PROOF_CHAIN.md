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
| V2-009 | Cubical incidence Clifford root, both rank-four copies, and C19 remainder | PASS |
| V2-010 | Proved change of coordinates to standard alpha/beta form | PASS |
| V2-011 | Positive/negative frequency grading and `m = chi0` | PASS |
| V2-012 | Exact centered/Verlet dispersion and continuum time limit | PASS |
| V2-013 | Position-space equation, density, bond current, and controlled remainders | PASS |
| V2-014 | Compiled adversarial axiom/dependency audit | PASS |
| V2-015 | Cumulative action-to-free-Dirac theorem | PASS |

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
