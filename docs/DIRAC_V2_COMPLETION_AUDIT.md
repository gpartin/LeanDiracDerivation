# DIRAC V2 completion audit

Verified: 2026-11-21

Production candidate: `feature/v2-free-dirac-production-port`

Hash-sealed proof-source commit:
`f10c9a144e8027e0033f7025a154c2b4ecbe9bfa`

## Goal audited

Complete the free LFM-to-Dirac derivation from the native two-field C19 action,
including the operator, invariant four-component sector, mass, first-order
time map, three spatial transport terms, density, and current, without assumed
Dirac matrices, a prepared Dirac-like background, a circular certificate, or
a simulation premise.

## Requirement ledger

| Requirement | Verdict | Authoritative evidence |
|---|---|---|
| Native complex-Psi/complex-chi C19 action | PASS | `NativeAction.lean` defines the local and exterior-area potential and native Lagrangian. |
| Reciprocal native equations from the action | PASS | `NativeVariation.lean` proves the local, bond, and complete potential derivatives. |
| Full finite-periodic tangent operator | PASS | `FinitePeriodicTangent.lean` derives the Hessian force from the full action. |
| Closed four-complex-component free sector | PASS | `FinitePeriodicSector.lean` proves injectivity, generator intertwining, and zero linear longitudinal/chi leakage. |
| Clifford root without assumed Dirac matrices | PASS | `NativeC19CliffordRoot.lean` derives the signed Kähler--Dirac incidence root first; conventional alpha/beta matrices are downstream coordinates proved to intertwine with it. |
| Mass | PASS | The native rest spectrum supplies `mass^2 = chi0^2`; nonnegativity uniquely gives `mass = chi0`. |
| First-order time evolution | PASS | The position/velocity map is bijective; centered and Verlet updates pass through the same readout with explicit vanishing time defects. |
| Three spatial transport directions | PASS | C19 isotropy and all three cubical incidence generators produce the Hamiltonian, covariant, and position-space free Dirac equations with the explicit finite-C19 remainder. |
| Density and current | PASS WITH EXACT SCOPE | Same-state positive modes have the native-density/Dirac-density bridge and native C19 bond flux equals Dirac current plus the proved remainder. Finite packets have exact integrated charge and integrated current with the summed remainder. Pointwise multimode equality is formally disproved and is not claimed. |
| Both Clifford copies retained | PASS | Both exact rank-four flavor embeddings remain in the theorem; neither is projected away. |
| No circular certificate | PASS | `complete_first_principles_free_dirac` assembles concrete preceding theorems. The compiled environment audit rejects custom axioms and `sorryAx`. |
| No simulation premise | PASS | Every premise is symbolic action variation, finite Fourier algebra, operator intertwining, or a proved limit/error bound. |
| No imported or maintained Dirac-like background | PASS | V2 imports no prepared-background or numbered DIRAC module. No period-two texture, corner ray, projection, reset, or maintained field is a theorem hypothesis. |
| No selected vacuum representative whatsoever | OPEN OUTSIDE FREE THEOREM | The action proves the uniform axial state is an exact critical/global-minimum solution, but also admits every sitewise chi orientation of the same magnitude. The theorem selects one uniform representative; the action does not dynamically align arbitrary orientations. |
| Nonlinear localized moving electron | SEPARATE OPEN GATE | The theorem is the free linear tangent sector. It does not identify or translate a nonlinear eight-site electron. |

## Non-circular chain

```text
native two-field C19 action
  -> proved first variation
  -> exact native uniform vacuum solution/global minimum
  -> full finite-lattice action Hessian
  -> two invariant transverse complex C19 Klein--Gordon channels
  -> complete four-complex-dimensional position/velocity phase space
  -> oriented cubical incidence Clifford root
  -> bijective first-order readout
  -> mass + time + three transport terms
  -> Hamiltonian, covariant, and position-space free Dirac equations
  -> native continuity and controlled density/current bridges
```

The mass shell used by the spectral readout is downstream of the native
Hessian dispersion. It is not an independent Dirac premise.

## Verification evidence

- Clean V2 build: `3775/3775` jobs passed.
- Historical public-chain regression: `3630/3630` jobs passed.
- Hardened artifact verifier:
  `PASS_WITH_SCOPED_OPEN_GATES_AND_COMPILED_AXIOM_AUDIT`.
- Axioms: only `propext`, `Classical.choice`, and `Quot.sound`.
- Hash seal: 30 proof/foundation sources and six audit/toolchain files.
- Historical numbered source delta: no `LeanDirac/Dirac*.lean` file changed.

## Completion boundary

The defensible completed statement is:

> Starting from the native two-field C19 action and selecting one exact
> uniform zero-energy vacuum representative, oriented cubical incidence gives
> a lossless four-component factorization of the complete transverse linear
> phase space into the free Dirac equation, with both flavor copies retained
> and explicit finite-lattice, timestep, and current remainders.

This closes the mathematical free-Dirac derivation. Dynamic vacuum alignment,
unique selection of a Clifford chart, and a nonlinear localized physical
electron are not silently inherited by that result.
