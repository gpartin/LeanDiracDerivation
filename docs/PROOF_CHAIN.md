# Cumulative proof chain through V3.6

This is the authoritative top-down map of the active production import
`LeanDirac`. Version labels identify meaningful cumulative handoffs; they are
not independent proof lines.

## Core action-to-Dirac chain (V2)

| Gate | Derived result | Status |
|---|---|---|
| V2-001 | Native `Psi, chi : C^3`, constants, C19 geometry, and kinetic law | PASS |
| V2-002 | Positive reciprocal two-field action | PASS |
| V2-003 | Both field variations from that one action | PASS |
| V2-004 | Centered/Verlet native update and stability identities | PASS |
| V2-005 | Exact uniform zero-energy native vacuum | PASS |
| V2-006 | Full finite-lattice tangent and invariant transverse sector | PASS |
| V2-007 | Lossless four-complex-component phase-space readout | PASS |
| V2-008 | Exact first-order generator intertwining | PASS |
| V2-009 | Cubical Clifford root and exactly two exhaustive rank-four copies | PASS |
| V2-010 | Change of coordinates to standard alpha/beta form | PASS |
| V2-011 | Frequency grading and native mass `m = chi0` | PASS |
| V2-012 | Centered-time dispersion and continuum time limit | PASS |
| V2-013 | Position-space equation, density/current bridge, and remainders | PASS |
| V2-014 | Compiled adversarial axiom/dependency audit | PASS |
| V2-015 | Direct finite-periodic action-to-free-Dirac theorem | PASS |

## Cumulative extensions

| Release | Meaningful handoff | Cumulative theorem | Status |
|---|---|---|---|
| V3.1 | Cubical Lorentz algebra, spin-one-half spectrum, finite rotations, double-cover sign, and Pauli--Lubanski rest invariant | `lorentz_spin_half_from_native_action` | PASS |
| V3.2 | Global C19 spectrum, projectors, artifact/doubling audit, discrete symmetries, nonrelativistic limit, transport references, and native scale identifiability | `v32_complete_free_fermion` | PASS |
| V3.3 | Free-spinor basis, projected branches, spin sums, bilinears, axis helicity, forward Gordon identity, and free resolvent | `v33_complete_free_spinor` | PASS |
| V3.4 | Native-derived covariant branch spinors, covariant spin sums, arbitrary-momentum helicity, non-forward Gordon identity, and propagator residues | `v34_complete_covariant_free_spinor` | PASS |
| V3.5 | Antiparticle residue closure, finite generated Lorentz covariance, complete bilinear/Fierz package, and moving Pauli--Lubanski invariant | `v35_complete_covariant_observables` | PASS |
| V3.6 | Upstream cube vector/bivector isometry, common-metric action normalization, direct live-action specialization, controls, and radial consequences | `v36_complete_geometrically_normalized_action` | PASS |

## V3.6 action-to-result route

```text
native complex-C3 fields
  -> upstream cube vector/bivector inverse and norm isometry
  -> common Euclidean cubical Clifford kinetic metric
  -> chi kinetic normalization B = 1 within that metric class
  -> live V3.6 reciprocal C19 action and Hamiltonian
  -> action variation and exact native vacuum
  -> finite-periodic action tangent
  -> invariant transverse phase space
  -> cubical Clifford factorization and four-component readout
  -> free Dirac equation with explicit C19/time remainders
  -> V3.1--V3.5 free representation and observable consequences
```

The direct theorem
`v36_live_research_action_mode_to_free_dirac` uses a tangent quotient defined
from `v36ResearchC19Lagrangian`; it is not an inference made by placing a new
action certificate beside an older Dirac certificate.

## Claim boundary

The chain establishes a free linearized tangent result and downstream free
representation/observable consequences. It does not establish a nonlinear
localized electron, indivisible occupation, electromagnetic charge, exchange
statistics, magnetic response, SI/MeV calibration, nonlinear mobility,
Einstein closure, or repository-wide promotion of this research action to the
canonical V40.1 framework.
