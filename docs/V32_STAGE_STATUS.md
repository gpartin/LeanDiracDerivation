# V3.2 Stage Status

Base commit: `61a4798b8b3ebfcffd24065079dc4db9d7b43f75`

No stage receives `PASS` until its Lean module, transitive dependency build,
axiom audit, and stage-specific hostile checks pass.

| Stage | Name | Initial status | Depends on |
|---|---|---|---|
| V32-001 | Framework and claim lock | PASS | V3.1 certificates |
| V32-002 | Global Hamiltonian spectrum | PASS_NARROWED | V32-001 |
| V32-003 | Projectors, completeness, and unitarity | PASS | V32-002 |
| V32-004 | Brillouin-zone and doubler audit | PASS | V32-002 |
| V32-005 | Two-copy equivalence and mixing | PASS | V32-003, V32-004 |
| V32-006 | Irreducible particle sector | PASS | V32-003, V32-005, V3.1 spin |
| V32-007 | Finite continuum Lorentz covariance | PASS_NARROWED | V32-006 |
| V32-008 | C, P, T, and CPT | PASS_NARROWED | V32-003, V32-005, V32-007 |
| V32-009 | Free nonrelativistic limit | PASS | V32-003 |
| V32-010 | Continuum remainder convergence | PASS_NARROWED | V32-003, V32-009 |
| V32-011 | Lattice artifacts and anisotropy | PASS | V32-004, V32-010 |
| V32-012 | Rest-mass certificate and scale identifiability | PASS | V32-002, V32-009 |
| V32-013 | Robustness and uniqueness | PASS | V32-005 through V32-012 |
| V32-014 | Transport observables | PASS | V32-003, V32-010, V32-011 |
| V32-015 | Centered-step transport spectrum | PASS_NARROWED | V32-002, V32-014 |
| V32-016 | Finite packet theorem contract | PASS_NARROWED | V32-003, V32-014, V32-015 |
| V32-017 | Cumulative certificate and release gate | PASS | V32-001 through V32-016 |

Stages marked `PASS_NARROWED` use an explicit owner-authorized or
obstruction-required claim boundary. In particular, V3.2 proves finite boost
group laws plus infinitesimal covariance, not a new full finite-observable
Lorentz monograph; it proves controlled low-momentum/current and time-step
convergence, not a nonexistent exact finite-C19 local Dirac projector; and it
exports arbitrary finite-packet integral theorems rather than treating a
numerical fixture table as a proof.

The cumulative build, forbidden-token scan, framework-lock diff, required
theorem audit, claim-status audit, and semantic axiom audit all pass through
`scripts/verify_v32_research.py`. The verifier reports
`V32_RELEASE_CANDIDATE_PASS`.

## Decisive Stage-2 result

Stage 2 now proves the exact finite-C19 energy branches are real,
nonnegative/gapped for nonzero mass, and satisfy the unchanged action's exact
dispersion. It also proves a no-go theorem:

`v32_no_global_exact_local_oriented_root`

The V3.1 local cubical first-order root equals an exact square root of the
finite-C19 Hessian if and only if the spatial remainder vanishes. At the
explicit Brillouin-zone momentum `(pi, 0, 0)`, Lean proves that remainder is
`48 - 12*pi^2`, hence nonzero. Therefore the original Stage-2 demand for the
existing local Dirac Hamiltonian to be globally exact is incompatible with
the unchanged V3.1 action. This is a proved obstruction, not an unfinished
proof search.

Stage 3 now fully proves the continuum-root projector finish line: branch
invariance/eigenvalues, completeness, idempotence, orthogonality, positive
density, and norm-preserving exact branch evolution. It cannot be promoted as
an exact finite-C19 projector theorem because Stage 2 proves that the required
global local first-order operator is absent.

The owner subsequently authorized Route 1: exact continuum closure plus the
explicit controlled finite-C19 remainder. Under that claim boundary Stage 2
is `PASS_NARROWED`, and the fully strengthened continuum projector Stage 3 is
`PASS`. The broader repair search is complete within the unchanged state space.
`v32_no_odd_periodic_sum_of_squares_factorization` rules out any finite number
of ordinary real odd periodic Dirac coefficients, not only the existing three.
`v32_exact_c19_symbol_is_local_bond_sum` exhibits the alternative exact local
factorization, but it uses separate complex C19 bond channels and would require
an enlarged first-order state/readout. That is outside the V3.2 framework lock.

## Status vocabulary

- `NOT_STARTED`: no proof attempt has begun.
- `IN_PROGRESS`: the stage is actively being proved.
- `FAILED_RCA_ACTIVE`: the stated finish line currently fails and RCA/fix is
  underway.
- `BLOCKED_UNKNOWN`: repeated RCA has reached an unresolved mathematical or
  authority boundary.
- `BLOCKED_PROVED_INCOMPATIBLE`: Lean proves that the stated finish line
  contradicts an unchanged-action result.
- `CONTINUUM_PASS_RELEASE_BLOCKED`: the continuum theorem passes, but a
  required exact finite-lattice dependency is proved unavailable.
- `PASS`: the full stated finish line and audits pass.
- `PASS_NARROWED`: allowed only after an explicit owner-approved claim change;
  never used silently.
