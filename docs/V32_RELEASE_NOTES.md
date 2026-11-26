# V3.2 Route-1 Free-Fermion Spectral Closure

V3.2 extends the V3.1 Lorentz/spin result without changing the declared
action, governing equations, C19 geometry, vacuum, or readout. The cumulative
Lean theorem is `LeanDirac.V2.v32_complete_free_fermion`.

## What V3.2 adds

- Exact continuum free-Dirac energy branches and spectral projectors.
- The exact finite-C19 scalar spectrum, nonzero-mass gap, and proof that the
  origin is its only zero in the fundamental Brillouin box.
- A two-copy classification, multiplicity commutant, scoped rest-spin
  irreducibility result, and combined two-copy rest-sector certificate.
- Represented free-sector C, P, T, and CPT identities.
- The exact free Schrodinger limit and first relativistic correction.
- Controlled spatial, current, centered-time, and Verlet continuum errors.
- The exact isotropic fourth-order C19 moment and the first directional split
  at sixth order.
- Exact centered-step and finite spectral-packet transport references for the
  separate V4 transport program.
- Native rest mass `19`, particle/antiparticle rest gap `38`, and a proof that
  the dimensionless theory does not determine a MeV conversion by itself.
- A theorem showing that the free transverse tangent result is uniform in the
  arbitrary live-chi kinetic coefficient `B`.

## Route-1 correction

The exact finite-C19 action spectrum is trigonometric, whereas the existing
local four-component cubical root squares to its quadratic continuum symbol.
Lean proves that they are not globally identical at finite spacing. V3.2
therefore makes the strongest unchanged-framework claim supported by the
proof: exact continuum Dirac closure with an explicit controlled finite-C19
remainder, alongside the exact finite-C19 second-order spectrum.

The proof also rules out an exact factorization by any finite sum of ordinary
real odd periodic transport coefficients. It exhibits an exact local bond-sum
factorization, but promoting those separate bond channels to a first-order
state operator would enlarge the readout and is outside V3.2.

Stages 002, 007, 008, 010, 015, and 016 are consequently labeled
`PASS_NARROWED`; their exact boundaries are recorded in
`docs/V32_STAGE_STATUS.md` and `docs/V32_RED_TEAM.md`.

## Rest-mass claim boundary

V3.2 proves a native dimensionless rest mass of `19` and rest gap of `38` for
both exhaustive copies. It proves only the conversion law
`E_rest,physical = 19 * E_native_unit`; it does not choose
`E_native_unit`, derive `0.511 MeV`, or use the measured electron mass as an
input.

## Reproduce

```bash
lake build LeanDirac LeanDiracAudit
python scripts/verify_release.py
python scripts/verify_v32_research.py
```

The V3.2 verifier checks the unchanged-framework boundary, forbidden proof
tokens, required hostile-audit theorems, release documentation/CI exposure,
stage statuses, and the cumulative theorem's semantic axiom report.

The legacy V3.1 verifier was made extension-aware: it still rejects changes to
the sealed V3.1 proof sources and reruns their original semantic build, axiom,
and Pauli--Lubanski regression audits, while permitting only named `V32*`
modules/docs and the public facade/index needed by a later release.

## Claim boundary

This is a free linearized fermion/operator result. It is not a proof of a
nonlinear localized physical electron, electromagnetic coupling, charge,
`g = 2`, an absolute physical mass, or completed finite-packet transport.
