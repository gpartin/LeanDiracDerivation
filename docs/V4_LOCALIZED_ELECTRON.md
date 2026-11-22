# V4 localized-electron gate ledger

V4 keeps the V40 two-field C19 action and every V2/V3 theorem unchanged. Its
target is stronger: one freely evolving localized state must realize the rest
mass, four-component readout, spin, translation, density, and current already
derived for the free linear sector.

| Gate | Required result | Status |
|---|---|---|
| V4-001 | Exact unrestricted localized rest orbit on all periodic boxes `L >= 3` | **PASS** |
| V4-002 | Spectral and nonlinear stability of that same orbit | **OPEN** |
| V4-003 | Lossless four-component readout of the same localized phase space | **SCOPED FIXED-CENTER OBSTRUCTION; NONCOMPACT OPTION OPEN** |
| V4-004 | Operational spin-one-half action on the localized carrier | **OPEN** |
| V4-005 | Native translation tangent and nonzero signed bond current | **SCOPED FIXED-CENTER OBSTRUCTION; TRAVELING FAMILY OPEN** |
| V4-006 | Freely solved multi-cell traveling/relative-periodic branch | **OPEN** |
| V4-007 | Same-state agreement with Dirac mass, transport, density, and current | **OPEN** |
| V4-008 | Determine whether localized dynamics selects `B`, remains uniform in it, or rules out the incumbent value | **OPEN** |
| V4-009 | Determine the dynamical role of both cubical four-component copies | **OPEN** |
| V4-010 | Cumulative localized Dirac-electron theorem and clean release audit | **OPEN** |

## V4-001 result

`LeanDirac.V4.exact_localized_rest_seed` proves, from the native reciprocal
action, an exact two-layer recurrence for the eight-site `Psi` carrier and its
live `chi` halo on every periodic C19 box of side at least three. It also proves
zero local imbalance and nontrivial phase advance in both fields. The theorem
uses no holding force, background reset, projection, or Dirac matrix.

The result is symbolic in `B` and holds for `B > 1` under the stated timestep
condition. It therefore does not select `B = 1197`.

## First unresolved obligation

V4-002 is first: exact closure is not stability. The full Floquet/tangent
operator must include carrier, halo, exterior, both committed layers, and both
fields. A stable rest orbit then becomes the input to the same-carrier readout
and motion gates. Translation cannot be claimed from the independent V3
plane-wave theorem.

## Fixed-center tangent classification

The exact symbolic controls have also been rerun. The isolated localized orbit
does not commute with unit translation (the canonical rational witness is
`43309086609 / 17378239864`), so its tangent is a defect operator rather than a
Bloch family. The exactly closed compact carrier block has momentum-derivative
ranks `(0,0,0)`. Enlarging to the natural four-dimensional eigenvalue-27 corner
space does not fix this: its axis response ranks are `(2,2,2)`, it has a common
transport-null direction, its cross anticommutators are nonzero, and its first
response requires exterior channels.

This retires only a fixed-center compact Bloch interpretation. It does not
exclude a noncompact joint field/scattering bundle or a family of translated
relative-periodic orbits. Those are the remaining admissible routes.

## Current movement RCA

**Issue:** No freely moving branch of the exact compact carrier has been
proved.

**Why it matters:** A localized Dirac electron requires the same state to
carry identity and charge across cube boundaries while preserving its internal
four-component structure.

**Symptoms:** Existing native trials move only a small fraction of a site and
complete no verified carrier handoff. Directly copying the rest seed into the
next cell creates conflicting shared halos rather than a translated solution.

**Expected symptoms:** A passing branch has signed bond current, preserves the
carrier under a common internal-frame alignment, and completes repeated
translations in both directions while matching the independently evolved
discrete Dirac comparator.

**Root cause:** The compact exact rest preparation has no proved narrow signed
momentum direction. Its exterior `chi` geometry cancels the coupling required
for the fixed-center orbit, while adjacent copied preparations overlap through
their halos. This identifies a preparation/operator obstruction; it is not a
universal no-go for the action.

**Corrective action:** First derive the full localized tangent and Floquet
operator and test whether it contains a translation mode. If it does, continue
that mode into a jointly solved multi-cell `Psi`--`chi` relative-periodic
orbit. Change the action only if an exact response-rank calculation proves the
required native translation channel absent.
