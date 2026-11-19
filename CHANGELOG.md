# Changelog

## DIRAC-023 (unreleased)

- Evaluated the inherited continuum plane-wave phase along the independently
  derived current-transport trajectory and proved its rate is `chi0^2/E`.
- Proved the packet-center clock ratio is the standard relativistic speed
  factor and is reciprocal to the energy ratio, without assuming a Lorentz
  factor or adding a clock rule.
- Proved directional group/phase velocity reciprocity
  `v_group*v_phase=c_eff^2`, after formally proving that the stated phase
  velocity keeps the corresponding plane-wave phase fixed.
- Surfaced the already-proved DIRAC-022 identity that zitter amplitude times
  zitter angular frequency is exactly `c_eff`.
- Added independent momentum-space reconstruction, wrong-transport and
  wrong-rest-frequency controls, a scoped red-team review, and no action,
  field, coefficient, stencil, recurrence, or governing-equation change.

## DIRAC-022 (unreleased)

- Derived the free-continuum zitterbewegung velocity expectation from the
  existing Hamiltonian momentum-response operator and the DIRAC-021 rest
  branches, with no native action or equation change.
- Proved the normalized coherent state, exact native frequency `38`, position
  amplitude `sqrt(12)/38`, and peak-to-peak reduced Compton length
  `sqrt(12)/19`.
- Added pure-branch, same-sign, incoherent-mixture and wrong-mass controls,
  independent reconstruction, full-chain audits and deterministic packaging.
- Kept localized native-electron trembling, SI calibration and electromagnetic
  response explicitly outside the claim.

## DIRAC-021

- Proved that the zero-momentum frequency of the previously derived
  four-component continuum Hamiltonian equals the action-selected native rest
  coefficient `chi0=19`.
- Constructed explicit positive- and negative-energy rest states and proved
  their exact phase evolutions satisfy the corresponding first-order Dirac
  time equations.
- Derived the opposite-branch relative phase as a quotient and proved its
  angular frequency is `2 chi0=38`, with exact native periods `2π/19` and
  `π/19`.
- Distinguished the inverse-mass scale `1/19` from the native-coordinate
  reduced Compton length `sqrt(12)/19`.
- Added an independent reconstruction with wrong-mass and same-sign controls,
  while explicitly excluding an observable pure-state clock, SI calibration,
  zitterbewegung, electromagnetic response, and a physical-electron claim.
- Added no native field, action term, force, projector, clock register, fitted
  coefficient, stencil change, or governing-equation change.

## DIRAC-020 (unreleased)

- Constructed an explicit normalized finite Fourier frame for every finite
  abelian periodic lattice and proved orthonormality from character
  orthogonality.
- Proved on a periodic cubic lattice that each Fourier character's axis-step
  phase is exactly the phase of the momentum stored in its corresponding
  positive-frequency mode; arbitrary basis/momentum pairings are excluded
  from the transport certificate.
- Synthesized the DIRAC-019 positive-frequency modes into one coherent
  four-component spatial field with exact Parseval density preservation.
- Proved its integrated spatial current equals the existing density-weighted
  transport velocity at every stable update and that nonzero derived current
  gives nonzero spatial transport.
- Proved an explicit pointwise finite-update recurrence-versus-continuum phase
  error bound and independently reconstructed the field movement under
  timestep refinement.
- Added no native field, action term, translation rule, projector, fitted
  coefficient, stencil change, or nonlinear electron claim.

## DIRAC-019 (unreleased)

- Proved exact recurrence-phase evolution and density preservation for every
  integer update count on each stable positive-frequency mode.
- Lifted the result to arbitrary finite spectral families and proved that
  total current is their density-weighted group-velocity moment.
- Proved strict subluminality for massive modes and convergence of the summed
  recurrence generator to the previously derived Dirac generator.
- Added an independent Fourier-packet movement diagnostic plus unstable-step
  and wrong-frequency controls, without adding a translation rule, field,
  action term, stencil change, projection or fitted coefficient.
- Kept coherent real-space Lean formalization, finite-time uniform error, and
  nonlinear localized-electron motion explicitly open for DIRAC-020.

## DIRAC-018 (unreleased)

- Classified the continuum square root inside the explicit real, local,
  first-order, flavor-preserving cubical-incidence coefficient class.
- Proved cubic covariance reduces the three transport coefficients to one,
  exact scalar squaring fixes its magnitude, and the upstream positive
  mass/current orientation fixes the sign.
- Proved the result independently on both exact rank-four copies without
  assuming equal copy coefficients; proved the embeddings exhaust their
  flavor eigenspaces and intersect only at zero.
- Proved the exact C19 scalar transport symbol has only the origin zero in the
  first Brillouin zone.
- Added formal wrong-orientation and broken-cubic-symmetry controls, independent
  reconstruction, adversarial evidence mutations, and explicit nonclaims for
  nonlocal/flavor-mixing roots, exact finite-spacing factorization, universal
  doubling, and a localized electron.
- Added no native field, action term, stencil change, runtime projection, or
  physical-electron assumption.

## Publication-quality repair (unreleased)

- Added a formal all-angle interpreted-Hamiltonian covariance theorem to close
  the gap between the documented continuous rotation claim and the prior
  quarter-turn-only covariance proof.
- Added independent DIRAC-016 and DIRAC-017 matrix reconstructions and
  fail-closed schema validation for both evidence bundles.
- Corrected machine status so DIRAC-017 remains explicitly conditional rather
  than being flattened to a generic `PASS`.
- Corrected the release workflow to package the DIRAC-017 archive, fail when
  artifacts are absent, pin actions by immutable commit, and embed exact Git
  provenance inside the archive.
- Replaced the Mathlib umbrella import with targeted imports, removed every
  recursion-depth override, and reduced heartbeat overrides from nine global
  four-million budgets to the smaller budgets actually required by the
  expensive proofs.
- Separated the historical DIRAC-015 package record from current DIRAC-017
  release readiness and synchronized citation, archive, tracker, claims, and
  reproducibility metadata.
- Removed the unused PyTorch dependency and added a publication-quality audit.

## DIRAC-017 (unreleased)

- Constructed all three rotation generators only from the existing signed
  cube-incidence operators.
- Proved that the actual nontrivial rotation commutator uniquely fixes their
  normalization to `1/2`; no spin matrix or half-angle rule is assumed.
- Proved all three cyclic commutators, self-adjointness for the native inner
  product, component square `1/4`, and Casimir `3/4`.
- Proved the native inner product is positive definite and that both retained
  rank-four copies intertwine exactly with conventional four-component spin.
- Exposed the two-plus/two-minus one-half component spectrum.
- Proved continuous unitary rotation composition, `2π` sign reversal, `4π`
  return, and exact standard-basis intertwiners on both copies.
- Proved covariance and unitarity uniquely characterize the positive native
  quarter-turn coefficients inside the scalar-plus-plane rotor family.
- Proved mass-grading compatibility and both quarter-turn and all-angle
  covariance of the complete interpreted mass-plus-three-direction transport
  Hamiltonian.
- Added explicit attribution to Teixeira's 2013 review and the foundational
  hypercubic lattice-fermion work of Becher--Joos and Rabin; no priority is
  claimed for the cell-based Clifford construction.
- Added no field, action term, stencil change, runtime projection,
  electromagnetic coupling, or localized-electron claim. Native realization
  of the representation on a freely evolving localized carrier remains open.

## Palomar bridge repair (unreleased)

- Proved that the advertised native two-field Lagrangian restricts exactly to
  the retained eight-site coordinates and that arbitrary retained first
  variations commute with that restriction.
- Proved the complementary force and momentum blocks vanish, so the bridge
  does not impose a runtime projector or alter the action.
- Replaced the inserted exponential mode as the claimed time-evolution bridge
  with a first-order branch derived from the centered recurrence's
  characteristic roots and the already-derived Hamiltonian square.
- Proved that the new branch solves the continuum centered recurrence exactly,
  has native defect exactly equal to the C19 spatial remainder, and converges
  at the generator level to the same Dirac Hamiltonian.
- Added independent 32-case reconstruction of the repaired branch identities.
- No field, action term, stencil weight, fitted coefficient, or physical
  electron claim was added.

## DIRAC-016

- Proved that the existing unsigned period-two cube translation exchanges
  the two retained rank-four copies in one coordinate direction.
- Proved the induced four-coordinate map is unique, preserves native corner
  density, and returns to the original embedded state after two translations.
- Calculated an exact 12-to-12 balance between copy-exchanging and
  copy-preserving C19 path weights.
- Proved separately that the signed incidence operator used by the cubical
  Dirac readout preserves each copy.
- Added no field, action term, runtime projection, transport law, or physical
  electron claim. Localized copy-alternating motion remains an open test.

## DIRAC-014--015

- Completed a top-down adversarial audit with zero critical/high findings.
- Added one cumulative Lean theorem connecting the action-derived massive C19
  Klein--Gordon tangent to its cubical four-component Dirac square root.
- Retained explicit finite-spacing and centered-time remainders and the
  action-derived density/current limit.
- Added deterministic clean-checkout packaging with an internal file manifest
  and external archive checksum.
- No new field, action term, Dirac matrix, spinor rule, or physical-electron
  claim was added.

## DIRAC-013

- Derived the oriented C19 bond flow by varying the same native bond action.
- Proved the exact local phase-density continuity identity before introducing
  any four-component observable.
- Used the derived positive-frequency graph and explicit normalization to
  recover the standard four-component density.
- Proved that the exact finite C19 group flow approaches the standard Dirac
  current in the low-momentum limit for the same embedding.
- Retained both cubical copies and added an independent 32-case reconstruction.
- Mixed-frequency positivity, exact finite-lattice current equality, nonlinear
  live-chi current closure, physical spin, and an electron remain open.

## DIRAC-012

- Derived the prepared-vacuum quadratic mass and C19 transport coefficient
  directly from the admitted local-plus-bond action.
- Connected the continuum Dirac frequency to the actual centered recurrence.
- Proved separate spatial and temporal errors and a normalized equation-error
  bound; no exact finite-lattice Dirac claim is made.
- Added an independent 32-case reconstruction and joint refinement check.
- Density/current, nonlinear live-chi evolution, physical spin, and an
  electron remain open.

## DIRAC-011

- Added positive- and negative-frequency graphs on the same native canonical
  phase space and for both retained rank-four copies.
- Kept the exact finite-lattice velocity defect equal to the bounded C19
  spatial remainder rather than claiming exact finite-lattice Dirac evolution.
- Proved the prepared native running rest frequency is `chi0=19` and that it is
  the unique nonnegative standard mass coefficient with the same rest spectrum.
- Added an independent reconstruction of both frequency graphs, both copies,
  the remainder identity, and the mass selection.
- The complete controlled continuum evolution theorem, current, spin, and an
  electron remain open.

## DIRAC-010

- Removed covariance-shape and signed-incidence energies from the admitted
  DIRAC-002 action.
- Restored the formal chain through DIRAC-008 to the base positive
  local-plus-exterior-area C19 action.
- Recast DIRAC-009 as a continuum square-root factorization of the native C19
  quadratic symbol with an explicit fourth-order lattice remainder.
- Proved exact invertible mappings from both native rank-four copies to the
  conventional Dirac alpha/beta representation.
- Added an independent matrix reconstruction with zero discrepancy.
- Removed superseded incidence-action modules and evidence from production.
- Physical mass, the full continuum Dirac theorem, current, spin, and an
  electron remain open.

## DIRAC-009 history

The prior incidence-action bridge is preserved in Git history. It was retired
because it inserted the desired signed response into the native action rather
than deriving it as a downstream continuum readout.
