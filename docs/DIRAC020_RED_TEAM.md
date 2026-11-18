# DIRAC-020 red-team review

## Verdict

`PASS_SCOPED_COHERENT_SPATIAL_TRANSPORT`

The gate closes the coherent finite spatial Fourier bridge without modifying
the native action or equations.  It remains a linear continuum-branch result,
not a nonlinear localized-electron theorem.

## Findings and repairs

| Severity | Finding | Repair | Result |
|---|---|---|---|
| Critical | A generic object called a spatial frame could hide the desired answer in an assumed Parseval law. | Constructed a normalized Fourier frame explicitly for every finite abelian periodic lattice and proved its orthonormality from Mathlib's independent finite-character orthogonality theorem. | Closed. |
| High | A Fourier character could be paired with an unrelated momentum stored in the positive-frequency mode. | Defined each mode momentum from the character's three one-site phases, proved the normalized basis transforms by exactly those phases on a periodic cubic lattice, constructed a compatible packet, and restricted the transport certificate to compatible packets. | Closed. |
| High | A finite list of independent modes is not one coherent spatial field. | Defined the four-component Fourier synthesis and proved the full componentwise Parseval identity, not merely a sum of mode norms. | Closed. |
| High | Nonzero “movement” could be asserted from a packet label or manual translation. | Proved integrated spatial current equals the already-derived density-weighted velocity moment. Nonzero movement requires that moment to be nonzero; a symmetric zero-current numerical control is retained. | Closed. |
| High | One update would not establish sustained movement. | Density and current results hold for every natural-number update count under the explicit centered-recurrence stability condition inherited from DIRAC-019. | Closed. |
| High | The recurrence phase could silently be equated with the exact continuum phase. | Kept the phases distinct and proved a finite-update pointwise field-error bound proportional to the accumulated one-step phase mismatch. | Closed. |
| High | The result could be mislabeled exact native finite-spacing motion. | The exact C19 spatial remainder and the lack of a uniform fixed-physical-time bound remain explicit downstream obligations. | Closed as a claim boundary. |
| Medium | Fourier normalization could be decorative. | Independent evidence deliberately removes the normalization and observes the expected factor-of-N Parseval failure. | Closed. |
| Medium | A zero-current superposition might still be called moving. | Independent equal-and-opposite momentum control yields zero velocity moment and is not admitted as directed transport. | Closed. |
| Medium | The stable-step hypothesis might be decorative. | The independent validator rejects an unstable timestep. | Closed. |

## Full-chain review

- The publication root imports DIRAC-020 transitively from the unchanged
  DIRAC-001--019 chain.
- No native field, action term, force, translation rule, projector, stencil
  change, fitted coefficient, or status proposition enters the proof.
- The Fourier characters supply only the ordinary spatial reconstruction; all
  dynamics, phase, density and current come from the existing derived branch.
- The formal certificate requires the spatial character and dynamical momentum
  labels to agree; it cannot pass by pairing unrelated labels.
- Every audited theorem depends only on `propext`, `Classical.choice`, and
  `Quot.sound`.
- Remaining physical gaps are exact native finite-spacing propagation,
  uniform fixed-physical-time error control, native packet selection,
  nonlinear unrestricted live-chi localization, electromagnetic response and
  electron identity.

