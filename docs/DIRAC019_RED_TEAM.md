# DIRAC-019 red-team record

## Question tested

Earlier gates derived a positive-frequency first-order branch one momentum at
a time. DIRAC-019 asks whether those modes can carry conserved density and the
derived current for arbitrarily many recurrence-derived continuum-branch updates, and whether a
finite spectral family has a mathematically determined transport velocity.

## Critical/high findings and repairs

| Severity | Finding | Repair | Disposition |
|---|---|---|---|
| High | A one-update norm identity would not establish sustained propagation. | Added exact all-integer-update theorems: the recurrence-derived phase advances each eigenmode while both density and density-weighted current persist for every update count. | Fixed. |
| High | Calling a list of modes a localized electron or a fully formalized spatial wave packet would overstate the theorem. | Scoped the formal result to a finite positive-frequency spectral family. Added an independent Fourier packet diagnostic, while leaving the coherent real-space/Parseval theorem and nonlinear electron transport open. | Fixed. |
| High | Movement could have been inserted by manually translating the state. | Defined transport only through the Hamiltonian momentum response, its previously derived Noether/Dirac current, and the recurrence phase. No coordinate shift, track, force, or update rule was added. | Fixed. |
| High | The continuum branch could be mislabeled as exact native finite-spacing motion. | Preserved the upstream exact C19 spatial remainder and explicitly excluded exact native finite-spacing packet evolution from DIRAC-019. | Fixed. |
| Medium | A positive-frequency statement could silently include the negative branch. | Kept the positive-eigenvalue equation explicit in the Lean mode type and added a numerical negative-frequency rejection control. | Fixed. |
| High | A theorem quantified over positive eigenmodes could be formally vacuous if no such mode were constructed. | Constructed an explicit nonzero positive-frequency eigenmode at every momentum from the already-proved Hamiltonian square. | Fixed. |
| Medium | The unit-phase claim is false outside the centered-recurrence stability interval. | Made the stability inequality explicit in every conservation theorem and added an unstable-timestep rejection control. | Fixed. |

## Chain-level result

The cumulative theorem still begins with the same positive two-field C19
action and contains the complete DIRAC-001--DIRAC-018 chain. DIRAC-019 adds no
field, action term, stencil change, projection, translation law, or fitted
coefficient. Its movement statement uses the same four-component embedding,
mass, transport matrices, recurrence branch, density, and current already
derived upstream.

## Remaining boundary

The Lean theorem is spectral: it does not yet construct a coherent localized
field on a finite spatial torus or prove a uniform finite-time continuum error
bound. The independent Fourier synthesis shows a narrow linear packet moving
at the current-predicted velocity under timestep refinement, but that is not a
nonlinear electron simulation. DIRAC-020 should formalize the real-space
Fourier packet, Parseval density, centroid/current relation, and a finite-time
error bound without adding a movement rule.
