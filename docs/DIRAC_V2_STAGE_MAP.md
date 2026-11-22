# DIRAC V2 stage map

V2 is the independent top-down route from the native action through emergence
of the complete free Dirac equation. It maps onto the useful V1 stage names
through `DIRAC-V2-015`, but it does not need to reproduce downstream V1
consequence stages. The Lean implementation remains split into content-named
modules so definitions are not duplicated merely to manufacture stage files.
A stage passes only when the theorem named below exists and its transitive
dependencies pass the compiled axiom audit.

The historical numbered files remain unchanged V1 provenance. They are not
dependencies of V2.

| V2 gate | V1 role retained | V2 proof boundary | Status |
|---|---|---|---|
| DIRAC-V2-001 | Native framework boundary | Native `Psi,chi : C3`, C19 bonds and constants remain the declared input. | PASS_REUSED |
| DIRAC-V2-002 | Positive two-field action | `NativeVariation.potential` is the unchanged canonical V40 action. | PASS_REUSED |
| DIRAC-V2-003 | Reciprocal action variation | `potential_hasDerivAt` and the finite action first-variation lemmas derive both field responses. | PASS |
| DIRAC-V2-004 | Reversible native update | The actual centered/Verlet recurrence is formalized; stability and modified-energy preservation are proved. | PASS |
| DIRAC-V2-005 | Prepared sector closure | Replaced: the uniform zero-energy native vacuum is proved critical/minimal and its full transverse tangent sector is closed without a maintained period-two background. | PASS_REPLACED |
| DIRAC-V2-006 | Native restriction and C19 transport | `periodic_full_native_action_generator_intertwines` gives one full `State x Velocity` equality backed by the action Hessian, with explicit complement leakage checks. | PASS |
| DIRAC-V2-007 | Lossless four-component readout | `cubical_phase_readout_unique_from_intertwining` proves the readout is the unique position-normalized intertwiner for the derived cubical root. | PASS |
| DIRAC-V2-008 | Complete first-order map | `uniform_action_tangent_intertwines_cubical_dirac` gives the exact four-component first-order generator. | PASS |
| DIRAC-V2-009 | Cubical Clifford square root | The oriented cube boundary/coboundary constructs the Kähler--Dirac root, its Clifford relations, both copies and the explicit finite-C19 remainder. | PASS |
| DIRAC-V2-010 | Standard representation | The two cubical flavor copies are converted to conventional alpha/beta coordinates by proved intertwiners. | PASS |
| DIRAC-V2-011 | Frequency branches and mass | Positive/negative rest grading and the unique nonnegative mass `m = chi0` are derived from the same native Hessian. | PASS |
| DIRAC-V2-012 | Native time to continuum Dirac | The actual stable Verlet recurrence has an exact all-iteration characteristic solution; its exact discrete angular frequency converges to the action-Hessian frequency as the timestep vanishes. | PASS |
| DIRAC-V2-013 | Density and current | One bijective canonical spectral readout now carries the full native generator, exact C19 defect, centered/Verlet update, positive-mode U(1) density, literal C19 bond current, and finite Fourier position-space equation.  The current remainder is bounded by `16 r^3 rho` and tends to zero.  Pointwise packet equality is explicitly disproved; the correct integrated finite-torus charge bridge is proved by Fourier orthogonality. | PASS |
| DIRAC-V2-014 | Adversarial audit | Independent top-down review found no critical/high mathematical defect; it confirmed the native-action path, lossless operator bridge, controlled C19/time corrections, and scoped nonlinear/localized exclusions. | PASS |
| DIRAC-V2-015 | Cumulative action-to-Dirac theorem | `CompleteFreeDirac` builds on the clean production base; proof and audit/toolchain sources are hash-sealed, all required declarations are compiled, and the compiled-environment audit admits only the standard Lean foundations. | PASS |
| DIRAC-V2-016--040 | Downstream consequences | Historical copy, spin, packet, boost, invariant, symmetry and Pauli-sector results are preserved in V1 but are outside this equation-derivation scope. | OUT_OF_SCOPE |

## Dependency rule

The controlling route is:

```text
DIRAC-V2-001 -> ... -> DIRAC-V2-015
```

No V1 prepared-background certificate or downstream consequence theorem may
satisfy a V2 gate.

## Current claim boundary

The target of V2-015 is the complete **free, linear tangent Dirac equation**
of the native action in one exact uniform zero-energy vacuum sector. The
sector is a proved native critical point and global minimum, not an external,
prepared, period-two, or maintained background. Dynamic alignment of arbitrary
vacuum orientations is an origin question and is not claimed. Root
orientation, global sign and the two cubical flavor charts remain
representation freedoms rather than action-selected observables.

The requirement-by-requirement completion review is recorded in
[`DIRAC_V2_COMPLETION_AUDIT.md`](DIRAC_V2_COMPLETION_AUDIT.md).
