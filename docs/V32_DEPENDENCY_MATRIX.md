# V3.2 Dependency and Claim Matrix

This table records what each V3.2 conclusion actually depends on. It is a
claim audit of the Lean declaration graph, not a new physical assumption.

| Result | Unchanged V3.1 action/C19 geometry | `chi0 = 19` | numerical `B = 1197` | physical unit anchor | finite-C19 exact first-order root |
|---|---:|---:|---:|---:|---:|
| Continuum cubical Clifford/Dirac operator | yes | symbolic mass, then `chi0` for canonical specialization | no | no | no |
| Exact finite-C19 scalar spectrum and gap | yes | yes for canonical mass | no | no | no |
| Continuum spectral projectors | yes | symbolic mass | no | no | no |
| Two-copy multiplicity and rest-spin irreducibility | yes | yes at canonical rest specialization | no | no | no |
| C/P/T/CPT continuum operator laws | yes | symbolic mass | no | no | no |
| Free Schrodinger limit and correction | yes | symbolic mass; `19` gives `1/38` | no | no | no |
| Finite-C19 spatial/current remainder bounds | yes | no | no | no | no |
| Native rest energy `19` and gap `38` | yes | yes | no | no | no |
| Conversion to MeV or kilograms | yes | yes | no | **yes** | no |
| Exact finite-C19 local Dirac projector | incompatible with current four-component local root | n/a | n/a | n/a | **absent** |

The free transverse action theorem is quantified over arbitrary `B`. Thus the
free Dirac mass, Clifford transport, density/current bridge, and the exact
finite-C19 scalar spectrum do not select the number `1197`. This does not make
`B` irrelevant to the full framework: it remains the kinetic normalization of
the live chi field outside the decoupled linear free sector.

This boundary is exported as the Lean theorem
`v32_free_transverse_tangent_uniform_in_B` and as the `arbitraryB` field of
`V32RobustnessCertificate`; it is not inferred only from this table.

The physical-unit row is intentionally negative. V3.2 proves
`E_rest,physical = 19 * E_native_unit` and proves that the unchanged
dimensionless theory does not choose `E_native_unit`. Therefore it does not
derive `0.511 MeV`, and no measured electron value is used as an input.

Route 1 also keeps the finite-spacing distinction explicit. The exact C19
second-order spectrum is proved, while the local four-component Dirac root is
the continuum root with a controlled remainder. Lean proves that identifying
the two globally at finite spacing would be false.

The rest specialization is also checked jointly for both exhaustive copies by
`v32_two_copy_rest_sector`: both copies carry the same positive rest mass on
both rest-spin basis states, unit-density explicit rest spinors, zero rest
current, native mass `19`, and gap `38`.
