# V3.4 Hostile Review 1

## Verdict before fixes

The seven-stage architecture was justified and no action term, field,
coefficient, standard Dirac matrix, textbook spinor, measured electron input,
or physical-unit scale was inserted. The first hostile pass nevertheless
found two high-severity claim-completeness gaps and one medium verifier defect.

## Findings and repairs

### HIGH-1: helicity projectors lacked an eigenbranch theorem

The initial Stage 4 implementation proved the arbitrary-momentum helicity
square, projector completeness, and projector idempotence. Those properties
did not by themselves establish the documented statement that each projector
lands in the corresponding `+/-1/2` eigenspace.

Repair: `v34_helicity_projector_is_eigenbranch` now proves the eigenvalue
equation for either sign under `sign^2 = 1`, and the theorem is a required
field of `V34MomentumHelicityCertificate`.

### HIGH-2: the non-forward native corollary derived only the incoming shell

The general Gordon theorem correctly used explicit left- and right-on-shell
hypotheses, but the first native corollary discharged only the incoming
right-shell condition. The outgoing native spinor was still passed through an
abstract left-shell premise. That was too weak for the advertised
native-to-native handoff.

Repair: Stage 5 now proves the bilinear residual/Hamiltonian-residual bridge,
uses the already-proved Hermiticity of the derived Hamiltonian to establish
`v34_positive_spinor_left_on_shell`, and proves
`v34_nonforward_gordon_for_native_spinors` with independently chosen incoming
and outgoing native-derived spinors.

### MEDIUM-1: the historical V3.2 verifier rejected all downstream modules

The V3.2 framework-lock audit classified every `V33*` module as a forbidden
core edit, so it could not be rerun from a cumulative V3.3 or V3.4 tree.

Repair: the audit remains strict for migrated V3.1/core files but explicitly
recognizes `V33*` and `V34*` as downstream modules. Its facade check also
recognizes the cumulative V3.4 import. The original V3.2 source, claim, theorem
inventory, and build gates remain unchanged.

### LOW-1: avoid wrapper-only stage inflation

No extra framework-lock or scope-audit Lean wrapper was added. Those checks
remain in the verifier and documentation, leaving seven scientific handoffs.

## Post-fix claim boundary

The repaired chain proves covariant free-spinor closure only. It does not
prove an interacting electromagnetic vertex, form factors, `g = 2`, charge
quantization, localization, transport, an absolute MeV scale, or a physical
electron.

## Gate status

All critical findings: none.

All high findings: repaired; cumulative verification required again.
