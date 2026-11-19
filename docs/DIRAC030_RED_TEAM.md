# DIRAC-030 red-team review

## Verdict

`PASS_SCOPED_FREE_GORDON_CURRENT_DECOMPOSITION`

No critical or high finding remains. No native framework object changes.

## Injection and circularity checks

- Covariant gamma operators are composed from the inherited mass grading and
  the three inherited C19 transport generators.
- The spin tensor is their commutator, not an independent spin matrix.
- Lean proves the Gordon kernel componentwise before using any on-shell state.
- The Dirac-adjoint equation is proved from the same operators and inner
  product; it is not separately assumed.
- Only the two standard on-shell equations and nonzero mass are premises of
  the final matrix-element split.
- The `1/(2m)` coefficient follows algebraically after the two on-shell mass
  contributions are combined.

## Adversarial controls

Changing a spatial metric sign breaks the kernel. Removing the commutator
term leaves momentum-transfer residuals. Replacing `1/(2m)` by `1/m` fails
for generic on-shell states.

## Claim boundary

This is the free Gordon decomposition into convection and intrinsic-spin
current. It does not prove electric coupling, a magnetic moment, `g = 2`, or
SI calibration.
