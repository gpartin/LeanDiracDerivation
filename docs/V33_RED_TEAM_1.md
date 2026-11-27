# V3.3 Red Team 1

## Verdict before fixes

No critical issue was found. Two high-severity claim/closure issues and three
medium documentation issues were found.

## Findings and dispositions

1. **HIGH — the first Gordon-labeled result was only a velocity-current
   identity.** Fixed before this formal pass by deriving the scalar bilinear
   and proving both forward time and spatial relations. The module explicitly
   excludes the non-forward electromagnetic vertex and `g = 2`.
2. **HIGH — “spin sum” could have meant only projector addition.** Fixed by
   adding an actual outer-product sum over the complete orthonormal component
   basis, while retaining the separately proved positive/negative spectral
   resolution. No covariant QFT spin-sum claim is made.
3. **MEDIUM — the free resolvent was only a right inverse.** Fixed by proving
   both left and right inverse identities under the same off-pole hypothesis.
4. **MEDIUM — static spin eigenvectors were insufficient evidence for
   helicity.** Fixed before this formal pass by constructing axis-aligned
   moving positive-frequency states and proving both half-valued helicities.
5. **MEDIUM — the verifier did not reject new axiom/constant declarations.**
   Fixed by adding a source gate in addition to Lean's printed axiom audit.

## Remaining limitations (not defects)

- The projected four seeds are a spanning construction; V3.3 does not claim
  that all four are independent in each rank-two branch.
- Helicity is proved for the exact axis-aligned case, not arbitrary momentum.
- Gordon is the forward equal-momentum specialization, not a non-forward
  vertex theorem.
- The resolvent is free and momentum-space algebraic, not interacting.
- No charge quantization, magnetic moment, localization, SI calibration, or
  physical-electron identity is claimed.

Critical open: 0. High open: 0.
