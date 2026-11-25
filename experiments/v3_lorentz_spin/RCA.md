# V3 release RCA

Issue: The first Lorentz/spin draft had a circular Pauli--Lubanski rest
definition and did not prove the full finite-rotation representation or the
covariance of the complete Dirac symbol. The V3.0 generalization also chose the
wrong cross-product sign for its own boost convention, and the release prose
did not clearly separate operator convergence from solution convergence.

Why it matters: Those gaps allowed the desired spin invariant and covariance
claim to be restated rather than derived.

Symptoms: The rest square was defined directly as `-m^2 S^2`; finite evidence
was limited to density and endpoint rotations; covariance stopped at gamma
commutators. Later, orthogonality and the rest square passed even though an
independent moving-frame check found a nonzero four-vector defect.

Expected symptoms: A general `W(P,J,K)`, `P.W = 0`, a rest specialization,
identity/composition/inverse/differentiation for rotations, and a joint
spinor--momentum transformation of the full operator.

Root cause: The initial extension packaged correct local algebra before its
representation-level consequences had been formalized. Its tests were
insensitive to the cross-product sign because those even/rest identities pass
for both signs. Rotation normalization was also justified only by algebraic
closure, leaving its agreement with native transport implicit.

Corrective action: V3.1 adds the missing definitions and theorems, fixes the
Pauli--Lubanski sign, proves its boost and rotation four-vector laws for every
axis, and adds a compiled witness showing the old sign fails. It proves that
the algebraic spin scale agrees uniquely, within the explicit local
plane-bivector class, with rotation of the native transport directions. It
intertwines the result through both native copies, narrows continuum language
to operator/mode claims, and seals all release-facing files with clean
compiled-environment and regression axiom audits.
