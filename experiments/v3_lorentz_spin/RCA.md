# V3 release RCA

Issue: The first Lorentz/spin draft had a circular Pauli--Lubanski rest
definition and did not prove the full finite-rotation representation or the
covariance of the complete Dirac symbol.

Why it matters: Those gaps allowed the desired spin invariant and covariance
claim to be restated rather than derived.

Symptoms: The rest square was defined directly as `-m^2 S^2`; finite evidence
was limited to density and endpoint rotations; covariance stopped at gamma
commutators.

Expected symptoms: A general `W(P,J,K)`, `P.W = 0`, a rest specialization,
identity/composition/inverse/differentiation for rotations, and a joint
spinor--momentum transformation of the full operator.

Root cause: The initial extension packaged correct local algebra before its
representation-level consequences had been formalized.

Corrective action: V3 adds those missing definitions and theorems, intertwines
them through both native copies, and seals the cumulative theorem with a clean
compiled-environment axiom audit.
