DIRAC-028 PASS: changing direction produces the Thomas half factor

Two boosts in different directions do not simply add. Their order leaves a
small rotation behind--the Wigner/Thomas rotation.

Lean decomposed two successive boosts built in DIRAC-026. The part that
changes sign when their order is reversed became the cube rotation generator,
with the exact coefficient

    one half of the boost commutator.

That `1/2` was not fitted or added to the framework. It is forced when the
ordered product is split into its symmetric and antisymmetric pieces. Lean
also used a nonzero cube state to prove no other coefficient can give the same
operator. Reversing the boost order reverses the rotation, as it should.

This is the local, small-boost result; we are not claiming an unproved exact
finite-angle formula. It shows that the electron sector's response to a
change of direction contains the same Thomas half factor known from
relativistic spin motion.

Next, DIRAC-029 checks the other free-electron half of the Dirac spectrum:
whether the same native operator carries an internally derived
charge-conjugate negative-frequency partner.
