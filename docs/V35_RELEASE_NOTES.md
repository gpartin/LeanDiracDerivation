# V3.5 Research Release Notes

V3.5 is a cumulative theorem-only extension of V3.4. It changes no LFM action,
governing equation, C19 geometry, coefficient, vacuum, readout, or physical
scale.

Machine-checked additions:

- antiparticle spin-sum equality with the negative spectral projector and
  propagator pole, including the required momentum convention;
- exact words generated from the existing finite cubical rotations and
  boosts, with composition and inverse laws;
- finite coordinate-axis boost covariance of the derived free Dirac residual
  and preservation of the mass shell;
- a cube-derived gamma-five and the standard scalar, pseudoscalar, vector,
  axial-vector, and antisymmetric tensor bilinears;
- an explicit sixteen-channel Clifford organization whose `1/4` spinor and
  `1/2` ordered-tensor normalizations are derived from finite channel counts;
- the Pauli--Lubanski spin-one-half invariant at arbitrary on-shell momentum;
- exact agreement between V3.4 helicity and the normalized Pauli--Lubanski
  time component when the normalization is the positive momentum norm;
- a cumulative invariant interface for testing the separate V4 transport
  implementation.

V3.5 remains a free linearized result. It does not prove finite-packet
transport, an interacting or localized electron, electromagnetic emergence,
physical charge, form factors, `g = 2`, charge quantization, statistics, or
conversion of native mass 19 into MeV.

The finite covariance result applies to the free Dirac residual and its
solution space. The bilinear branch proves the five definitions, their
Clifford organization, and the Fierz completeness identity. It does not add a
separate finite transformation theorem for every bilinear class.
