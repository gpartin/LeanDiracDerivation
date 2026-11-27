# V3.4 Research Release Notes

V3.4 is a cumulative theorem-only extension of V3.3. It changes no LFM
action, governing equation, C19 geometry, coefficient, vacuum, readout, or
physical scale.

New machine-checked results:

- covariant positive-frequency and antiparticle spinors constructed from the
  existing native two-channel frequency embeddings;
- exact two-state branch coordinates, injectivity, spanning, and
  positive/negative orthogonality;
- covariant positive and antiparticle spin sums;
- arbitrary nonzero-momentum helicity with square `1/4`, complete projectors,
  projector idempotence, and the corresponding `+/-1/2` eigenbranches;
- the non-forward Gordon identity for independently chosen native-derived
  incoming and outgoing spinors, with both on-shell conditions derived;
- spectral decomposition of the free resolvent into V3.2 projectors;
- identification of positive and negative pole numerators with those
  projectors and of the positive spin sum with the positive projector after
  beta conversion;
- cumulative theorem `v34_complete_covariant_free_spinor`.

V3.4 remains a free linearized result. It does not prove an interacting or
localized electron, electromagnetic emergence, form factors, `g = 2`, charge
quantization, transport, or conversion of native mass `19` into MeV.

Post-implementation organization cleanup strengthened the cumulative
certificate to expose the unique two-plus-two decomposition, both branch
Gram relations, and the native-to-native Gordon theorem. It also replaced the
artificial serial import chain with the actual dependency graph, reused the
existing covariant gamma definitions, factored shared spin-sum normalization,
and removed superseded helpers. No scientific claim or equation changed.
