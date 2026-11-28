# V3.6 Research Release Notes

V3.6 intentionally revises the branch-local research action's relative
kinetic normalization. The previous values `B = 1197` and `kappa = 1/63` are
retained only as a historical control. The live research action now computes
`B` from the corresponding cubical vector/bivector unit-blade norm ratio in
one common Euclidean Clifford kinetic metric, giving

```text
B = 1
kappa = chi0 / B = 19
```

The complete V2--V3.5 free-Dirac, Lorentz/spin, free-fermion, free-spinor, and
covariant-observable theorem graph recompiles under the revised foundation.
The free transverse tangent theorem is uniform in `B`, so this no-regression
result does not use the new number to manufacture the Dirac operator.

The theorem `v36_live_research_action_mode_to_free_dirac` closes the direct
formal link: its tangent quotient is defined from
`v36ResearchC19Lagrangian`, and its conclusion is the complete
finite-periodic action-to-free-Dirac statement with controlled remainder.

The release also keeps the boundary explicit: choosing one common Euclidean
kinetic metric is the V3.6 action-design premise. Lean proves the coefficient
uniquely within that class, not among every possible two-field action.
Nonlinear source-free electron localization, transport, and an operational
Einstein limit remain open. This research release does not alter the
repository-wide canonical v40.1 framework.

Run:

```text
python scripts/verify_release.py
```

The strict verifier builds the cumulative facade and environment audit,
checks the revised source declarations, rejects unfinished proofs and new
axioms, and confirms that the cumulative theorem uses only Lean's standard
`propext`, `Classical.choice`, and `Quot.sound` axioms.
