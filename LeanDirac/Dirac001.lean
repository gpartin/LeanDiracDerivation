import LeanDirac.C19Geometry

namespace LeanDirac

/-!
DIRAC-001: explicit native-framework boundary.

This gate fixes the two native complex-C3 field families, the declared
constants, and the positive C19 neighborhood weights.  It makes no action,
prepared-background, spinor, electron, or Dirac claim.
-/

theorem dirac001_native_framework (state : State Site) :
    chi0 = 19 ∧
    kappa = 1 / 63 ∧
    chiKineticNormalization = 1197 ∧
    state = ⟨state.psi, state.chi⟩ ∧
    ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor := by
  exact ⟨rfl, rfl, chi_kinetic_normalization_eq_1197,
    state_has_exactly_two_field_families state, c19_weight_positive⟩

end LeanDirac
