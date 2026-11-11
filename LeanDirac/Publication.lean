import LeanDirac.Dirac010

namespace LeanDirac

/-!
Sole cumulative research proof root through DIRAC-010.

The nested certificate starts from the positive two-field C19 action, retains
the exact native second-order and first-order maps, derives the cubical
continuum square root with its finite-lattice remainder, and identifies both
rank-four cubical copies with the conventional Dirac representation. Physical
mass selection, the full continuum evolution theorem, current, spin, and an
electron remain downstream claims.
-/

structure CumulativeDirac010Certificate : Prop where
  nativeConstants : chi0 = 19 ∧ kappa = 1 / 63 ∧
    chiKineticNormalization = 1197
  nativeC19Weights : ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  standardIdentification : Dirac010Certificate

theorem current_highest_gate_is_dirac010 :
    CumulativeDirac010Certificate where
  nativeConstants := ⟨rfl, rfl, chi_kinetic_normalization_eq_1197⟩
  nativeC19Weights := by
    intro neighbor
    fin_cases neighbor <;> norm_num [c19Weight]
  standardIdentification := dirac010_native_operator_equals_standard_dirac

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac010
