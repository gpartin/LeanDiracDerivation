import LeanDirac.Dirac012

namespace LeanDirac

/-!
Sole cumulative research proof root through DIRAC-012.

The nested certificate starts from the positive two-field C19 action, retains
the exact native second-order and first-order maps, derives the cubical
continuum square root with its finite-lattice remainder, and identifies both
rank-four cubical copies with the conventional Dirac representation, constructs
both frequency graphs, selects the unique nonnegative rest-mass coefficient
from the native running frequency, and proves the prepared-vacuum,
low-momentum continuum evolution with explicit spatial and centered-time
errors. Current, spin, and an electron remain downstream claims.
-/

structure CumulativeDirac012Certificate : Prop where
  nativeConstants : chi0 = 19 ∧ kappa = 1 / 63 ∧
    chiKineticNormalization = 1197
  nativeC19Weights : ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  continuumEvolution : Dirac012Certificate

theorem current_highest_gate_is_dirac012 :
    CumulativeDirac012Certificate where
  nativeConstants := ⟨rfl, rfl, chi_kinetic_normalization_eq_1197⟩
  nativeC19Weights := by
    intro neighbor
    fin_cases neighbor <;> norm_num [c19Weight]
  continuumEvolution := dirac012_controlled_continuum_evolution

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac012
