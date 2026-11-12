import LeanDirac.Dirac013

namespace LeanDirac

/-!
Sole cumulative research proof root through DIRAC-013.

The nested certificate starts from the positive two-field C19 action, retains
the exact native second-order and first-order maps, derives the cubical
continuum square root with its finite-lattice remainder, and identifies both
rank-four cubical copies with the conventional Dirac representation, constructs
both frequency graphs, selects the unique nonnegative rest-mass coefficient
from the native running frequency, and proves the prepared-vacuum,
low-momentum continuum evolution with explicit spatial and centered-time
errors, and derives the positive-frequency density/current limit from the same
action and embeddings. Nonlinear live-chi current, spin, and an electron remain
downstream claims.
-/

structure CumulativeDirac013Certificate : Prop where
  nativeConstants : chi0 = 19 ∧ kappa = 1 / 63 ∧
    chiKineticNormalization = 1197
  nativeC19Weights : ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  densityCurrentContinuity : Dirac013Certificate

theorem current_highest_gate_is_dirac013 :
    CumulativeDirac013Certificate where
  nativeConstants := ⟨rfl, rfl, chi_kinetic_normalization_eq_1197⟩
  nativeC19Weights := by
    intro neighbor
    fin_cases neighbor <;> norm_num [c19Weight]
  densityCurrentContinuity := dirac013_native_density_current_continuity

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac013
