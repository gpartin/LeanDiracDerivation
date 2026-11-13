import LeanDirac.Dirac016

namespace LeanDirac

/-!
Sole cumulative publication root through DIRAC-016.

The nested certificate starts from the positive two-field C19 action, retains
the exact native second-order and first-order maps, derives the cubical
continuum square root with its finite-lattice remainder, and identifies both
rank-four cubical copies with the conventional Dirac representation, constructs
both frequency graphs, selects the unique nonnegative rest-mass coefficient
from the native running frequency, and proves the prepared-vacuum,
low-momentum continuum evolution with explicit spatial and centered-time
errors, and derives the positive-frequency density/current limit from the same
action and embeddings.  The final certificate states explicitly that the
action-derived Klein--Gordon tangent symbol is the square of the derived
continuum Dirac Hamiltonian.  Nonlinear live-chi closure and a localized
physical electron remain downstream claims.

DIRAC-016 adds a downstream finite-cube theorem: unsigned microscopic
translation exchanges the two retained rank-four copies in one cube
direction and a second translation returns to the original copy.  This does
not change the action or the DIRAC-015 continuum theorem.  It does not assert
that native time evolution performs one exchange per update or that a
localized electron moves by this mechanism.
-/

structure CumulativeDirac015Certificate : Prop where
  nativeConstants : chi0 = 19 ∧ kappa = 1 / 63 ∧
    chiKineticNormalization = 1197
  nativeC19Weights : ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  freeDiracContinuumLimit : LFMFreeDiracContinuumCertificate

theorem current_highest_gate_is_dirac015 :
    CumulativeDirac015Certificate where
  nativeConstants := ⟨rfl, rfl, chi_kinetic_normalization_eq_1197⟩
  nativeC19Weights := by
    intro neighbor
    fin_cases neighbor <;> norm_num [c19Weight]
  freeDiracContinuumLimit := lfm_free_dirac_continuum_limit

structure CumulativeDirac016Certificate : Prop where
  completedDiracDerivation : CumulativeDirac015Certificate
  copyTranslationStructure : Dirac016CopyTranslationCertificate

theorem current_highest_gate_is_dirac016 :
    CumulativeDirac016Certificate where
  completedDiracDerivation := current_highest_gate_is_dirac015
  copyTranslationStructure := dirac016_exact_copy_translation_structure

end LeanDirac

#print axioms LeanDirac.current_highest_gate_is_dirac015
#print axioms LeanDirac.current_highest_gate_is_dirac016
