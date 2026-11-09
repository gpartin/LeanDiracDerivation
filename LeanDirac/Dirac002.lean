import LeanDirac.Dirac001
import LeanDirac.PreparedBackground

namespace LeanDirac

/-!
DIRAC-002: reduced positive two-field action and prepared geometry.

This gate depends explicitly on DIRAC-001 and makes no Dirac claim yet.
-/

noncomputable section

theorem dirac002_positive_two_field_action
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) (hchi : 0 < chi0) (hB : 0 ≤ B)
    (hweight : ∀ edge, 0 ≤ (bond edge).weight) :
    0 ≤ hamiltonian chi0 B state velocity bond ∧
      state = ⟨state.psi, state.chi⟩ := by
  exact ⟨hamiltonian_nonnegative chi0 B state velocity bond
    hchi hB hweight, state_has_exactly_two_field_families state⟩

theorem dirac002_prepared_geometry :
    (∀ corner : CubeCorner,
      wedge (tetraCorner corner) (companion corner) = unitArea) ∧
    faceProduct 0 2 3 1 = (-1/9 : ℂ) := by
  exact ⟨tetra_wedge_companion_constant,
    all_face_products_negative.1⟩

end
end LeanDirac

