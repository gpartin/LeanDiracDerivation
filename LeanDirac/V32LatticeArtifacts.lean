import LeanDirac.V32ContinuumOperatorConvergence
import LeanDirac.V32BrillouinZoneAudit

/-! # V3.2 Stage 11: explicit finite-C19 artifact -/

namespace LeanDirac

noncomputable section

set_option maxHeartbeats 1000000

def v32SpatialArtifact (momentum : LatticeMomentum) : ℝ :=
  c19TransportSymbol momentum - c19QuadraticSymbol momentum

theorem v32_spatial_artifact_is_c19_remainder (momentum : LatticeMomentum) :
    v32SpatialArtifact momentum = c19SpatialRemainder momentum := by
  rw [v32SpatialArtifact, c19_symbol_exact_quadratic_plus_remainder]
  ring

theorem v32_spatial_artifact_bound
    (momentum : LatticeMomentum) (radius : ℝ) (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (hsmall : 2 * radius ≤ 1) :
    |v32SpatialArtifact momentum| ≤ 40 * radius^4 := by
  rw [v32_spatial_artifact_is_c19_remainder]
  exact c19_spatial_remainder_bound momentum radius hradius hmomentum hsmall

/-- Geometry-only fourth moment of the C19 bond phases. -/
def v32C19FourthPhaseMoment (momentum : LatticeMomentum) : ℝ :=
  ∑ direction,
    c19Weight direction * c19PhaseAngle momentum direction ^ 4

/-- The leading (quartic) spatial correction is rotationally isotropic. -/
theorem v32_c19_fourth_phase_moment_isotropic (momentum : LatticeMomentum) :
    v32C19FourthPhaseMoment momentum =
      12 * (momentum 0 ^ 2 + momentum 1 ^ 2 + momentum 2 ^ 2) ^ 2 := by
  simp [v32C19FourthPhaseMoment, c19Weight, c19PhaseAngle,
    c19Coord, Fin.sum_univ_succ]
  ring

/-- Geometry-only sixth moment, used to locate the first directional split. -/
def v32C19SixthPhaseMoment (momentum : LatticeMomentum) : ℝ :=
  ∑ direction,
    c19Weight direction * c19PhaseAngle momentum direction ^ 6

def v32AxisMomentum (r : ℝ) : LatticeMomentum
  | 0 => r
  | _ => 0

def v32FaceDiagonalMomentum (r : ℝ) : LatticeMomentum
  | 0 => r
  | 1 => r
  | _ => 0

theorem v32_c19_sixth_axis_moment (r : ℝ) :
    v32C19SixthPhaseMoment (v32AxisMomentum r) = 12 * r^6 := by
  simp [v32C19SixthPhaseMoment, v32AxisMomentum, c19Weight,
    c19PhaseAngle, c19Coord, Fin.sum_univ_succ]
  ring

theorem v32_c19_sixth_face_diagonal_moment (r : ℝ) :
    v32C19SixthPhaseMoment (v32FaceDiagonalMomentum r) = 144 * r^6 := by
  simp [v32C19SixthPhaseMoment, v32FaceDiagonalMomentum, c19Weight,
    c19PhaseAngle, c19Coord, Fin.sum_univ_succ]
  ring

/-- Equal-norm axis and face-diagonal directions agree through fourth order
but split at sixth order.  The explicit integer witness avoids a numerical
or fitted anisotropy claim. -/
theorem v32_c19_sixth_order_anisotropy_witness :
    v32C19SixthPhaseMoment (v32AxisMomentum (Real.sqrt 2)) = 96 ∧
    v32C19SixthPhaseMoment (v32FaceDiagonalMomentum 1) = 144 ∧
    (Real.sqrt 2)^2 = 2 * (1 : ℝ)^2 := by
  have hsqrt : (Real.sqrt 2)^2 = (2 : ℝ) := Real.sq_sqrt (by norm_num)
  constructor
  · rw [v32_c19_sixth_axis_moment]
    rw [show (Real.sqrt 2)^6 = ((Real.sqrt 2)^2)^3 by ring, hsqrt]
    norm_num
  constructor
  · rw [v32_c19_sixth_face_diagonal_moment]
    norm_num
  rw [hsqrt]
  norm_num

structure V32LatticeArtifactCertificate : Prop where
  exactArtifact : ∀ momentum,
    v32SpatialArtifact momentum = c19SpatialRemainder momentum
  fourthOrderEnvelope : ∀ momentum radius,
    0 ≤ radius → (∀ axis, |momentum axis| ≤ radius) →
      2 * radius ≤ 1 → |v32SpatialArtifact momentum| ≤ 40 * radius^4
  artifactZeroAtRest : v32SpatialArtifact (fun _ => 0) = 0
  leadingFourthMomentIsotropic : ∀ momentum,
    v32C19FourthPhaseMoment momentum =
      12 * (momentum 0 ^ 2 + momentum 1 ^ 2 + momentum 2 ^ 2) ^ 2
  sixthAxisMoment : ∀ r,
    v32C19SixthPhaseMoment (v32AxisMomentum r) = 12 * r^6
  sixthFaceDiagonalMoment : ∀ r,
    v32C19SixthPhaseMoment (v32FaceDiagonalMomentum r) = 144 * r^6

theorem v32_lattice_artifacts : V32LatticeArtifactCertificate where
  exactArtifact := v32_spatial_artifact_is_c19_remainder
  fourthOrderEnvelope := v32_spatial_artifact_bound
  artifactZeroAtRest := by
    rw [v32_spatial_artifact_is_c19_remainder]
    exact c19_spatial_remainder_zero_at_origin
  leadingFourthMomentIsotropic := v32_c19_fourth_phase_moment_isotropic
  sixthAxisMoment := v32_c19_sixth_axis_moment
  sixthFaceDiagonalMoment := v32_c19_sixth_face_diagonal_moment

#print axioms v32_lattice_artifacts

end
end LeanDirac
