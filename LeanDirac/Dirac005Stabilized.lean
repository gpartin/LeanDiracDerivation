import LeanDirac.Dirac004Stabilized
import LeanDirac.Dirac005

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-!
DIRAC-005 compatibility of the covariance correction with the declared
already-running equal-amplitude cube background.  The older arbitrary-
amplitude aligned sector is not reused as an invariant sector for the changed
action.
-/

noncomputable section
open Complex

def preparedRunningState (A R : ℝ) : State CubeCorner where
  psi corner component := (A : ℂ) * tetraCorner corner component
  chi corner component := (R : ℂ) * companion corner component

theorem prepared_running_covariance_residual_zero
    (A R : ℝ) (corner : CubeCorner) :
    chiCovarianceShapeResidual chi0 (preparedRunningState A R)
      cubeC19Geometry corner = 0 := by
  funext component
  fin_cases corner <;> fin_cases component <;>
    simp [chiCovarianceShapeResidual, chiCovarianceAction,
      preparedRunningState, cubeC19Geometry, cubeC19Neighbor,
      cellFlip0, cellFlip1, cellFlip2, c19Weight, inner, wedge, companion,
      tetraCorner, tetraRay0, tetraRay1, tetraRay2, tetraRay3,
      c3Of, c3Neg, omega, omegaBar, tetraA, tetraB,
      Fin.sum_univ_succ, chi0] <;>
    field_simp [sqrt_three_ne_zero] <;>
    simp <;> ring_nf <;>
    simp only [map_ofNat] <;> norm_num
  all_goals
    right
    have htwo : ((Real.sqrt 2 : ℂ)^2) = 2 := by
      exact_mod_cast sqrt_two_sq
    have hthree : ((Real.sqrt 3 : ℂ)^2) = 3 := by
      exact_mod_cast sqrt_three_sq
    rw [htwo, hthree]
    apply Complex.ext <;> norm_num <;>
      nlinarith [sqrt_three_sq, sqrt_two_sq, sqrt_three_pow_four,
        sqrt_three_pow_five, sqrt_three_pow_six, sqrt_three_pow_eight,
        sqrt_two_pow_four]

theorem prepared_running_shape_energy_zero (A R : ℝ) :
    chiCovarianceShapeEnergy chi0 (preparedRunningState A R)
      cubeC19Geometry = 0 := by
  unfold chiCovarianceShapeEnergy chiCovarianceShapeSiteEnergy
    c3EuclideanSq
  apply Finset.sum_eq_zero
  intro corner _
  rw [prepared_running_covariance_residual_zero A R corner]
  simp

theorem prepared_running_shape_first_variation_zero
    (A R : ℝ) (direction : State CubeCorner) :
    chiCovarianceShapeFirstVariation chi0
      (preparedRunningState A R) direction cubeC19Geometry = 0 := by
  let f : ℝ → ℝ := fun eps => chiCovarianceShapeEnergy chi0
    (stateLine (preparedRunningState A R) direction eps) cubeC19Geometry
  have hmin : IsLocalMin f 0 := by
    apply Filter.Eventually.of_forall
    intro eps
    have hnonneg := chi_covariance_shape_energy_nonnegative chi0
      (stateLine (preparedRunningState A R) direction eps) cubeC19Geometry
    have hzero := prepared_running_shape_energy_zero A R
    have hf0 : f 0 = 0 := by
      simpa [f, stateLine, c3Line, c3Add, c3ScaleReal] using hzero
    rw [hf0]
    exact hnonneg
  have hderiv := chi_covariance_shape_hasDerivAt chi0
    (preparedRunningState A R) direction cubeC19Geometry
  exact hmin.hasDerivAt_eq_zero hderiv

theorem prepared_running_stabilized_force_eq_base
    (A R : ℝ) (direction : State CubeCorner) :
    stabilizedC19LagrangianStateFirstVariation chi0
        (preparedRunningState A R) direction cubeC19Geometry =
      lagrangianStateFirstVariation chi0
        (preparedRunningState A R) direction (c19Bond cubeC19Geometry) := by
  unfold stabilizedC19LagrangianStateFirstVariation
  rw [prepared_running_shape_first_variation_zero]
  ring

structure StabilizedPreparedSectorCertificate : Prop where
  amplitudeRecovery : ∀ amplitudes : NativeUnitCellAmplitudes,
    recoverEightSiteAmplitudes
        (alignedEightSiteStateFromAmplitudes amplitudes) = amplitudes
  preparedShapeResidualZero : ∀ A R,
    ∀ corner, chiCovarianceShapeResidual chi0 (preparedRunningState A R)
      cubeC19Geometry corner = 0
  preparedShapeForceZero : ∀ A R direction,
    chiCovarianceShapeFirstVariation chi0
      (preparedRunningState A R) direction cubeC19Geometry = 0
  stabilizedForceMatchesBase : ∀ A R direction,
    stabilizedC19LagrangianStateFirstVariation chi0
        (preparedRunningState A R) direction cubeC19Geometry =
      lagrangianStateFirstVariation chi0
        (preparedRunningState A R) direction (c19Bond cubeC19Geometry)

theorem dirac005_stabilized_prepared_sector :
    StabilizedPreparedSectorCertificate where
  amplitudeRecovery := recover_aligned_eight_site_amplitudes
  preparedShapeResidualZero := prepared_running_covariance_residual_zero
  preparedShapeForceZero := prepared_running_shape_first_variation_zero
  stabilizedForceMatchesBase := prepared_running_stabilized_force_eq_base

end
end LeanDirac
