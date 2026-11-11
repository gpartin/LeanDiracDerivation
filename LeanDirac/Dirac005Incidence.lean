import LeanDirac.Dirac004Incidence
import LeanDirac.Dirac005Stabilized

namespace LeanDirac

/-!
DIRAC-005 compatibility for the incidence action.  On the period-two prepared
cell, each positive face direction and its negative partner reach the same
corner with opposite incidence sign.  The added residual therefore vanishes
without constraining the prepared amplitudes.  Its nonzero content appears in
the surrounding-cell Bloch response used at DIRAC-009.
-/

noncomputable section

theorem cube_chi_incidence_residual_zero
    (state : State CubeCorner) (corner : CubeCorner) :
    chiIncidenceResidual chi0 state cubeC19Geometry corner = 0 := by
  funext component
  fin_cases corner <;>
    simp [chiIncidenceResidual, cubeC19Geometry, cubeC19Neighbor,
      c19FaceIncidenceSign, Fin.sum_univ_succ] <;> ring

theorem cube_chi_incidence_energy_zero (state : State CubeCorner) :
    chiIncidenceEnergy chi0 state cubeC19Geometry = 0 := by
  unfold chiIncidenceEnergy chiIncidenceSiteEnergy
  apply Finset.sum_eq_zero
  intro corner _
  rw [cube_chi_incidence_residual_zero]
  simp [c3EuclideanSq]

theorem cube_chi_incidence_first_variation_zero
    (state direction : State CubeCorner) :
    chiIncidenceEnergyFirstVariation chi0 state direction cubeC19Geometry = 0 := by
  have hderiv := chi_incidence_energy_hasDerivAt
    chi0 state direction cubeC19Geometry
  have heq :
      (fun eps : ℝ =>
        chiIncidenceEnergy chi0 (stateLine state direction eps)
          cubeC19Geometry) =
      (fun _ : ℝ => 0) := by
    funext eps
    exact cube_chi_incidence_energy_zero (stateLine state direction eps)
  rw [heq] at hderiv
  have hconst : AnalyticHasDerivAt (fun _ : ℝ => 0) 0 0 :=
    hasDerivAt_const (𝕜 := ℝ) (0 : ℝ) (0 : ℝ)
  exact hderiv.unique hconst

theorem cube_incidence_force_matches_stabilized
    (state direction : State CubeCorner) :
    incidenceC19LagrangianStateFirstVariation chi0 state direction
        cubeC19Geometry =
      stabilizedC19LagrangianStateFirstVariation chi0 state direction
        cubeC19Geometry := by
  unfold incidenceC19LagrangianStateFirstVariation
  rw [cube_chi_incidence_first_variation_zero]
  ring

structure IncidencePreparedSectorCertificate : Prop where
  stabilizedPreparedSector : StabilizedPreparedSectorCertificate
  incidenceResidualZero : ∀ state corner,
    chiIncidenceResidual chi0 state cubeC19Geometry corner = 0
  incidenceForceZero : ∀ state direction,
    chiIncidenceEnergyFirstVariation chi0 state direction cubeC19Geometry = 0
  incidenceForceMatchesStabilized : ∀ state direction,
    incidenceC19LagrangianStateFirstVariation chi0 state direction
        cubeC19Geometry =
      stabilizedC19LagrangianStateFirstVariation chi0 state direction
        cubeC19Geometry

theorem dirac005_incidence_prepared_sector :
    IncidencePreparedSectorCertificate where
  stabilizedPreparedSector := dirac005_stabilized_prepared_sector
  incidenceResidualZero := cube_chi_incidence_residual_zero
  incidenceForceZero := cube_chi_incidence_first_variation_zero
  incidenceForceMatchesStabilized := cube_incidence_force_matches_stabilized

end
end LeanDirac
