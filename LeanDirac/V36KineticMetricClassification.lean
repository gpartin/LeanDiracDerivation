import LeanDirac.V36FreeDiracNormalizationIndependence

/-!
# V3.6 Stage 4: historical controls and scoped uniqueness

Theorems in this module separate three statements that must not be conflated:
the Hodge identification is an isometry; a common inherited metric forces
`B = 1`; and a C19-weighted metric forces `B = 19`.  A common cubical turn
preserves the two-sector metric for every `B`, so that symmetry test alone
does not choose among the candidates.
-/

namespace LeanDirac

noncomputable section

def v36RelativeKineticMetric (B : ℝ) (psi chi : C3) : ℝ :=
  density psi + B * density chi

def v36CubeQuarterTurn (v : C3) : C3
  | 0 => -v 1
  | 1 => v 0
  | _ => v 2

theorem v36_cube_quarter_turn_preserves_density (v : C3) :
    density (v36CubeQuarterTurn v) = density v := by
  simp [density, v36CubeQuarterTurn, Fin.sum_univ_three]
  ring

theorem v36_common_cubic_turn_preserves_every_relative_weight
    (B : ℝ) (psi chi : C3) :
    v36RelativeKineticMetric B
        (v36CubeQuarterTurn psi) (v36CubeQuarterTurn chi) =
      v36RelativeKineticMetric B psi chi := by
  simp [v36RelativeKineticMetric, v36_cube_quarter_turn_preserves_density]

theorem v36_equal_inherited_metric_holds_at_B_one (v : C3) :
    v36EqualMetricB * density (cubicalVectorToBivector v) = density v := by
  simp [v36EqualMetricB, cube_vector_bivector_density_isometry]

theorem v36_equal_inherited_metric_forces_B_one
    (B : ℝ)
    (hmetric : ∀ v : C3,
      B * density (cubicalVectorToBivector v) = density v) :
    B = 1 := by
  have h := hmetric (basis 0)
  norm_num [density, cubicalVectorToBivector, basis,
    Fin.sum_univ_three, Complex.normSq] at h
  exact h

theorem v36_c19_weighted_metric_holds_at_B_nineteen (v : C3) :
    v36C19WeightB * density (cubicalVectorToBivector v) =
      chi0 * density v := by
  simp [v36C19WeightB, chi0, cube_vector_bivector_density_isometry]

theorem v36_c19_weighted_metric_forces_B_nineteen
    (B : ℝ)
    (hmetric : ∀ v : C3,
      B * density (cubicalVectorToBivector v) = chi0 * density v) :
    B = 19 := by
  have h := hmetric (basis 0)
  norm_num [density, cubicalVectorToBivector, basis, chi0,
    Fin.sum_univ_three, Complex.normSq] at h
  exact h

theorem v36_B_one_relation_forces_kappa_nineteen
    (candidateKappa : ℝ)
    (hrelation : V36NormalizationRelation candidateKappa 1) :
    candidateKappa = 19 := by
  norm_num [V36NormalizationRelation, chi0] at hrelation ⊢
  exact hrelation

theorem v36_B_nineteen_relation_forces_kappa_one
    (candidateKappa : ℝ)
    (hrelation : V36NormalizationRelation candidateKappa 19) :
    candidateKappa = 1 := by
  norm_num [V36NormalizationRelation, chi0] at hrelation ⊢
  linarith

structure V36KineticMetricClassificationCertificate : Prop where
  hodgeIsometry : ∀ v, density (cubicalVectorToBivector v) = density v
  equalMetricCandidate : ∀ v,
    v36EqualMetricB * density (cubicalVectorToBivector v) = density v
  equalMetricUnique : ∀ B,
    (∀ v : C3, B * density (cubicalVectorToBivector v) = density v) → B = 1
  c19MetricCandidate : ∀ v,
    v36C19WeightB * density (cubicalVectorToBivector v) =
      chi0 * density v
  c19MetricUnique : ∀ B,
    (∀ v : C3,
      B * density (cubicalVectorToBivector v) = chi0 * density v) → B = 19
  commonTurnAllB : ∀ B psi chi,
    v36RelativeKineticMetric B
        (v36CubeQuarterTurn psi) (v36CubeQuarterTurn chi) =
      v36RelativeKineticMetric B psi chi

theorem v36_kinetic_metric_classification :
    V36KineticMetricClassificationCertificate where
  hodgeIsometry := cube_vector_bivector_density_isometry
  equalMetricCandidate := v36_equal_inherited_metric_holds_at_B_one
  equalMetricUnique := v36_equal_inherited_metric_forces_B_one
  c19MetricCandidate := v36_c19_weighted_metric_holds_at_B_nineteen
  c19MetricUnique := v36_c19_weighted_metric_forces_B_nineteen
  commonTurnAllB := v36_common_cubic_turn_preserves_every_relative_weight

#print axioms v36_kinetic_metric_classification

end
end LeanDirac

