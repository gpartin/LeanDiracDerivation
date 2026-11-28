import LeanDirac.V36RadialResponse

/-!
# V3.6 Stage 6: normalization decision surface

This stage records why the live action is selected geometrically rather than
numerically.  All three positive values satisfy the same free transverse
tangent theorem, so free Dirac cannot choose among them.  The common
Euclidean Clifford metric selects `B = 1`; the other values are controls with
different nonlinear chi response.  Electron mobility and an Einstein-limit
normalization remain independent validation obligations.
-/

namespace LeanDirac

noncomputable section

theorem v36_candidate_B_values_positive :
    0 < v36LegacyB ∧ 0 < v36C19WeightB ∧ 0 < v36EqualMetricB := by
  norm_num [v36LegacyB, v36C19WeightB, v36EqualMetricB]

theorem v36_candidate_B_values_pairwise_distinct :
    v36LegacyB ≠ v36C19WeightB ∧
      v36LegacyB ≠ v36EqualMetricB ∧
      v36C19WeightB ≠ v36EqualMetricB := by
  norm_num [v36LegacyB, v36C19WeightB, v36EqualMetricB]

theorem v36_candidate_kappa_values_pairwise_distinct :
    v36LegacyKappa ≠ v36C19WeightKappa ∧
      v36LegacyKappa ≠ v36EqualMetricKappa ∧
      v36C19WeightKappa ≠ v36EqualMetricKappa := by
  norm_num [v36LegacyKappa, v36C19WeightKappa, v36EqualMetricKappa]

structure V36NormalizationDecisionSurfaceCertificate : Prop where
  framework : V36FrameworkAndNormalizationLockCertificate
  diracIndependence : V36FreeDiracNormalizationIndependenceCertificate
  cubicalIsometry : V36CubicalVectorBivectorIsometryCertificate
  metricClassification : V36KineticMetricClassificationCertificate
  radialResponse : V36RadialResponseCertificate
  liveActionPair :
    kappa = v36EqualMetricKappa ∧
      chiKineticNormalization = v36EqualMetricB
  positiveCandidates :
    0 < v36LegacyB ∧ 0 < v36C19WeightB ∧ 0 < v36EqualMetricB
  distinctB :
    v36LegacyB ≠ v36C19WeightB ∧
      v36LegacyB ≠ v36EqualMetricB ∧
      v36C19WeightB ≠ v36EqualMetricB
  distinctKappa :
    v36LegacyKappa ≠ v36C19WeightKappa ∧
      v36LegacyKappa ≠ v36EqualMetricKappa ∧
      v36C19WeightKappa ≠ v36EqualMetricKappa

theorem v36_normalization_decision_surface :
    V36NormalizationDecisionSurfaceCertificate where
  framework := v36_framework_and_normalization_lock
  diracIndependence := v36_free_dirac_normalization_independence
  cubicalIsometry := v36_cubical_vector_bivector_isometry
  metricClassification := v36_kinetic_metric_classification
  radialResponse := v36_radial_response
  liveActionPair := v36_live_foundation_is_equal_metric_pair
  positiveCandidates := v36_candidate_B_values_positive
  distinctB := v36_candidate_B_values_pairwise_distinct
  distinctKappa := v36_candidate_kappa_values_pairwise_distinct

#print axioms v36_normalization_decision_surface

end
end LeanDirac

