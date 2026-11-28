import LeanDirac.V36CubicalVectorBivectorIsometry
import LeanDirac.V32Robustness

/-!
# V3.6 Stage 2: geometrically normalized action lock

V3.6 changes the live chi kinetic normalization while preserving the action's
two fields, potential, C19 geometry, and reciprocal variational structure.
The live action uses the equal-grade cubical Clifford metric, `B = 1`, hence
`kappa = 19`.  The previous value and the C19-weight value remain explicit
historical controls.
-/

namespace LeanDirac

noncomputable section

def v36BFromKappa (candidateKappa : ℝ) : ℝ := chi0 / candidateKappa

def v36LegacyKappa : ℝ := 1 / 63
def v36LegacyB : ℝ := 1197

/-- Candidate obtained if C19 multiplicity is assigned as the chi kinetic
weight.  This is a hypothesis under audit, not a promoted framework value. -/
def v36C19WeightKappa : ℝ := 1
def v36C19WeightB : ℝ := 19

/-- Live V3.6 values obtained from the common Euclidean Clifford metric. -/
def v36EqualMetricKappa : ℝ := 19
def v36EqualMetricB : ℝ := 1

def V36NormalizationRelation (candidateKappa candidateB : ℝ) : Prop :=
  candidateB * candidateKappa = chi0

theorem v36_legacy_pair_exact :
    v36LegacyKappa = 1 / 63 ∧ v36LegacyB = 1197 ∧
      V36NormalizationRelation v36LegacyKappa v36LegacyB := by
  norm_num [v36LegacyKappa, v36LegacyB, V36NormalizationRelation, chi0]

theorem v36_c19_weight_pair_exact :
    v36C19WeightKappa = 1 ∧ v36C19WeightB = 19 ∧
      V36NormalizationRelation v36C19WeightKappa v36C19WeightB := by
  norm_num [v36C19WeightKappa, v36C19WeightB,
    V36NormalizationRelation, chi0]

theorem v36_equal_metric_pair_exact :
    v36EqualMetricKappa = 19 ∧ v36EqualMetricB = 1 ∧
      V36NormalizationRelation v36EqualMetricKappa v36EqualMetricB := by
  norm_num [v36EqualMetricKappa, v36EqualMetricB,
    V36NormalizationRelation, chi0]

theorem v36_live_foundation_is_equal_metric_pair :
    kappa = v36EqualMetricKappa ∧
      chiKineticNormalization = v36EqualMetricB := by
  constructor
  · simpa [v36EqualMetricKappa] using kappa_eq_nineteen
  · simpa [v36EqualMetricB] using chi_kinetic_normalization_eq_one

theorem v36_live_action_is_B_one
    [Fintype Site] (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) :
    v36ResearchC19Lagrangian state velocity geometry =
      c19Lagrangian chi0 1 state velocity geometry :=
  v36_research_c19_lagrangian_uses_equal_grade_kinetic_weight
    state velocity geometry

structure V36FrameworkAndNormalizationLockCertificate : Prop where
  prior : V35CompleteCovariantObservableCertificate
  geometry : V36CubicalVectorBivectorIsometryCertificate
  freeDiracRobustness : V32RobustnessCertificate
  livePair :
    kappa = v36EqualMetricKappa ∧
      chiKineticNormalization = v36EqualMetricB
  liveAction : ∀ [Fintype Site] (state : State Site)
    (velocity : Velocity Site) (geometry : C19Lattice Site),
    v36ResearchC19Lagrangian state velocity geometry =
      c19Lagrangian chi0 1 state velocity geometry
  legacyRelation : V36NormalizationRelation v36LegacyKappa v36LegacyB
  c19WeightRelation :
    V36NormalizationRelation v36C19WeightKappa v36C19WeightB
  equalMetricRelation :
    V36NormalizationRelation v36EqualMetricKappa v36EqualMetricB

theorem v36_framework_and_normalization_lock :
    V36FrameworkAndNormalizationLockCertificate where
  prior := v35_complete_covariant_observables
  geometry := v36_cubical_vector_bivector_isometry
  freeDiracRobustness := v32_robustness
  livePair := v36_live_foundation_is_equal_metric_pair
  liveAction := v36_live_action_is_B_one
  legacyRelation := v36_legacy_pair_exact.2.2
  c19WeightRelation := v36_c19_weight_pair_exact.2.2
  equalMetricRelation := v36_equal_metric_pair_exact.2.2

#print axioms v36_framework_and_normalization_lock

end
end LeanDirac

