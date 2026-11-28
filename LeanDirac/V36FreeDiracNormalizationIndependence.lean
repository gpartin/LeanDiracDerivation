import LeanDirac.V36FrameworkAndNormalizationLock

/-!
# V3.6 Stage 3: revised-action Dirac inheritance

The existing finite-periodic tangent theorem is first specialized to the live
geometric action `B = 1`, then to the two controls.  The common limit contains
no candidate `B` value, so the cumulative free Dirac theorem survives the
action revision without changing its operator.
-/

namespace LeanDirac

noncomputable section
open Filter Topology

structure V36FreeDiracNormalizationIndependenceCertificate : Prop where
  fullFreeDirac : CompleteFreeDiracCertificate
  liveB : chiKineticNormalization = v36EqualMetricB
  equalMetric : ∀ {L : ℕ} [NeZero L] {momentum : LatticeMomentum}
    (r : ℝ), r ≠ 0 →
    ∀ (mode : PeriodicC19FourierMode L momentum)
      (position velocity : UniformCell2),
      Tendsto
        (fun eps : ℝ => periodicC19LagrangianTangentQuotient
          r chiKineticNormalization eps mode position velocity)
        (nhds 0)
        (nhds (2 * periodicSiteCount L *
          uniformVacuumModalLagrangian r momentum position velocity))
  legacy : ∀ {L : ℕ} [NeZero L] {momentum : LatticeMomentum}
    (r : ℝ), r ≠ 0 →
    ∀ (mode : PeriodicC19FourierMode L momentum)
      (position velocity : UniformCell2),
      Tendsto
        (fun eps : ℝ => periodicC19LagrangianTangentQuotient
          r v36LegacyB eps mode position velocity)
        (nhds 0)
        (nhds (2 * periodicSiteCount L *
          uniformVacuumModalLagrangian r momentum position velocity))
  c19Weight : ∀ {L : ℕ} [NeZero L] {momentum : LatticeMomentum}
    (r : ℝ), r ≠ 0 →
    ∀ (mode : PeriodicC19FourierMode L momentum)
      (position velocity : UniformCell2),
      Tendsto
        (fun eps : ℝ => periodicC19LagrangianTangentQuotient
          r v36C19WeightB eps mode position velocity)
        (nhds 0)
        (nhds (2 * periodicSiteCount L *
          uniformVacuumModalLagrangian r momentum position velocity))

theorem v36_free_dirac_normalization_independence :
    V36FreeDiracNormalizationIndependenceCertificate where
  fullFreeDirac := complete_first_principles_free_dirac
  liveB := v36_live_foundation_is_equal_metric_pair.2
  equalMetric := fun r hr mode position velocity =>
    v32_free_transverse_tangent_uniform_in_B
      r chiKineticNormalization hr mode position velocity
  legacy := fun r hr mode position velocity =>
    v32_free_transverse_tangent_uniform_in_B
      r v36LegacyB hr mode position velocity
  c19Weight := fun r hr mode position velocity =>
    v32_free_transverse_tangent_uniform_in_B
      r v36C19WeightB hr mode position velocity

#print axioms v36_free_dirac_normalization_independence

end
end LeanDirac

