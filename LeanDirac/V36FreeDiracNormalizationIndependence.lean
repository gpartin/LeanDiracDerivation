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
open Complex Filter Topology

/-- The symmetric second-difference quotient of the live V3.6 research
action, with no free normalization parameter left in the definition. -/
def periodicV36ResearchLagrangianTangentQuotient
    {L : ℕ} [NeZero L] {momentum : LatticeMomentum}
    (eps : ℝ) (mode : PeriodicC19FourierMode L momentum)
    (position velocity : UniformCell2) : ℝ :=
  if eps = 0 then
    2 * periodicSiteCount L *
      uniformVacuumModalLagrangian chi0 momentum position velocity
  else
    (v36ResearchC19Lagrangian
        (periodicModeState chi0 eps mode position)
        (periodicModeVelocity eps mode velocity)
        (periodicC19Geometry L) +
      v36ResearchC19Lagrangian
        (periodicModeState chi0 (-eps) mode position)
        (periodicModeVelocity (-eps) mode velocity)
        (periodicC19Geometry L) -
      2 * v36ResearchC19Lagrangian
        (periodicModeState chi0 0 mode position)
        (periodicModeVelocity 0 mode velocity)
        (periodicC19Geometry L)) / eps^2

theorem periodic_v36_research_tangent_is_live_c19_tangent
    {L : ℕ} [NeZero L] {momentum : LatticeMomentum}
    (eps : ℝ) (mode : PeriodicC19FourierMode L momentum)
    (position velocity : UniformCell2) :
    periodicV36ResearchLagrangianTangentQuotient
        eps mode position velocity =
      periodicC19LagrangianTangentQuotient
        chi0 chiKineticNormalization eps mode position velocity := by
  rfl

/-- The complete action-to-Dirac conclusion with the live V3.6 action named
directly in the tangent term. -/
def V36LiveActionModeToFreeDirac : Prop :=
  ∀ {L : ℕ} [NeZero L] (momentum : LatticeMomentum)
    (mode : PeriodicC19FourierMode L momentum)
    (phase : UniformTwoChannelPhase) (direction : UniformCell2)
    (radius : ℝ), 0 ≤ radius →
    (∀ axis, |momentum axis| ≤ radius) →
    2 * radius ≤ 1 →
    Function.Injective (periodicFullNativeTangentEmbed mode) ∧
    periodicNativeTransverseTangentGenerator chi0
        (periodicFullNativeTangentEmbed mode phase) =
      periodicFullNativeTangentEmbed mode
        (uniformVacuumNativeGenerator chi0 momentum phase) ∧
    (∀ candidate : UniformTwoChannelPhase → CubicalCell4,
      (∀ nativePhase,
        cubicalStandardPosition (candidate nativePhase) = nativePhase.1) →
      (∀ nativePhase,
        candidate (uniformTwoChannelKleinGordonVector (chi0 : ℂ)
            (uniformContinuumPx momentum)
            (uniformContinuumPy momentum)
            (uniformContinuumPz momentum) nativePhase) =
          fun component =>
            -I * cubicalStandardHamiltonian (chi0 : ℂ)
              (uniformContinuumPx momentum)
              (uniformContinuumPy momentum)
              (uniformContinuumPz momentum)
              (candidate nativePhase) component) →
      candidate = cubicalPhaseReadout (chi0 : ℂ)
        (uniformContinuumPx momentum)
        (uniformContinuumPy momentum)
        (uniformContinuumPz momentum)) ∧
    (∀ eps longitudinal,
      potentialFirstVariation chi0
        (uniformTransverseState chi0 eps
          (periodicModeField mode (phase.1 0))
          (periodicModeField mode (phase.1 1)))
        (periodicLongitudinalPsiDirection longitudinal)
        (c19Bond (periodicC19Geometry L)) = 0) ∧
    (∀ chiDirection,
      AnalyticHasDerivAt
        (fun eps : ℝ =>
          potentialFirstVariation chi0
            (uniformTransverseState chi0 eps
              (periodicModeField mode (phase.1 0))
              (periodicModeField mode (phase.1 1)))
            (pureChiDirection chiDirection)
            (c19Bond (periodicC19Geometry L))) 0 0) ∧
    periodicTransverseGenerator chi0 (periodicModePhaseEmbed mode phase) =
      periodicModePhaseEmbed mode
        (uniformVacuumNativeGenerator chi0 momentum phase) ∧
    ((∀ (component : Fin 2) (site : PeriodicCubicSite L)
        (testAmplitude : ℂ),
      AnalyticHasDerivAt
        (fun eps : ℝ =>
          potentialFirstVariation chi0
            (periodicTransverseComponentState chi0 eps component
              (periodicModeField mode (phase.1 component)))
            (periodicTransverseComponentDirection component
              (periodicSingleSiteScalarDirection site testAmplitude))
            (c19Bond (periodicC19Geometry L)))
        (-uniformScalarRealPair
          (periodicModeField mode
            ((uniformVacuumModalAcceleration
              chi0 momentum phase.1) component) site)
          testAmplitude) 0) ∧
      Tendsto
          (fun eps : ℝ =>
            periodicV36ResearchLagrangianTangentQuotient
              eps mode phase.1 phase.2)
          (nhds 0)
          (nhds (2 * periodicSiteCount L *
            uniformVacuumModalLagrangian
              chi0 momentum phase.1 phase.2)) ∧
      (uniformCell2RealPair
          (uniformVacuumModalAcceleration chi0 momentum phase.1) direction +
        uniformVacuumModalPotentialVariation
          chi0 momentum phase.1 direction = 0)) ∧
    (uniformVacuumNativeGenerator chi0 momentum phase =
      (fun component =>
        (uniformVacuumContinuumGenerator chi0 momentum phase).1 component +
          (uniformNativeSpatialDefectPhase momentum phase).1 component,
       fun component =>
        (uniformVacuumContinuumGenerator chi0 momentum phase).2 component +
          (uniformNativeSpatialDefectPhase momentum phase).2 component)) ∧
    |c19SpatialRemainder momentum| ≤ 40 * radius^4 ∧
    cubicalCovariantDiracResidual chi0
        (uniformActionToStandardDirac chi0 momentum phase)
        (uniformActionToStandardDirac chi0 momentum
          (uniformVacuumContinuumGenerator chi0 momentum phase))
        (uniformPlaneWaveSpatialDerivative momentum
          (uniformActionToStandardDirac chi0 momentum phase)) =
      fun _ => 0

theorem v36_live_research_action_mode_to_free_dirac :
    V36LiveActionModeToFreeDirac := by
  intro L inst momentum mode phase direction radius hradius hmomentum hsmall
  simpa [periodicV36ResearchLagrangianTangentQuotient,
    periodicC19LagrangianTangentQuotient,
    v36ResearchC19Lagrangian] using
    (lfm_periodic_action_mode_to_free_dirac momentum
      chiKineticNormalization mode phase direction radius hradius
      hmomentum hsmall)

structure V36FreeDiracNormalizationIndependenceCertificate : Prop where
  fullFreeDirac : CompleteFreeDiracCertificate
  liveActionToFreeDirac : V36LiveActionModeToFreeDirac
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
  liveActionToFreeDirac := v36_live_research_action_mode_to_free_dirac
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
#print axioms v36_live_research_action_mode_to_free_dirac

end
end LeanDirac

