import LeanDirac.Dirac006ActionTransport

namespace LeanDirac

/-!
# DIRAC-009 closed native Clifford gate

This certificate differs from the superseded algebraic scaffold in one
essential way: the native Clifford operator is connected to the declared
positive action by the exact nondegenerate Bloch bridge.  The certificate
does not identify cube parity with physical mass.  That identification and
the full Dirac symbol belong to DIRAC-010.
-/

structure Dirac009ClosedCertificate : Prop where
  candidateAction : Dirac002Certificate
  reciprocalVariation : ∀
      (state : State (PeriodicCubicSite 4))
      (velocity : Velocity (PeriodicCubicSite 4)),
    IncidenceLocalTwoFieldVariationCertificate chi0 chiKineticNormalization
      state velocity (periodicC19Geometry 4)
  completeTransport : Dirac006ActionTransportCertificate
  actionToIncidence : Dirac009ActionBridgeCertificate
  gammaSquare : ∀ axis amplitude,
    cubeIncidenceGamma axis (cubeIncidenceGamma axis amplitude) = amplitude
  gammaAnticommutation : ∀ axis other, axis ≠ other → ∀ amplitude,
    cubeIncidenceGamma axis (cubeIncidenceGamma other amplitude) =
      fun corner =>
        -cubeIncidenceGamma other
          (cubeIncidenceGamma axis amplitude) corner
  flavorInvolution : ∀ amplitude,
    cubeFlavor (cubeFlavor amplitude) = amplitude
  flavorCommutes : ∀ axis amplitude,
    cubeFlavor (cubeIncidenceGamma axis amplitude) =
      cubeIncidenceGamma axis (cubeFlavor amplitude)
  twoRetainedCopies :
    Function.Injective cubeEmbedMinus ∧ Function.Injective cubeEmbedPlus

theorem dirac009_closed_native_clifford :
    Dirac009ClosedCertificate where
  candidateAction := dirac002_candidate_action_and_geometry
  reciprocalVariation := by
    intro state velocity
    exact incidence_c19_variations_from_actual_action
      chi0 chiKineticNormalization state velocity (periodicC19Geometry 4)
  completeTransport := dirac006_complete_action_transport
  actionToIncidence := dirac009_action_bridge_certificate
  gammaSquare := cube_incidence_gamma_square
  gammaAnticommutation := cube_incidence_gamma_anticommute
  flavorInvolution := cube_flavor_involution
  flavorCommutes := cube_flavor_commutes_gamma
  twoRetainedCopies :=
    ⟨cube_embed_minus_injective, cube_embed_plus_injective⟩

end LeanDirac

#print axioms LeanDirac.dirac009_closed_native_clifford
