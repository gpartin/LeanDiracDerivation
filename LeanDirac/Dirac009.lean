import LeanDirac.ChiIncidenceVariation
import LeanDirac.Dirac009Geometry

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-!
# DIRAC-009 prepared-link algebraic scaffold

This module records the prepared complex-chi link algebra on the declared
eight-corner background.  It is deliberately not the action-to-incidence
proof.  That proof is isolated in `Dirac009ActionBridge`, which derives this
link as the surrounding-cell response of the positive action.  The gauge
below is not guessed: it is the explicit lift of
`preparedCubeGauge4`, whose relation to the measured prepared overlaps was
proved in `prepared_cube_connection_gauge_eq_incidence`.
-/

noncomputable section
open Complex

def preparedCubeGaugePhase (corner : CubeCorner) : ℂ :=
  match corner.val with
  | 0 | 1 | 2 => 1
  | 3 | 5 => Complex.I
  | 4 => -1
  | _ => -Complex.I

def preparedCubeGaugePhaseInv (corner : CubeCorner) : ℂ :=
  star (preparedCubeGaugePhase corner)

def gaugeCell8 (amplitude : Cell8) : Cell8 :=
  fun corner => preparedCubeGaugePhase corner * amplitude corner

def ungaugeCell8 (amplitude : Cell8) : Cell8 :=
  fun corner => preparedCubeGaugePhaseInv corner * amplitude corner

theorem prepared_cube_gauge_phase_unit (corner : CubeCorner) :
    preparedCubeGaugePhase corner * preparedCubeGaugePhaseInv corner = 1 := by
  fin_cases corner <;>
    simp [preparedCubeGaugePhase, preparedCubeGaugePhaseInv]

theorem gauge_ungauge_cell8 (amplitude : Cell8) :
    gaugeCell8 (ungaugeCell8 amplitude) = amplitude := by
  funext corner
  rw [show gaugeCell8 (ungaugeCell8 amplitude) corner =
      (preparedCubeGaugePhase corner * preparedCubeGaugePhaseInv corner) *
        amplitude corner by simp [gaugeCell8, ungaugeCell8, mul_assoc]]
  rw [prepared_cube_gauge_phase_unit]
  simp

def preparedCompanionEdgeOverlap
    (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  inner (companion corner) (companion (cubeIncidenceFlip axis corner))

def preparedCompanionExpectedEdgeOverlap
    (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  star (preparedCubeExpectedEdgeOverlap axis corner)

def planarCompanion (u : C3) : C3
  | 0 => -star (u 1)
  | 1 => star (u 0)
  | _ => 0

theorem inner_eq_overlap (u v : C3) : inner u v = overlap u v := by
  apply Complex.ext <;>
    simp [inner, overlap, Fin.sum_univ_succ]; ring

theorem inner_planar_companion
    (u v : C3) (hu : u 2 = 0) (hv : v 2 = 0) :
    inner (planarCompanion u) (planarCompanion v) =
      star (inner u v) := by
  apply Complex.ext <;>
    simp [inner, planarCompanion, Fin.sum_univ_succ, hu, hv] <;> ring

theorem prepared_companion_edge_overlap_values
    (axis : Fin 3) (corner : CubeCorner) :
    preparedCompanionEdgeOverlap axis corner =
      preparedCompanionExpectedEdgeOverlap axis corner := by
  unfold preparedCompanionEdgeOverlap
  rw [show companion corner = planarCompanion (tetraCorner corner) by rfl]
  rw [show companion (cubeIncidenceFlip axis corner) =
      planarCompanion (tetraCorner (cubeIncidenceFlip axis corner)) by rfl]
  rw [inner_planar_companion]
  · rw [inner_eq_overlap]
    change star (preparedCubeEdgeOverlap axis corner) =
      preparedCompanionExpectedEdgeOverlap axis corner
    rw [prepared_cube_edge_overlap_values]
    rfl
  · fin_cases corner <;>
      simp [tetraCorner, tetraRay0, tetraRay1, tetraRay2, tetraRay3,
        c3Of, c3Neg]
  · fin_cases axis <;> fin_cases corner <;>
      simp [cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
        tetraCorner, tetraRay0, tetraRay1, tetraRay2, tetraRay3,
        c3Of, c3Neg]

def preparedChiLinkTransport (axis : Fin 3) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    (Real.sqrt 3 : ℂ) * preparedCompanionEdgeOverlap axis corner *
      amplitude (cubeIncidenceFlip axis corner)

theorem sqrt_three_mul_tetraA : Real.sqrt 3 * tetraA = 1 := by
  simp [tetraA]

/-!
The measured companion-field overlap, after the independently derived cube
gauge, is exactly the signed incidence operator.  Thus the three matrices are
not inserted into the equations or into the readout.
-/
theorem prepared_chi_link_gauge_is_cube_incidence
    (axis : Fin 3) (amplitude : Cell8) :
    gaugeCell8 (preparedChiLinkTransport axis (ungaugeCell8 amplitude)) =
      cubeIncidenceGamma axis amplitude := by
  have hscale : (Real.sqrt 3 : ℂ) * (tetraA : ℂ) = 1 := by
    exact_mod_cast sqrt_three_mul_tetraA
  funext corner
  simp only [gaugeCell8, ungaugeCell8, preparedChiLinkTransport]
  rw [prepared_companion_edge_overlap_values]
  fin_cases axis <;> fin_cases corner <;>
    simp [
      preparedCompanionExpectedEdgeOverlap, preparedCubeExpectedEdgeOverlap,
      preparedCubeGaugePhase,
      preparedCubeGaugePhaseInv, cubeIncidenceGamma, cubeIncidenceSign,
      cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, hscale] <;>
    ring_nf; try rw [Complex.I_sq]; try rw [hscale]
  all_goals simp_all

theorem prepared_chi_links_square
    (axis : Fin 3) (amplitude : Cell8) :
    gaugeCell8
        (preparedChiLinkTransport axis
          (ungaugeCell8
            (gaugeCell8
              (preparedChiLinkTransport axis (ungaugeCell8 amplitude))))) =
      amplitude := by
  rw [prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence,
    cube_incidence_gamma_square]

theorem prepared_chi_links_anticommute
    (axis other : Fin 3) (haxis : axis ≠ other) (amplitude : Cell8) :
    gaugeCell8
        (preparedChiLinkTransport axis
          (ungaugeCell8
            (gaugeCell8
              (preparedChiLinkTransport other (ungaugeCell8 amplitude))))) =
      fun corner =>
        -gaugeCell8
          (preparedChiLinkTransport other
            (ungaugeCell8
              (gaugeCell8
                (preparedChiLinkTransport axis
                  (ungaugeCell8 amplitude))))) corner := by
  rw [prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence]
  exact cube_incidence_gamma_anticommute axis other haxis amplitude

/-!
The same native flavor involution commutes with the prepared-field links and
therefore splits the eight corner amplitudes into two invariant rank-four
copies.  Neither copy is deleted or declared to be the physical one here.
-/
theorem prepared_links_commute_flavor
    (axis : Fin 3) (amplitude : Cell8) :
    cubeFlavor
        (gaugeCell8
          (preparedChiLinkTransport axis (ungaugeCell8 amplitude))) =
      gaugeCell8
        (preparedChiLinkTransport axis
          (ungaugeCell8 (cubeFlavor amplitude))) := by
  rw [prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence]
  exact cube_flavor_commutes_gamma axis amplitude

/-!
This is deliberately an algebraic scaffold.  It records the prepared-link
algebra but does not claim that the link is the linearization of the action.
That missing implication is proved separately in `Dirac009ActionBridge`.
-/
structure Dirac009AlgebraicScaffold : Prop where
  candidateAction : Dirac002Certificate
  throughFirstOrderMap : ∀ (dt : ℝ) (hdt : dt ≠ 0),
    IncidenceFirstOrderMapCertificate dt hdt
  positiveNativeAction : ∀
      (state : State (PeriodicCubicSite 2))
      (velocity : Velocity (PeriodicCubicSite 2)),
    0 ≤ incidenceC19Hamiltonian chi0 chiKineticNormalization state velocity
      (periodicC19Geometry 2)
  reciprocalTwoFieldVariation : ∀
      (state : State (PeriodicCubicSite 2))
      (velocity : Velocity (PeriodicCubicSite 2)),
    IncidenceLocalTwoFieldVariationCertificate chi0 chiKineticNormalization
      state velocity (periodicC19Geometry 2)
  nativeSquare : ∀ axis amplitude,
    cubeIncidenceGamma axis (cubeIncidenceGamma axis amplitude) = amplitude
  nativeAnticommutator : ∀ axis other, axis ≠ other → ∀ amplitude,
    cubeIncidenceGamma axis (cubeIncidenceGamma other amplitude) =
      fun corner =>
        -cubeIncidenceGamma other
          (cubeIncidenceGamma axis amplitude) corner
  betaAnticommutator : ∀ axis amplitude,
    cubeParityBeta (cubeIncidenceGamma axis amplitude) =
      fun corner =>
        -cubeIncidenceGamma axis (cubeParityBeta amplitude) corner
  twoRankFourCopies :
    Function.Injective cubeEmbedMinus ∧ Function.Injective cubeEmbedPlus

theorem dirac009_algebraic_scaffold :
    Dirac009AlgebraicScaffold where
  candidateAction := dirac002_candidate_action_and_geometry
  throughFirstOrderMap := dirac008_incidence_complete_first_order_map
  positiveNativeAction := by
    intro state velocity
    exact incidence_c19_hamiltonian_nonnegative
      chi0 chiKineticNormalization state velocity (periodicC19Geometry 2)
      (by norm_num [chi0])
      (by rw [chi_kinetic_normalization_eq_1197]; norm_num)
  reciprocalTwoFieldVariation := by
    intro state velocity
    exact incidence_c19_variations_from_actual_action
      chi0 chiKineticNormalization state velocity (periodicC19Geometry 2)
  nativeSquare := cube_incidence_gamma_square
  nativeAnticommutator := cube_incidence_gamma_anticommute
  betaAnticommutator := cube_parity_beta_anticommutes
  twoRankFourCopies := ⟨cube_embed_minus_injective, cube_embed_plus_injective⟩

end
end LeanDirac

#print axioms LeanDirac.prepared_chi_link_gauge_is_cube_incidence
#print axioms LeanDirac.prepared_chi_links_anticommute
#print axioms LeanDirac.dirac009_algebraic_scaffold
