import LeanDirac.ChiIncidenceVariation
import LeanDirac.CubeIncidenceAlgebra

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-!
# Prepared-link algebra

This module records only the prepared complex-chi link algebra on the
declared eight-corner background.  The action-to-incidence proof is isolated
in `Dirac009ActionBridge`, which derives this link as the surrounding-cell
response of the positive action.  The gauge below is the explicit lift of
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

end
end LeanDirac

#print axioms LeanDirac.prepared_chi_link_gauge_is_cube_incidence
#print axioms LeanDirac.prepared_chi_links_anticommute
