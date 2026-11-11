import LeanDirac.Dirac009

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-!
# The nondegenerate action-to-incidence bridge

`CubeCorner` is a period-two unit cell.  On that cell the positive and
negative face neighbours coincide, so a centred odd incidence residual is
identically zero.  The missing information is the phase carried when a face
crosses into the neighbouring unit cell.

This file keeps that surrounding-cell phase explicit.  The two functions
`cellForwardBlochPhase` and `cellBackwardBlochPhase` are the first-order
Bloch lift of the *same* face terms occurring in `chiIncidenceResidual`.
No gamma matrix or spinor is supplied.  The theorem
`prepared_action_bloch_bridge` proves that the linear coefficient of the
native residual is, after the already proved prepared-field gauge, precisely
the cubical incidence operator.
-/

noncomputable section
open Complex BigOperators

def cubeAxisBit (axis : Fin 3) (corner : CubeCorner) : Nat :=
  match axis with
  | 0 => cellBit0 corner
  | 1 => cellBit1 corner
  | _ => cellBit2 corner

def cellForwardBlochPhase
    (axis : Fin 3) (corner : CubeCorner) (eps : ℝ)
    (momentum : Fin 3 → ℝ) : ℂ :=
  if cubeAxisBit axis corner = 0 then 1
  else 1 + (eps * momentum axis : ℝ) * I

def cellBackwardBlochPhase
    (axis : Fin 3) (corner : CubeCorner) (eps : ℝ)
    (momentum : Fin 3 → ℝ) : ℂ :=
  if cubeAxisBit axis corner = 0 then
    1 - (eps * momentum axis : ℝ) * I
  else 1

def preparedBlochFacePhase
    (corner : CubeCorner) (neighbor : C19Neighbor) (eps : ℝ)
    (momentum : Fin 3 → ℝ) : ℂ :=
  match neighbor.val with
  | 0 => cellForwardBlochPhase 0 corner eps momentum
  | 1 => cellBackwardBlochPhase 0 corner eps momentum
  | 2 => cellForwardBlochPhase 1 corner eps momentum
  | 3 => cellBackwardBlochPhase 1 corner eps momentum
  | 4 => cellForwardBlochPhase 2 corner eps momentum
  | 5 => cellBackwardBlochPhase 2 corner eps momentum
  | _ => 1

def preparedComplexChiBackground : State CubeCorner where
  psi := fun _ _ => 0
  chi := fun corner component =>
    (chi0 : ℂ) * companion corner component

/-!
This is the exact six-face factor in the positive square, restricted to the
prepared complex-chi background and to one scalar component of
`chi wedge Psi`.  The two neighbours of a corner have the same within-cell
corner but different surrounding-cell phase.  Therefore they must not be
identified before the Bloch reduction.
-/
def preparedActionBlochResidual
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    chiIncidenceComponentSymbol chi0 preparedComplexChiBackground
      cubeC19Geometry
      (fun source neighbor =>
        preparedBlochFacePhase source neighbor eps momentum)
      amplitude corner

theorem prepared_action_bloch_residual_expanded
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    preparedActionBlochResidual eps momentum amplitude =
      fun corner =>
        ∑ axis : Fin 3,
          ((3 / 2 : ℝ) : ℂ) * preparedCompanionEdgeOverlap axis corner *
            (cellForwardBlochPhase axis corner eps momentum -
              cellBackwardBlochPhase axis corner eps momentum) *
            amplitude (cubeIncidenceFlip axis corner) := by
  funext corner
  fin_cases corner <;>
    simp [preparedActionBlochResidual, chiIncidenceComponentSymbol,
      preparedComplexChiBackground, preparedBlochFacePhase,
      cubeC19Geometry, cubeC19Neighbor, c19FaceIncidenceSign,
      cellForwardBlochPhase, cellBackwardBlochPhase, cubeAxisBit,
      preparedCompanionEdgeOverlap, inner, chi0, Fin.sum_univ_succ,
      cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2] <;>
    ring

def preparedActionLinkTransport
    (axis : Fin 3) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    (3 : ℂ) * preparedCompanionEdgeOverlap axis corner *
      amplitude (cubeIncidenceFlip axis corner)

def preparedLinkLinear
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    ∑ axis : Fin 3,
      (momentum axis : ℂ) * preparedActionLinkTransport axis amplitude corner

theorem prepared_action_bloch_residual_linear
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    preparedActionBlochResidual eps momentum amplitude =
      fun corner =>
        (((eps / 2 : ℝ) : ℂ) * I) *
          preparedLinkLinear momentum amplitude corner := by
  rw [prepared_action_bloch_residual_expanded]
  funext corner
  fin_cases corner <;>
    simp [preparedLinkLinear,
      preparedActionLinkTransport, cellForwardBlochPhase,
      cellBackwardBlochPhase, cubeAxisBit, cubeIncidenceFlip,
      cellBit0, cellBit1, cellBit2, Fin.sum_univ_succ] <;>
    ring

theorem gauge_cell8_add (left right : Cell8) :
    gaugeCell8 (fun corner => left corner + right corner) =
      fun corner => gaugeCell8 left corner + gaugeCell8 right corner := by
  funext corner
  simp [gaugeCell8]
  ring

theorem gauge_cell8_smul (scale : ℂ) (amplitude : Cell8) :
    gaugeCell8 (fun corner => scale * amplitude corner) =
      fun corner => scale * gaugeCell8 amplitude corner := by
  funext corner
  simp [gaugeCell8]
  ring

theorem prepared_action_link_eq_sqrt_three_link
    (axis : Fin 3) (amplitude : Cell8) :
    preparedActionLinkTransport axis amplitude =
      fun corner =>
        (Real.sqrt 3 : ℂ) *
          preparedChiLinkTransport axis amplitude corner := by
  have hsqrt : ((Real.sqrt 3 : ℝ) : ℂ)^2 = 3 := by
    exact_mod_cast sqrt_three_sq
  funext corner
  simp only [preparedActionLinkTransport, preparedChiLinkTransport]
  rw [← hsqrt]
  ring

def preparedChiLinkLinear
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    ∑ axis : Fin 3,
      (momentum axis : ℂ) * preparedChiLinkTransport axis amplitude corner

theorem prepared_link_linear_eq_sqrt_three_chi_link
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    preparedLinkLinear momentum amplitude =
      fun corner =>
        (Real.sqrt 3 : ℂ) * preparedChiLinkLinear momentum amplitude corner := by
  funext corner
  simp [preparedLinkLinear, preparedChiLinkLinear, Fin.sum_univ_succ]
  rw [congrFun
      (prepared_action_link_eq_sqrt_three_link 0 amplitude) corner,
    congrFun (prepared_action_link_eq_sqrt_three_link 1 amplitude) corner,
    congrFun (prepared_action_link_eq_sqrt_three_link 2 amplitude) corner]
  ring

theorem prepared_chi_link_linear_gauge_is_incidence
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    gaugeCell8 (preparedChiLinkLinear momentum (ungaugeCell8 amplitude)) =
      cubeIncidenceLinear
        (momentum 0) (momentum 1) (momentum 2) amplitude := by
  funext corner
  unfold gaugeCell8 preparedChiLinkLinear cubeIncidenceLinear
  simp only [Fin.sum_univ_succ]
  simp
  change
    preparedCubeGaugePhase corner *
        ((momentum 0 : ℂ) *
            preparedChiLinkTransport 0 (ungaugeCell8 amplitude) corner +
          ((momentum 1 : ℂ) *
              preparedChiLinkTransport 1 (ungaugeCell8 amplitude) corner +
            (momentum 2 : ℂ) *
              preparedChiLinkTransport 2 (ungaugeCell8 amplitude) corner)) = _
  rw [show
      preparedCubeGaugePhase corner *
          ((momentum 0 : ℂ) *
              preparedChiLinkTransport 0 (ungaugeCell8 amplitude) corner +
            ((momentum 1 : ℂ) *
                preparedChiLinkTransport 1 (ungaugeCell8 amplitude) corner +
              (momentum 2 : ℂ) *
                preparedChiLinkTransport 2 (ungaugeCell8 amplitude) corner)) =
        (momentum 0 : ℂ) *
            gaugeCell8
              (preparedChiLinkTransport 0 (ungaugeCell8 amplitude)) corner +
          ((momentum 1 : ℂ) *
              gaugeCell8
                (preparedChiLinkTransport 1 (ungaugeCell8 amplitude)) corner +
            (momentum 2 : ℂ) *
              gaugeCell8
                (preparedChiLinkTransport 2 (ungaugeCell8 amplitude)) corner) by
      simp [gaugeCell8]; ring]
  rw [prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence,
    prepared_chi_link_gauge_is_cube_incidence]
  ring

theorem prepared_link_linear_gauge_is_incidence
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    gaugeCell8 (preparedLinkLinear momentum (ungaugeCell8 amplitude)) =
      fun corner =>
        (Real.sqrt 3 : ℂ) *
          cubeIncidenceLinear
            (momentum 0) (momentum 1) (momentum 2) amplitude corner := by
  rw [prepared_link_linear_eq_sqrt_three_chi_link]
  rw [gauge_cell8_smul]
  rw [prepared_chi_link_linear_gauge_is_incidence]

/-!
The central theorem missing from the previous DIRAC-009 certificate.  Its
left side is the surrounding-cell Bloch coefficient of the native positive
square.  Its right side is the incidence operator.  At `eps = 0` the
residual vanishes; its exact first-order coefficient is nonzero and no
period-two neighbour identification has been made.
-/
theorem prepared_action_bloch_bridge
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    gaugeCell8
        (preparedActionBlochResidual eps momentum
          (ungaugeCell8 amplitude)) =
      fun corner =>
        (((eps / 2 : ℝ) : ℂ) * I) * (Real.sqrt 3 : ℂ) *
          cubeIncidenceLinear
            (momentum 0) (momentum 1) (momentum 2) amplitude corner := by
  rw [prepared_action_bloch_residual_linear]
  rw [gauge_cell8_smul]
  rw [prepared_link_linear_gauge_is_incidence]
  funext corner
  ring

theorem prepared_action_bloch_bridge_zero
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    preparedActionBlochResidual 0 momentum amplitude = 0 := by
  rw [prepared_action_bloch_residual_linear]
  funext corner
  simp

theorem prepared_action_bloch_bridge_squares_to_scalar
    (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    cubeIncidenceLinear
        (momentum 0) (momentum 1) (momentum 2)
        (cubeIncidenceLinear
          (momentum 0) (momentum 1) (momentum 2) amplitude) =
      fun corner =>
        ((momentum 0 : ℂ)^2 + (momentum 1 : ℂ)^2 +
          (momentum 2 : ℂ)^2) * amplitude corner := by
  exact cube_incidence_linear_square
    (momentum 0) (momentum 1) (momentum 2) amplitude

structure Dirac009ActionBridgeCertificate : Prop where
  exactNativeBlochBridge : ∀ (eps : ℝ) (momentum : Fin 3 → ℝ)
      (amplitude : Cell8),
    gaugeCell8
        (preparedActionBlochResidual eps momentum
          (ungaugeCell8 amplitude)) =
      fun corner =>
        (((eps / 2 : ℝ) : ℂ) * I) * (Real.sqrt 3 : ℂ) *
          cubeIncidenceLinear
            (momentum 0) (momentum 1) (momentum 2) amplitude corner
  zeroBackgroundResidual : ∀ (momentum : Fin 3 → ℝ)
      (amplitude : Cell8),
    preparedActionBlochResidual 0 momentum amplitude = 0
  nativeSquare : ∀ (momentum : Fin 3 → ℝ) (amplitude : Cell8),
    cubeIncidenceLinear
        (momentum 0) (momentum 1) (momentum 2)
        (cubeIncidenceLinear
          (momentum 0) (momentum 1) (momentum 2) amplitude) =
      fun corner =>
        ((momentum 0 : ℂ)^2 + (momentum 1 : ℂ)^2 +
          (momentum 2 : ℂ)^2) * amplitude corner

theorem dirac009_action_bridge_certificate :
    Dirac009ActionBridgeCertificate where
  exactNativeBlochBridge := prepared_action_bloch_bridge
  zeroBackgroundResidual := prepared_action_bloch_bridge_zero
  nativeSquare := prepared_action_bloch_bridge_squares_to_scalar

end
end LeanDirac

#print axioms LeanDirac.prepared_action_bloch_bridge
#print axioms LeanDirac.dirac009_action_bridge_certificate
