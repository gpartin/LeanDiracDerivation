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

theorem infinite_c19_neighbor_reverse
    (site : InfiniteCubicSite) (direction : C19Neighbor) :
    infiniteC19Neighbor (infiniteC19Neighbor site direction)
        (c19Reverse direction) = site := by
  funext axis
  simp [infiniteC19Neighbor, c19_reverse_coord]

def infiniteC19Geometry : C19Lattice InfiniteCubicSite where
  neighbor := infiniteC19Neighbor
  neighbor_reverse := infinite_c19_neighbor_reverse

def periodTwoCellCoordinate
    (site : InfiniteCubicSite) (axis : Fin 3) : ℤ :=
  site axis / 2

def periodTwoCellPhase
    (momentum : Fin 3 → ℝ) (site : InfiniteCubicSite) : ℝ :=
  momentum 0 * (periodTwoCellCoordinate site 0 : ℝ) +
    momentum 1 * (periodTwoCellCoordinate site 1 : ℝ) +
    momentum 2 * (periodTwoCellCoordinate site 2 : ℝ)

def affineCellBlochPhase
    (eps : ℝ) (momentum : Fin 3 → ℝ)
    (site : InfiniteCubicSite) : ℂ :=
  1 + (eps * periodTwoCellPhase momentum site : ℝ) * I

def canonicalCornerSite (corner : CubeCorner) : InfiniteCubicSite
  | 0 => (cellBit0 corner : ℤ)
  | 1 => (cellBit1 corner : ℤ)
  | _ => (cellBit2 corner : ℤ)

@[simp] theorem canonical_corner_site_axis0 (corner : CubeCorner) :
    canonicalCornerSite corner 0 = (cellBit0 corner : ℤ) := rfl

@[simp] theorem canonical_corner_site_axis1 (corner : CubeCorner) :
    canonicalCornerSite corner 1 = (cellBit1 corner : ℤ) := rfl

@[simp] theorem canonical_corner_site_axis2 (corner : CubeCorner) :
    canonicalCornerSite corner 2 = (cellBit2 corner : ℤ) := rfl

@[simp] theorem c19_coord_face0 (axis : Fin 3) :
    c19Coord axis 0 = if axis = 0 then 1 else 0 := by
  fin_cases axis <;> rfl

@[simp] theorem c19_coord_face1 (axis : Fin 3) :
    c19Coord axis 1 = if axis = 0 then -1 else 0 := by
  fin_cases axis <;> rfl

@[simp] theorem c19_coord_face2 (axis : Fin 3) :
    c19Coord axis 2 = if axis = 1 then 1 else 0 := by
  fin_cases axis <;> rfl

@[simp] theorem c19_coord_face3 (axis : Fin 3) :
    c19Coord axis 3 = if axis = 1 then -1 else 0 := by
  fin_cases axis <;> rfl

@[simp] theorem c19_coord_face4 (axis : Fin 3) :
    c19Coord axis 4 = if axis = 2 then 1 else 0 := by
  fin_cases axis <;> rfl

@[simp] theorem c19_coord_face5 (axis : Fin 3) :
    c19Coord axis 5 = if axis = 2 then -1 else 0 := by
  fin_cases axis <;> rfl

theorem canonical_corner_has_declared_parity (corner : CubeCorner) :
    periodTwoCornerAt (canonicalCornerSite corner) = corner := by
  fin_cases corner <;>
    simp [periodTwoCornerAt, canonicalCornerSite, integerParityBit,
      cornerFromParity, cellBit0, cellBit1, cellBit2]

def preparedNativeBlochState
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    State InfiniteCubicSite where
  psi := fun site =>
    c3ScaleComplex
      (-(affineCellBlochPhase eps momentum site *
          amplitude (periodTwoCornerAt site)) / (chi0 : ℂ))
      (tetraCorner (periodTwoCornerAt site))
  chi := fun site =>
    c3ScaleComplex (chi0 : ℂ) (companion (periodTwoCornerAt site))

theorem prepared_native_bloch_area
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8)
    (site : InfiniteCubicSite) :
    wedge
        ((preparedNativeBlochState eps momentum amplitude).chi site)
        ((preparedNativeBlochState eps momentum amplitude).psi site) =
      c3ScaleComplex
        (affineCellBlochPhase eps momentum site *
          amplitude (periodTwoCornerAt site)) unitArea := by
  rw [show
      (preparedNativeBlochState eps momentum amplitude).chi site =
        c3ScaleComplex (chi0 : ℂ)
          (companion (periodTwoCornerAt site)) by rfl,
    show
      (preparedNativeBlochState eps momentum amplitude).psi site =
        c3ScaleComplex
          (-(affineCellBlochPhase eps momentum site *
              amplitude (periodTwoCornerAt site)) / (chi0 : ℂ))
          (tetraCorner (periodTwoCornerAt site)) by rfl,
    wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
  funext component
  fin_cases component <;>
    simp [c3ScaleComplex, c3Neg, unitArea, chi0] <;> ring

theorem period_two_corner_of_canonical_neighbor
    (corner : CubeCorner) (neighbor : C19Neighbor) :
    periodTwoCornerAt
        (infiniteC19Neighbor (canonicalCornerSite corner) neighbor) =
      cubeC19Neighbor corner neighbor := by
  fin_cases corner <;> fin_cases neighbor <;>
    simp [periodTwoCornerAt, infiniteC19Neighbor, canonicalCornerSite,
      integerParityBit, cornerFromParity, cubeC19Neighbor,
      c19Coord, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2]

theorem affine_phase_of_canonical_face_neighbor
    (eps : ℝ) (momentum : Fin 3 → ℝ)
    (corner : CubeCorner) (neighbor : C19Neighbor)
    (hface : c19FaceIncidenceSign neighbor ≠ 0) :
    affineCellBlochPhase eps momentum
        (infiniteC19Neighbor (canonicalCornerSite corner) neighbor) =
      preparedBlochFacePhase corner neighbor eps momentum := by
  fin_cases corner <;> fin_cases neighbor <;>
    simp_all +decide [c19FaceIncidenceSign,
      affineCellBlochPhase, periodTwoCellPhase,
      periodTwoCellCoordinate, infiniteC19Neighbor,
      preparedBlochFacePhase, cellForwardBlochPhase,
      cellBackwardBlochPhase,
      cellBit0, cellBit1, cellBit2] <;>
    ring

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

/-!
The symbol above is not merely a phase-decorated copy of the answer.  It is
the zeroth component of `chiIncidenceResidual` evaluated on an actual pair of
native complex C3 fields repeated over the infinite cubic lattice.  The Bloch
phase lives on the period-two cells, so positive and negative face neighbours
remain distinct even when their within-cell corners agree.
-/
theorem prepared_action_bloch_residual_is_native_residual
    (eps : ℝ) (momentum : Fin 3 → ℝ) (amplitude : Cell8)
    (corner : CubeCorner) :
    chiIncidenceResidual chi0
        (preparedNativeBlochState eps momentum amplitude)
        infiniteC19Geometry (canonicalCornerSite corner) 0 =
      preparedActionBlochResidual eps momentum amplitude corner := by
  rw [show
      chiIncidenceResidual chi0
          (preparedNativeBlochState eps momentum amplitude)
          infiniteC19Geometry (canonicalCornerSite corner) 0 =
        chiIncidenceComponentSymbol chi0
          (preparedNativeBlochState eps momentum amplitude)
          infiniteC19Geometry (fun _ _ => 1)
          (fun target =>
            wedge
              ((preparedNativeBlochState eps momentum amplitude).chi target)
              ((preparedNativeBlochState eps momentum amplitude).psi target) 0)
          (canonicalCornerSite corner) by
      symm
      exact chi_incidence_component_symbol_is_native_residual
        chi0 (preparedNativeBlochState eps momentum amplitude)
        infiniteC19Geometry 0 (canonicalCornerSite corner)]
  unfold chiIncidenceComponentSymbol preparedActionBlochResidual
  apply Finset.sum_congr rfl
  intro neighbor _
  by_cases hface : c19FaceIncidenceSign neighbor = 0
  · simp [hface]
  · simp only [infiniteC19Geometry]
    rw [prepared_native_bloch_area,
      period_two_corner_of_canonical_neighbor,
      affine_phase_of_canonical_face_neighbor eps momentum corner neighbor hface]
    have hcorner := canonical_corner_has_declared_parity corner
    simp [preparedNativeBlochState, preparedComplexChiBackground,
      cubeC19Geometry, hcorner, c3ScaleComplex, inner, chi0]
    rw [period_two_corner_of_canonical_neighbor]
    simp [unitArea]
    ring

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
  repeatedNativeResidual : ∀ (eps : ℝ) (momentum : Fin 3 → ℝ)
      (amplitude : Cell8) (corner : CubeCorner),
    chiIncidenceResidual chi0
        (preparedNativeBlochState eps momentum amplitude)
        infiniteC19Geometry (canonicalCornerSite corner) 0 =
      preparedActionBlochResidual eps momentum amplitude corner
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
  repeatedNativeResidual := prepared_action_bloch_residual_is_native_residual
  exactNativeBlochBridge := prepared_action_bloch_bridge
  zeroBackgroundResidual := prepared_action_bloch_bridge_zero
  nativeSquare := prepared_action_bloch_bridge_squares_to_scalar

end
end LeanDirac

#print axioms LeanDirac.prepared_action_bloch_bridge
#print axioms LeanDirac.prepared_action_bloch_residual_is_native_residual
#print axioms LeanDirac.dirac009_action_bridge_certificate
