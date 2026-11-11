import LeanDirac.Dirac008Incidence

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
noncomputable section

/-!
# Native cube-incidence algebra

DIRAC-006 proves that the action-derived aligned transport has the even C19
symbol and zero first moment.  Therefore that scalar transport channel cannot
itself be renamed as a nonzero first-order Dirac transport generator.

The definitions below construct the signed incidence maps of the oriented
cube directly from its three parity bits.  The signs are not Dirac matrices
supplied as data: they are the usual orientation signs for crossing successive
coordinate directions on a cube.  Their anticommutation follows from the odd
sign around every oriented square face.

This file deliberately does **not** promote DIRAC-009 by itself.  It contains
only the cube-incidence and flavor algebra used by the admitted proof chain.
The action-to-incidence obligation is closed downstream in
`Dirac009ActionBridge`, where the surrounding-cell Bloch coefficient of the native
action residual reduces exactly to this incidence operator.  Keeping the
geometry and action bridge in separate modules prevents this file from
assuming the conclusion it is later used to prove.
-/

def cubeIncidenceFlip (axis : Fin 3) : CubeCorner → CubeCorner :=
  match axis with
  | 0 => cellFlip0
  | 1 => cellFlip1
  | _ => cellFlip2

def preparedCubeEdgeOverlap (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  overlap (tetraCorner corner)
    (tetraCorner (cubeIncidenceFlip axis corner))

def preparedCubeExpectedEdgeOverlap
    (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis.val, corner.val with
  | 0, 0 => -tetraA | 0, 1 => -(tetraA * Complex.I)
  | 0, 2 => tetraA * Complex.I | 0, 3 => -tetraA
  | 0, 4 => -tetraA | 0, 5 => tetraA * Complex.I
  | 0, 6 => -(tetraA * Complex.I) | 0, _ => -tetraA
  | 1, 0 => tetraA | 1, 1 => -(tetraA * Complex.I)
  | 1, 2 => tetraA | 1, 3 => tetraA * Complex.I
  | 1, 4 => tetraA * Complex.I | 1, 5 => tetraA
  | 1, 6 => -(tetraA * Complex.I) | 1, _ => tetraA
  | _, 0 => tetraA | _, 1 => tetraA
  | _, 2 => tetraA * Complex.I | _, 3 => -(tetraA * Complex.I)
  | _, 4 => -(tetraA * Complex.I) | _, 5 => tetraA * Complex.I
  | _, 6 => tetraA | _, _ => tetraA

theorem prepared_cube_edge_overlap_values
    (axis : Fin 3) (corner : CubeCorner) :
    preparedCubeEdgeOverlap axis corner =
      preparedCubeExpectedEdgeOverlap axis corner := by
  apply Complex.ext <;> fin_cases axis <;> fin_cases corner <;>
    simp [preparedCubeEdgeOverlap, preparedCubeExpectedEdgeOverlap,
      overlap, Fin.sum_univ_succ, tetraCorner, tetraRay0,
      tetraRay1, tetraRay2, tetraRay3, c3Of, c3Neg,
      omega, omegaBar, tetraA, tetraB, cubeIncidenceFlip,
      cellFlip0, cellFlip1, cellFlip2] <;>
    field_simp [sqrt_three_ne_zero] <;> ring_nf <;>
    nlinarith [sqrt_three_sq, sqrt_two_sq]

def preparedGaussianPhaseExponent (z : ℂ) : ZMod 4 :=
  if z.im = 0 then
    if 0 < z.re then 0 else 2
  else if 0 < z.im then 1 else 3

def preparedCubeConnection4
    (axis : Fin 3) (corner : CubeCorner) : ZMod 4 :=
  preparedGaussianPhaseExponent (preparedCubeEdgeOverlap axis corner)

def preparedCubeGauge4 (corner : CubeCorner) : ZMod 4 :=
  match corner.val with
  | 0 => 0 | 1 => 0 | 2 => 0 | 3 => 1
  | 4 => 2 | 5 => 1 | 6 => 3 | _ => 3

def cubeIncidenceSign (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cellBit0 corner = 0 then 1 else -1
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 1 else -1

def cubeIncidenceConnection4
    (axis : Fin 3) (corner : CubeCorner) : ZMod 4 :=
  match axis with
  | 0 => 0
  | 1 => if cellBit0 corner = 0 then 0 else 2
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 0 else 2

theorem prepared_cube_connection_gauge_eq_incidence
    (axis : Fin 3) (corner : CubeCorner) :
    preparedCubeConnection4 axis corner - preparedCubeGauge4 corner +
        preparedCubeGauge4 (cubeIncidenceFlip axis corner) =
      cubeIncidenceConnection4 axis corner := by
  rw [preparedCubeConnection4, prepared_cube_edge_overlap_values]
  have hA : 0 < Real.sqrt 3 / 3 := by positivity
  have hAnot : ¬ Real.sqrt 3 / 3 < 0 := not_lt_of_ge (le_of_lt hA)
  have hfour : (4 : ZMod 4) = 0 := by decide
  have hnegTwo : (-2 : ZMod 4) = 2 := by decide
  fin_cases axis <;> fin_cases corner <;>
    simp [preparedCubeExpectedEdgeOverlap, preparedGaussianPhaseExponent,
      preparedCubeGauge4, cubeIncidenceConnection4,
      cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, tetraA, hA, hAnot]
  all_goals first | exact hfour | exact hnegTwo

def cubeIncidenceGamma (axis : Fin 3) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    cubeIncidenceSign axis corner *
      amplitude (cubeIncidenceFlip axis corner)

def cubeParityBeta (amplitude : Cell8) : Cell8 :=
  fun corner =>
    if (cellBit0 corner + cellBit1 corner + cellBit2 corner) % 2 = 0 then
      amplitude corner
    else
      -amplitude corner

theorem cube_incidence_gamma_square (axis : Fin 3) (amplitude : Cell8) :
    cubeIncidenceGamma axis (cubeIncidenceGamma axis amplitude) = amplitude := by
  funext corner
  fin_cases axis <;> fin_cases corner <;>
    simp [cubeIncidenceGamma, cubeIncidenceSign, cubeIncidenceFlip,
      cellFlip0, cellFlip1, cellFlip2, cellBit0, cellBit1]

theorem cube_incidence_gamma_anticommute
    (axis other : Fin 3) (haxis : axis ≠ other) (amplitude : Cell8) :
    cubeIncidenceGamma axis (cubeIncidenceGamma other amplitude) =
      fun corner =>
        -cubeIncidenceGamma other (cubeIncidenceGamma axis amplitude) corner := by
  fin_cases axis <;> fin_cases other <;> try { contradiction }
  all_goals
    funext corner
    fin_cases corner <;>
      simp [cubeIncidenceGamma, cubeIncidenceSign, cubeIncidenceFlip,
        cellFlip0, cellFlip1, cellFlip2, cellBit0, cellBit1]

theorem cube_parity_beta_square (amplitude : Cell8) :
    cubeParityBeta (cubeParityBeta amplitude) = amplitude := by
  funext corner
  fin_cases corner <;>
    simp [cubeParityBeta, cellBit0, cellBit1, cellBit2]

theorem cube_parity_beta_anticommutes
    (axis : Fin 3) (amplitude : Cell8) :
    cubeParityBeta (cubeIncidenceGamma axis amplitude) =
      fun corner =>
        -cubeIncidenceGamma axis (cubeParityBeta amplitude) corner := by
  fin_cases axis
  all_goals
    funext corner
    fin_cases corner <;>
      simp [cubeParityBeta, cubeIncidenceGamma, cubeIncidenceSign,
        cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
        cellBit0, cellBit1, cellBit2]

def cubeIncidenceLinear
    (k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    k0 * cubeIncidenceGamma 0 amplitude corner +
    k1 * cubeIncidenceGamma 1 amplitude corner +
    k2 * cubeIncidenceGamma 2 amplitude corner

theorem cube_incidence_linear_square
    (k0 k1 k2 : ℂ) (amplitude : Cell8) :
    cubeIncidenceLinear k0 k1 k2
        (cubeIncidenceLinear k0 k1 k2 amplitude) =
      fun corner => (k0^2 + k1^2 + k2^2) * amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [cubeIncidenceLinear, cubeIncidenceGamma, cubeIncidenceSign,
      cubeIncidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1] <;>
    ring

/-!
The eight cube amplitudes contain two equivalent four-coordinate sectors.
The commuting `cubeFlavor` map is assembled from the same cube translations
and parity bits as the incidence connection; it is not an independent
projector supplied to the dynamics.
-/

abbrev Cell4 := Fin 4 → ℂ

def cubeFlavorSign (corner : CubeCorner) : ℝ :=
  if cellBit2 corner % 2 = 0 then 1 else -1

def cubeFlavor (amplitude : Cell8) : Cell8 := fun corner =>
  Complex.I * (cubeFlavorSign corner : ℂ) *
    amplitude (cellFlip1 (cellFlip2 corner))

theorem cube_flavor_involution (amplitude : Cell8) :
    cubeFlavor (cubeFlavor amplitude) = amplitude := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, cubeFlavorSign, cellFlip1, cellFlip2, cellBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cube_flavor_commutes_gamma
    (axis : Fin 3) (amplitude : Cell8) :
    cubeFlavor (cubeIncidenceGamma axis amplitude) =
      cubeIncidenceGamma axis (cubeFlavor amplitude) := by
  funext corner
  fin_cases axis <;> fin_cases corner <;>
    simp [cubeFlavor, cubeFlavorSign, cubeIncidenceGamma,
      cubeIncidenceSign, cubeIncidenceFlip, cellFlip0, cellFlip1,
      cellFlip2, cellBit0, cellBit1, cellBit2]

theorem cube_flavor_commutes_beta (amplitude : Cell8) :
    cubeFlavor (cubeParityBeta amplitude) =
      cubeParityBeta (cubeFlavor amplitude) := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, cubeFlavorSign, cubeParityBeta,
      cellFlip1, cellFlip2, cellBit0, cellBit1, cellBit2]

def cubeEmbedMinus (z : Cell4) : Cell8
  | 0 => z 0 | 1 => z 1 | 2 => -Complex.I * z 1 | 3 => Complex.I * z 0
  | 4 => z 2 | 5 => z 3 | 6 => -Complex.I * z 3 | _ => Complex.I * z 2

def cubeEmbedPlus (z : Cell4) : Cell8
  | 0 => z 0 | 1 => z 1 | 2 => Complex.I * z 1 | 3 => -Complex.I * z 0
  | 4 => z 2 | 5 => z 3 | 6 => Complex.I * z 3 | _ => -Complex.I * z 2

theorem cube_embed_minus_injective : Function.Injective cubeEmbedMinus := by
  intro first second h
  funext component
  fin_cases component
  · exact congrFun h 0
  · exact congrFun h 1
  · exact congrFun h 4
  · exact congrFun h 5

theorem cube_embed_plus_injective : Function.Injective cubeEmbedPlus := by
  intro first second h
  funext component
  fin_cases component
  · exact congrFun h 0
  · exact congrFun h 1
  · exact congrFun h 4
  · exact congrFun h 5

theorem cube_embed_minus_has_negative_flavor (z : Cell4) :
    cubeFlavor (cubeEmbedMinus z) = fun corner => -cubeEmbedMinus z corner := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, cubeFlavorSign, cubeEmbedMinus,
      cellFlip1, cellFlip2, cellBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cube_embed_plus_has_positive_flavor (z : Cell4) :
    cubeFlavor (cubeEmbedPlus z) = cubeEmbedPlus z := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, cubeFlavorSign, cubeEmbedPlus,
      cellFlip1, cellFlip2, cellBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

end
end LeanDirac

#print axioms LeanDirac.cube_incidence_gamma_anticommute
#print axioms LeanDirac.cube_flavor_commutes_gamma
