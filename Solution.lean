module

public import LeanDirac.Dirac009Geometry

public section

/-!
# Proved Palomar solution

The declarations below deliberately mirror `Challenge.lean`.  Their bodies
are aliases of the production definitions, so the proof is checked against
the same cubical C19 operator used in the full LeanDiracDerivation chain.
-/

namespace LFMDiracChallenge

noncomputable section

abbrev CubeCorner := Fin 8
abbrev Cell8 := CubeCorner → ℂ

def cellFlip0 : CubeCorner → CubeCorner
  | 0 => 4 | 1 => 5 | 2 => 6 | 3 => 7
  | 4 => 0 | 5 => 1 | 6 => 2 | _ => 3

def cellFlip1 : CubeCorner → CubeCorner
  | 0 => 2 | 1 => 3 | 2 => 0 | 3 => 1
  | 4 => 6 | 5 => 7 | 6 => 4 | _ => 5

def cellFlip2 : CubeCorner → CubeCorner
  | 0 => 1 | 1 => 0 | 2 => 3 | 3 => 2
  | 4 => 5 | 5 => 4 | 6 => 7 | _ => 6

def cellBit2 : CubeCorner → Nat
  | 0 | 2 | 4 | 6 => 0
  | _ => 1

def cellBit0 : CubeCorner → Nat
  | 0 | 1 | 2 | 3 => 0
  | _ => 1

def cellBit1 : CubeCorner → Nat
  | 0 | 1 | 4 | 5 => 0
  | _ => 1

def cubeIncidenceFlip (axis : Fin 3) : CubeCorner → CubeCorner :=
  match axis with
  | 0 => cellFlip0
  | 1 => cellFlip1
  | _ => cellFlip2

def cubeIncidenceSign (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cellBit0 corner = 0 then 1 else -1
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 1 else -1

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

def cubeIncidenceLinear
    (k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    k0 * cubeIncidenceGamma 0 amplitude corner +
    k1 * cubeIncidenceGamma 1 amplitude corner +
    k2 * cubeIncidenceGamma 2 amplitude corner

def cubeDiracSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    mass * cubeParityBeta amplitude corner +
      cubeIncidenceLinear k0 k1 k2 amplitude corner

def c19ContinuumSpeed : ℝ := Real.sqrt 12

def c19ContinuumDiracSymbol
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  cubeDiracSymbol mass
    (c19ContinuumSpeed * momentum 0)
    (c19ContinuumSpeed * momentum 1)
    (c19ContinuumSpeed * momentum 2) amplitude

def c19ContinuumScalarSymbol
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    (mass ^ 2 + (12 : ℂ) *
      ((momentum 0 : ℂ) ^ 2 + (momentum 1 : ℂ) ^ 2 +
        (momentum 2 : ℂ) ^ 2)) * amplitude corner

theorem main_result
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    c19ContinuumDiracSymbol mass momentum
        (c19ContinuumDiracSymbol mass momentum amplitude) =
      c19ContinuumScalarSymbol mass momentum amplitude := by
  change LeanDirac.c19ContinuumDiracSymbol mass momentum
      (LeanDirac.c19ContinuumDiracSymbol mass momentum amplitude) =
    LeanDirac.c19ContinuumScalarSymbol mass momentum amplitude
  exact LeanDirac.c19_continuum_dirac_square_is_native_quadratic_symbol
    mass momentum amplitude

end

end LFMDiracChallenge
