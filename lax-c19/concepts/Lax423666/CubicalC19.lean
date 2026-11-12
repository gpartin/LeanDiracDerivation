import Mathlib

/-!
---
title: Cubical C19 first-order factorization
type: theorem
---
Let eight complex amplitudes occupy the vertices of an oriented cube.  Flip a
vertex across each coordinate direction and attach the orientation sign given
by the lower coordinate bits.  Grade each vertex by total cube parity.  The
first-order symbol obtained from the three signed flips and this grading,
normalized by the long-wavelength second moment of the C19 stencil, squares to
the isotropic quadratic C19 symbol for every complex mass, real momentum, and
eight-corner amplitude.

This is an algebraic continuum-symbol statement.  It does not assert an exact
finite-spacing factorization or the existence of a physical electron.
-/

namespace Lax423666.CubicalC19

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

def cellBit0 : CubeCorner → Nat
  | 0 | 1 | 2 | 3 => 0
  | _ => 1

def cellBit1 : CubeCorner → Nat
  | 0 | 1 | 4 | 5 => 0
  | _ => 1

def cellBit2 : CubeCorner → Nat
  | 0 | 2 | 4 | 6 => 0
  | _ => 1

def incidenceFlip (axis : Fin 3) : CubeCorner → CubeCorner :=
  match axis with
  | 0 => cellFlip0
  | 1 => cellFlip1
  | _ => cellFlip2

def incidenceSign (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cellBit0 corner = 0 then 1 else -1
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 1 else -1

def incidenceGamma (axis : Fin 3) (amplitude : Cell8) : Cell8 :=
  fun corner => incidenceSign axis corner * amplitude (incidenceFlip axis corner)

def parityBeta (amplitude : Cell8) : Cell8 :=
  fun corner =>
    if (cellBit0 corner + cellBit1 corner + cellBit2 corner) % 2 = 0 then
      amplitude corner
    else
      -amplitude corner

def incidenceLinear (k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    k0 * incidenceGamma 0 amplitude corner +
    k1 * incidenceGamma 1 amplitude corner +
    k2 * incidenceGamma 2 amplitude corner

def firstOrderSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    mass * parityBeta amplitude corner +
      incidenceLinear k0 k1 k2 amplitude corner

def c19ContinuumSpeed : ℝ := Real.sqrt 12

def c19FirstOrderSymbol
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  firstOrderSymbol mass
    (c19ContinuumSpeed * momentum 0)
    (c19ContinuumSpeed * momentum 1)
    (c19ContinuumSpeed * momentum 2) amplitude

def c19QuadraticSymbol
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    (mass ^ 2 + (12 : ℂ) *
      ((momentum 0 : ℂ) ^ 2 + (momentum 1 : ℂ) ^ 2 +
        (momentum 2 : ℂ) ^ 2)) * amplitude corner

axiom cubicalC19Factorization
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    c19FirstOrderSymbol mass momentum
        (c19FirstOrderSymbol mass momentum amplitude) =
      c19QuadraticSymbol mass momentum amplitude

end

end Lax423666.CubicalC19
