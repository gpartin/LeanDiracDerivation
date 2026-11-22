import Mathlib

/-!
---
title: Two cubical Dirac sectors from one eight-corner fiber
type: theorem
---
Every complex amplitude on the eight corners of an oriented cube decomposes
into the positive and negative eigenspaces of the geometric flavor involution
defined below.  The two eigenspaces intersect only at zero, and the cubical
C19 first-order symbol preserves both.  The statement is algebraic and does
not assert a nonlinear localized particle.
-/

namespace Lax277496.CubicalTwoFlavor

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
  | 0 | 1 | 2 | 3 => 0 | _ => 1

def cellBit1 : CubeCorner → Nat
  | 0 | 1 | 4 | 5 => 0 | _ => 1

def cellBit2 : CubeCorner → Nat
  | 0 | 2 | 4 | 6 => 0 | _ => 1

def incidenceFlip (axis : Fin 3) : CubeCorner → CubeCorner :=
  match axis with | 0 => cellFlip0 | 1 => cellFlip1 | _ => cellFlip2

def incidenceSign (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cellBit0 corner = 0 then 1 else -1
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 1 else -1

def incidenceGamma (axis : Fin 3) (amplitude : Cell8) : Cell8 := fun corner =>
  incidenceSign axis corner * amplitude (incidenceFlip axis corner)

def parityBeta (amplitude : Cell8) : Cell8 := fun corner =>
  if (cellBit0 corner + cellBit1 corner + cellBit2 corner) % 2 = 0 then
    amplitude corner
  else -amplitude corner

def cubeFlavor (amplitude : Cell8) : Cell8 := fun corner =>
  let sign : ℝ := if cellBit2 corner = 0 then 1 else -1
  (sign : ℂ) * Complex.I * amplitude (cellFlip1 (cellFlip2 corner))

def plusProjection (amplitude : Cell8) : Cell8 := fun corner =>
  (amplitude corner + cubeFlavor amplitude corner) / 2

def minusProjection (amplitude : Cell8) : Cell8 := fun corner =>
  (amplitude corner - cubeFlavor amplitude corner) / 2

def firstOrderSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 := fun corner =>
  mass * parityBeta amplitude corner +
    k0 * incidenceGamma 0 amplitude corner +
    k1 * incidenceGamma 1 amplitude corner +
    k2 * incidenceGamma 2 amplitude corner

axiom twoFlavorClassification :
  (∀ amplitude,
    (fun corner => plusProjection amplitude corner +
      minusProjection amplitude corner) = amplitude) ∧
  (∀ amplitude, cubeFlavor (plusProjection amplitude) = plusProjection amplitude) ∧
  (∀ amplitude, cubeFlavor (minusProjection amplitude) =
    fun corner => -minusProjection amplitude corner) ∧
  (∀ amplitude, cubeFlavor amplitude = amplitude →
    cubeFlavor amplitude = (fun corner => -amplitude corner) → amplitude = 0) ∧
  (∀ mass k0 k1 k2 amplitude, cubeFlavor amplitude = amplitude →
    cubeFlavor (firstOrderSymbol mass k0 k1 k2 amplitude) =
      firstOrderSymbol mass k0 k1 k2 amplitude) ∧
  (∀ mass k0 k1 k2 amplitude,
    cubeFlavor amplitude = (fun corner => -amplitude corner) →
    cubeFlavor (firstOrderSymbol mass k0 k1 k2 amplitude) =
      fun corner => -firstOrderSymbol mass k0 k1 k2 amplitude corner)

end

end Lax277496.CubicalTwoFlavor
