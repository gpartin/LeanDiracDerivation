module

public import Mathlib.Data.Complex.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Tactic.FinCases
public import Mathlib.Tactic.LinearCombination
public import Mathlib.Tactic.Ring

public section

namespace LFMDiracV2Challenge

noncomputable section

open Complex

def chi0 : ℝ := 19
abbrev Momentum := Fin 3 → ℝ
abbrev Corner := Fin 8
abbrev Cell8 := Corner → ℂ

def nativeSitePotential (rhoPsi rhoChi : ℝ) : ℝ :=
  (rhoChi - chi0^2)^2 / 4 + rhoChi * rhoPsi / 2 + rhoPsi^2 / 4

def c19TransportSymbol (momentum : Momentum) : ℝ :=
  8 * ((1 - Real.cos (momentum 0)) +
    (1 - Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 1) * Real.cos (momentum 2)))

def uniformModalPotential
    (eps : ℝ) (amplitude : ℂ) (momentum : Momentum) : ℝ :=
  nativeSitePotential (Complex.normSq ((eps : ℂ) * amplitude)) (chi0^2) +
    Complex.normSq ((eps : ℂ) * amplitude) / 2 *
      c19TransportSymbol momentum

def nativeTangentSymbol (momentum : Momentum) : ℝ :=
  chi0^2 + c19TransportSymbol momentum

def cellFlip0 : Corner → Corner
  | 0 => 4 | 1 => 5 | 2 => 6 | 3 => 7
  | 4 => 0 | 5 => 1 | 6 => 2 | _ => 3

def cellFlip1 : Corner → Corner
  | 0 => 2 | 1 => 3 | 2 => 0 | 3 => 1
  | 4 => 6 | 5 => 7 | 6 => 4 | _ => 5

def cellFlip2 : Corner → Corner
  | 0 => 1 | 1 => 0 | 2 => 3 | 3 => 2
  | 4 => 5 | 5 => 4 | 6 => 7 | _ => 6

def cellBit0 : Corner → Nat
  | 0 | 1 | 2 | 3 => 0 | _ => 1

def cellBit1 : Corner → Nat
  | 0 | 1 | 4 | 5 => 0 | _ => 1

def cellBit2 : Corner → Nat
  | 0 | 2 | 4 | 6 => 0 | _ => 1

def incidenceFlip (axis : Fin 3) : Corner → Corner :=
  match axis with | 0 => cellFlip0 | 1 => cellFlip1 | _ => cellFlip2

def incidenceSign (axis : Fin 3) (corner : Corner) : ℂ :=
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

def firstOrderSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 := fun corner =>
  mass * parityBeta amplitude corner +
    k0 * incidenceGamma 0 amplitude corner +
    k1 * incidenceGamma 1 amplitude corner +
    k2 * incidenceGamma 2 amplitude corner

def quadraticSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 := fun corner =>
  (mass^2 + k0^2 + k1^2 + k2^2) * amplitude corner

def cubeFlavor (amplitude : Cell8) : Cell8 := fun corner =>
  let sign : ℝ := if cellBit2 corner = 0 then 1 else -1
  (sign : ℂ) * Complex.I * amplitude (cellFlip1 (cellFlip2 corner))

def plusProjection (amplitude : Cell8) : Cell8 := fun corner =>
  (amplitude corner + cubeFlavor amplitude corner) / 2

def minusProjection (amplitude : Cell8) : Cell8 := fun corner =>
  (amplitude corner - cubeFlavor amplitude corner) / 2

theorem exact_vacuum_has_zero_potential :
    nativeSitePotential 0 (chi0^2) = 0 := by
  simp [nativeSitePotential]

theorem native_action_gives_c19_tangent
    (eps : ℝ) (amplitude : ℂ) (momentum : Momentum) :
    uniformModalPotential eps amplitude momentum =
      eps^2 * Complex.normSq amplitude / 2 * nativeTangentSymbol momentum +
        eps^4 * Complex.normSq amplitude ^ 2 / 4 := by
  simp [uniformModalPotential, nativeSitePotential, nativeTangentSymbol,
    chi0, Complex.normSq]
  ring

theorem cubical_first_order_square
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    firstOrderSymbol mass k0 k1 k2
        (firstOrderSymbol mass k0 k1 k2 amplitude) =
      quadraticSymbol mass k0 k1 k2 amplitude := by
  funext corner
  fin_cases corner <;>
    simp [firstOrderSymbol, quadraticSymbol, parityBeta, incidenceGamma,
      incidenceSign, incidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2] <;> ring

private theorem flavor_commutes_first_order
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    cubeFlavor (firstOrderSymbol mass k0 k1 k2 amplitude) =
      firstOrderSymbol mass k0 k1 k2 (cubeFlavor amplitude) := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, firstOrderSymbol, parityBeta, incidenceGamma,
      incidenceSign, incidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2] <;> ring_nf

private theorem first_order_neg
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    firstOrderSymbol mass k0 k1 k2 (fun corner => -amplitude corner) =
      fun corner => -firstOrderSymbol mass k0 k1 k2 amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [firstOrderSymbol, parityBeta, incidenceGamma, incidenceSign,
      incidenceFlip, cellFlip0, cellFlip1, cellFlip2, cellBit0, cellBit1,
      cellBit2] <;> ring

theorem cubical_two_sector_classification :
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
      fun corner => -firstOrderSymbol mass k0 k1 k2 amplitude corner) := by
  constructor
  · intro amplitude
    funext corner
    simp [plusProjection, minusProjection]
    ring
  constructor
  · intro amplitude
    funext corner
    fin_cases corner <;>
      simp [plusProjection, cubeFlavor, cellFlip1, cellFlip2, cellBit2] <;>
      ring_nf <;> rw [Complex.I_sq] <;> ring
  constructor
  · intro amplitude
    funext corner
    fin_cases corner <;>
      simp [minusProjection, cubeFlavor, cellFlip1, cellFlip2, cellBit2] <;>
      ring_nf <;> rw [Complex.I_sq] <;> ring
  constructor
  · intro amplitude hplus hminus
    funext corner
    have h := congrFun (hplus.symm.trans hminus) corner
    simp only [Pi.zero_apply]
    have hz : amplitude corner - (-amplitude corner) = 0 := sub_eq_zero.mpr h
    linear_combination (1 / 2 : ℂ) * hz
  constructor
  · intro mass k0 k1 k2 amplitude hplus
    rw [flavor_commutes_first_order, hplus]
  · intro mass k0 k1 k2 amplitude hminus
    rw [flavor_commutes_first_order, hminus, first_order_neg]

end

end LFMDiracV2Challenge
