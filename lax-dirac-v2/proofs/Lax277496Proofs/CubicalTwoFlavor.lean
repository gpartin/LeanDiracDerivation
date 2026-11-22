import Lax277496.CubicalTwoFlavor

namespace Lax277496Proofs

open Lax277496.CubicalTwoFlavor

noncomputable section

private theorem flavor_commutes_first_order
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    cubeFlavor (firstOrderSymbol mass k0 k1 k2 amplitude) =
      firstOrderSymbol mass k0 k1 k2 (cubeFlavor amplitude) := by
  funext corner
  fin_cases corner <;>
    simp [cubeFlavor, firstOrderSymbol, parityBeta, incidenceGamma,
      incidenceSign, incidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2] <;>
    ring_nf

private theorem first_order_neg
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    firstOrderSymbol mass k0 k1 k2 (fun corner => -amplitude corner) =
      fun corner => -firstOrderSymbol mass k0 k1 k2 amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [firstOrderSymbol, parityBeta, incidenceGamma, incidenceSign,
      incidenceFlip, cellFlip0, cellFlip1, cellFlip2, cellBit0, cellBit1,
      cellBit2] <;> ring

/--
---
conclusion: Lax277496.CubicalTwoFlavor.twoFlavorClassification
---
The proof expands the eight corners, proves that the flavor operation is an
involution, constructs its two projections, and checks that the cubical
first-order symbol commutes with the flavor operation.
-/
theorem twoFlavorClassification_proof :
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

end Lax277496Proofs
