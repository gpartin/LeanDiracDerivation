/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.V2.CubicalClifford

@[expose] public section

namespace LeanDirac.V2

/-!
# Exhaustive classification of the two cubical four-component sectors

The cubical flavor involution commutes with every native cubical transport
generator and with the mass grading.  This file proves the missing converse:
its `+1` and `-1` eigenspaces are exactly the images of the two already
constructed four-component embeddings.  Consequently the eight-corner
fiber is the direct sum of two, and only two, rank-four sectors.

This is an algebraic prediction of the declared cubical root.  It does not
assert that a measurement dynamically selects one sector, nor that the two
sectors are two independently localized particles.
-/

noncomputable section
open Complex

def cubicalFlavorPlusProjection (amplitude : CubicalCell8) : CubicalCell8 :=
  fun corner => (amplitude corner + cubicalFlavor amplitude corner) / 2

def cubicalFlavorMinusProjection (amplitude : CubicalCell8) : CubicalCell8 :=
  fun corner => (amplitude corner - cubicalFlavor amplitude corner) / 2

theorem cubical_flavor_projection_decomposition (amplitude : CubicalCell8) :
    (fun corner =>
      cubicalFlavorPlusProjection amplitude corner +
        cubicalFlavorMinusProjection amplitude corner) = amplitude := by
  funext corner
  simp [cubicalFlavorPlusProjection, cubicalFlavorMinusProjection]
  ring

theorem cubical_flavor_plus_projection_is_positive (amplitude : CubicalCell8) :
    cubicalFlavor (cubicalFlavorPlusProjection amplitude) =
      cubicalFlavorPlusProjection amplitude := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavorPlusProjection, cubicalFlavor, cubicalFlavorSign,
      cubicalFlip1, cubicalFlip2, cubicalBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

theorem cubical_flavor_minus_projection_is_negative (amplitude : CubicalCell8) :
    cubicalFlavor (cubicalFlavorMinusProjection amplitude) =
      fun corner => -cubicalFlavorMinusProjection amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [cubicalFlavorMinusProjection, cubicalFlavor, cubicalFlavorSign,
      cubicalFlip1, cubicalFlip2, cubicalBit2] <;>
    ring_nf <;> rw [Complex.I_sq] <;> ring

/-- Four standard coordinates recovered from a positive-flavor amplitude. -/
def cubicalPositiveFlavorRead (amplitude : CubicalCell8) : CubicalCell4
  | 0 => amplitude 5
  | 1 => -amplitude 0
  | 2 => amplitude 1
  | _ => amplitude 4

/-- Four standard coordinates recovered from a negative-flavor amplitude. -/
def cubicalNegativeFlavorRead (amplitude : CubicalCell8) : CubicalCell4
  | 0 => amplitude 0
  | 1 => amplitude 5
  | 2 => amplitude 4
  | _ => -amplitude 1

theorem positive_flavor_is_exactly_plus_embedding
    (amplitude : CubicalCell8) (hflavor : cubicalFlavor amplitude = amplitude) :
    cubicalDiracEmbedPlus (cubicalPositiveFlavorRead amplitude) = amplitude := by
  have h0 := congrFun hflavor 0
  have h1 := congrFun hflavor 1
  have h2 := congrFun hflavor 2
  have h3 := congrFun hflavor 3
  have h4 := congrFun hflavor 4
  have h5 := congrFun hflavor 5
  have h6 := congrFun hflavor 6
  have h7 := congrFun hflavor 7
  funext corner
  fin_cases corner <;>
    simp [cubicalDiracEmbedPlus, cubicalEmbedPlus, standardToCubicalPlus,
      cubicalPositiveFlavorRead, cubicalFlavor, cubicalFlavorSign,
      cubicalFlip1, cubicalFlip2, cubicalBit2] at h0 h1 h2 h3 h4 h5 h6 h7 ⊢ <;>
    assumption

theorem negative_flavor_is_exactly_minus_embedding
    (amplitude : CubicalCell8)
    (hflavor : cubicalFlavor amplitude = fun corner => -amplitude corner) :
    cubicalDiracEmbedMinus (cubicalNegativeFlavorRead amplitude) = amplitude := by
  have h0 := congrFun hflavor 0
  have h1 := congrFun hflavor 1
  have h2 := congrFun hflavor 2
  have h3 := congrFun hflavor 3
  have h4 := congrFun hflavor 4
  have h5 := congrFun hflavor 5
  have h6 := congrFun hflavor 6
  have h7 := congrFun hflavor 7
  funext corner
  fin_cases corner <;>
    simp [cubicalDiracEmbedMinus, cubicalEmbedMinus, standardToCubicalMinus,
      cubicalNegativeFlavorRead, cubicalFlavor, cubicalFlavorSign,
      cubicalFlip1, cubicalFlip2, cubicalBit2] at h0 h1 h2 h3 h4 h5 h6 h7 ⊢ <;>
    first | assumption | linear_combination (-1 : ℂ) * h2 |
      linear_combination (-1 : ℂ) * h6

theorem cubical_flavor_copies_intersect_only_at_zero
    (amplitude : CubicalCell8)
    (hplus : cubicalFlavor amplitude = amplitude)
    (hminus : cubicalFlavor amplitude = fun corner => -amplitude corner) :
    amplitude = 0 := by
  funext corner
  have h := congrFun (hplus.symm.trans hminus) corner
  simp only [Pi.zero_apply]
  have hzero : amplitude corner - (-amplitude corner) = 0 :=
    sub_eq_zero.mpr h
  linear_combination (1 / 2 : ℂ) * hzero

structure CubicalFlavorClassificationCertificate : Prop where
  everyAmplitudeDecomposes : ∀ amplitude,
    (fun corner =>
      cubicalFlavorPlusProjection amplitude corner +
        cubicalFlavorMinusProjection amplitude corner) = amplitude
  positiveProjectionHasPositiveFlavor : ∀ amplitude,
    cubicalFlavor (cubicalFlavorPlusProjection amplitude) =
      cubicalFlavorPlusProjection amplitude
  negativeProjectionHasNegativeFlavor : ∀ amplitude,
    cubicalFlavor (cubicalFlavorMinusProjection amplitude) =
      fun corner => -cubicalFlavorMinusProjection amplitude corner
  positiveCopyIsExhaustive : ∀ amplitude,
    cubicalFlavor amplitude = amplitude →
      cubicalDiracEmbedPlus (cubicalPositiveFlavorRead amplitude) = amplitude
  negativeCopyIsExhaustive : ∀ amplitude,
    cubicalFlavor amplitude = (fun corner => -amplitude corner) →
      cubicalDiracEmbedMinus (cubicalNegativeFlavorRead amplitude) = amplitude
  positiveEmbeddingIsInjective : Function.Injective cubicalDiracEmbedPlus
  negativeEmbeddingIsInjective : Function.Injective cubicalDiracEmbedMinus
  copyIntersectionIsZero : ∀ amplitude,
    cubicalFlavor amplitude = amplitude →
    cubicalFlavor amplitude = (fun corner => -amplitude corner) →
      amplitude = 0

theorem cubical_flavor_classification :
    CubicalFlavorClassificationCertificate where
  everyAmplitudeDecomposes := cubical_flavor_projection_decomposition
  positiveProjectionHasPositiveFlavor :=
    cubical_flavor_plus_projection_is_positive
  negativeProjectionHasNegativeFlavor :=
    cubical_flavor_minus_projection_is_negative
  positiveCopyIsExhaustive := positive_flavor_is_exactly_plus_embedding
  negativeCopyIsExhaustive := negative_flavor_is_exactly_minus_embedding
  positiveEmbeddingIsInjective := cubical_dirac_embed_plus_injective
  negativeEmbeddingIsInjective := cubical_dirac_embed_minus_injective
  copyIntersectionIsZero := cubical_flavor_copies_intersect_only_at_zero

end
end LeanDirac.V2

#print axioms LeanDirac.V2.positive_flavor_is_exactly_plus_embedding
#print axioms LeanDirac.V2.negative_flavor_is_exactly_minus_embedding
#print axioms LeanDirac.V2.cubical_flavor_classification
