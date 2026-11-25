/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.V4.LocalizedRestSeed

@[expose] public section

namespace LeanDirac.V4

/-!
# V4-004: spin one-half on the localized four-component readout

V4-003 supplies the fixed numerical bridge from the localized native
two-layer carrier to either exhaustive cubical four-component copy.  This
module records the exact algebraic side of the same bridge.  The rotation is
the native cube-incidence turn already derived in V2; no spinor rule, half
angle, or new action term is introduced here.

The theorem is deliberately scoped to the readout.  The numerical V4-003/004
artifacts verify that this is the same localized carrier/readout and retain
the finite-timestep complementary remainder.  A cumulative analytic bridge
back to the full localized fields remains a V4-010 obligation.
-/

open LeanDirac.V2

noncomputable section

structure LocalizedReadoutSpinCertificate : Prop where
  uniqueNonzeroScale : ∃! scale : ℝ, admissibleCubicalSpinScale scale
  bothCopyTurns :
    (∀ angle axis z,
      cubicalContinuousTurn angle axis (cubicalDiracEmbedMinus z) =
        cubicalDiracEmbedMinus
          (standardCubicalContinuousTurn angle axis z)) ∧
    (∀ angle axis z,
      cubicalContinuousTurn angle axis (cubicalDiracEmbedPlus z) =
        cubicalDiracEmbedPlus
          (standardCubicalContinuousTurn angle axis z))
  twoPiSign : ∀ axis amplitude,
    cubicalContinuousTurn (2 * Real.pi) axis amplitude =
      fun corner => -amplitude corner
  fourPiReturn : ∀ axis amplitude,
    cubicalContinuousTurn (4 * Real.pi) axis amplitude = amplitude
  densityPreserved : ∀ angle axis z,
    cubicalCell4Density (standardCubicalContinuousTurn angle axis z) =
      cubicalCell4Density z

theorem localized_readout_spin_half :
    LocalizedReadoutSpinCertificate where
  uniqueNonzeroScale := exists_unique_admissible_cubical_spin_scale
  bothCopyTurns :=
    ⟨cubical_continuous_turn_intertwines_minus,
      cubical_continuous_turn_intertwines_plus⟩
  twoPiSign := cubical_continuous_turn_two_pi
  fourPiReturn := cubical_continuous_turn_four_pi
  densityPreserved := standard_cubical_continuous_turn_preserves_density

end
end LeanDirac.V4

#print axioms LeanDirac.V4.localized_readout_spin_half
