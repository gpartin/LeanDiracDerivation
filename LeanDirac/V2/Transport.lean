/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module
public import LeanDirac.Dirac004DiscreteAction

@[expose] public section
namespace LeanDirac.V2
noncomputable section
open BigOperators Complex

/-- Generator of the site's complex phase. -/
def phaseTurn (u : C3) : C3 := fun c => I * u c

private theorem pair_turn_self (u : C3) : c3RealPair u (phaseTurn u) = 0 := by
  apply Finset.sum_eq_zero
  intro c _
  simp [phaseTurn, Complex.mul_re, Complex.mul_im]
  ring

/-- The onsite phase imbalance uses both native fields and the chi kinetic weight. -/
def imbalance (B : ℝ) (prev curr : State Site) (s : Site) : ℝ :=
  c3RealPair (curr.psi s) (phaseTurn (prev.psi s)) -
    B * c3RealPair (curr.chi s) (phaseTurn (prev.chi s))

private def localizedTurn [DecidableEq Site] (s : Site) (u : Site → C3) : Site → C3 :=
  fun i => if i = s then phaseTurn (u i) else c3Zero

private theorem turning_variations_agree [Fintype Site] [DecidableEq Site] [Fintype Edge]
    (chi₀ : ℝ) (q : State Site) (b : Edge → Bond Site) (s : Site) :
    psiEulerLagrangeForce chi₀ q b (localizedTurn s q.psi) =
    chiEulerLagrangeForce chi₀ q b (localizedTurn s q.chi) := by
  have hw (i : Site) :
      wedgeFirstVariation q (purePsiDirection (localizedTurn s q.psi)) i =
      wedgeFirstVariation q (pureChiDirection (localizedTurn s q.chi)) i := by
    funext c
    by_cases h : i = s <;> fin_cases c <;>
      simp [wedgeFirstVariation, purePsiDirection, pureChiDirection,
        localizedTurn, h, phaseTurn, c3Add, c3Zero, wedge] <;> ring
  unfold psiEulerLagrangeForce chiEulerLagrangeForce lagrangianStateFirstVariation
  unfold potentialFirstVariation
  congr 2
  · apply Finset.sum_congr rfl
    intro i _
    have hz (u : C3) : c3RealPair u c3Zero = 0 := by simp [c3RealPair, c3Zero]
    by_cases h : i = s <;>
      simp [sitePotentialFirstVariation, purePsiDirection, pureChiDirection,
        localizedTurn, h, pair_turn_self, hz]
  · apply Finset.sum_congr rfl
    intro e _
    simp only [exteriorAreaBondFirstVariation, bondAreaFirstVariation, hw]

private theorem sum_localized_turn [Fintype Site] [DecidableEq Site] (s : Site)
    (u v : Site → C3) :
    (∑ i, c3RealPair (v i) (localizedTurn s u i)) =
      c3RealPair (v s) (phaseTurn (u s)) := by
  classical
  rw [Finset.sum_eq_single s]
  · simp [localizedTurn]
  · intro i _ hi
    simp [localizedTurn, hi, c3RealPair, c3Zero]
  · simp

/-- The reciprocal phase torques are equal at each site of the native action. -/
theorem native_phase_torques_equal [Fintype Site] [DecidableEq Site] [Fintype Edge]
    (chi₀ B : ℝ) (q : State Site) (b : Edge → Bond Site) (hB : B ≠ 0) (s : Site) :
    c3RealPair ((accelerationVelocity chi₀ B q b).psiDot s) (phaseTurn (q.psi s)) =
    B * c3RealPair ((accelerationVelocity chi₀ B q b).chiDot s) (phaseTurn (q.chi s)) := by
  have hp := psi_acceleration_all_directions chi₀ B q b (localizedTurn s q.psi)
  have hc := chi_acceleration_all_directions chi₀ B q b hB (localizedTurn s q.chi)
  simp only [psiMomentumPair, chiMomentumPair, sum_localized_turn] at hp hc
  exact hp.trans ((turning_variations_agree chi₀ q b s).trans hc.symm)

private theorem turn_next_identity (prev curr acc : C3) (dt : ℝ) :
    c3RealPair (fun c => 2 * curr c - prev c + (dt^2 : ℂ) * acc c) (phaseTurn curr) =
      c3RealPair curr (phaseTurn prev) + dt^2 * c3RealPair acc (phaseTurn curr) := by
  unfold c3RealPair
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro c _
  simp [phaseTurn, Complex.mul_re, Complex.mul_im, pow_two]
  ring

/-- Every unrestricted native update preserves the local charge difference,
even for moving or distorted two-layer preparations and on arbitrary bond graphs. -/
theorem native_next_preserves_site_imbalance [Fintype Site] [DecidableEq Site] [Fintype Edge]
    (chi₀ B dt : ℝ) (b : Edge → Bond Site) (hB : B ≠ 0)
    (prev curr : State Site) (s : Site) :
    imbalance B curr (nativeNextState chi₀ B dt b prev curr) s = imbalance B prev curr s := by
  simp only [imbalance, nativeNextState, turn_next_identity]
  rw [native_phase_torques_equal chi₀ B curr b hB s]
  ring

/-- On an unrestricted history, a site keeps its initial imbalance indefinitely. -/
theorem history_site_imbalance [Fintype Site] [DecidableEq Site] [Fintype Edge]
    (chi₀ B dt : ℝ) (b : Edge → Bond Site) (hB : B ≠ 0) (q : ℕ → State Site)
    (steps : ∀ n, q (n+2) = nativeNextState chi₀ B dt b (q n) (q (n+1)))
    (s : Site) (n : ℕ) :
    imbalance B (q n) (q (n+1)) s = imbalance B (q 0) (q 1) s := by
  induction n with
  | zero => rfl
  | succ n ih =>
    calc
      _ = imbalance B (q (n+1)) (nativeNextState chi₀ B dt b (q n) (q (n+1))) s := by
        rw [← steps n]
      _ = imbalance B (q n) (q (n+1)) s :=
        native_next_preserves_site_imbalance chi₀ B dt b hB _ _ s
      _ = _ := ih

/-- Relabels both native fields by the same site map. -/
def shiftedState (shift : Site → Site) (q : State Site) : State Site :=
  ⟨q.psi ∘ shift, q.chi ∘ shift⟩

/-- An exact translation of both committed layers requires equal initial
imbalance on each translation orbit. This necessary condition allows balanced movers. -/
theorem translated_history_imbalance_constraint
    [Fintype Site] [DecidableEq Site] [Fintype Edge]
    (chi₀ B dt : ℝ) (b : Edge → Bond Site) (hB : B ≠ 0) (q : ℕ → State Site)
    (steps : ∀ n, q (n+2) = nativeNextState chi₀ B dt b (q n) (q (n+1)))
    (shift : Site → Site) (n : ℕ)
    (first : q n = shiftedState shift (q 0)) (second : q (n+1) = shiftedState shift (q 1))
    (s : Site) : imbalance B (q 0) (q 1) (shift s) = imbalance B (q 0) (q 1) s := by
  have h := history_site_imbalance chi₀ B dt b hB q steps s n
  rwa [first, second] at h

end
end LeanDirac.V2

