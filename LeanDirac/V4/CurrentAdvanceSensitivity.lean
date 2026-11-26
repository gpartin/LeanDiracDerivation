/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module

public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.Deriv.Prod
public import Mathlib.Tactic.Abel

@[expose] public section

namespace LeanDirac.V4.CurrentAdvanceSensitivity

/-! # Implicit-update current sensitivity

The shooting correction varies a committed native layer and differentiates
its following layer only along a curve of actual Euler–Lagrange roots.
This theorem proves the adjoint derivative formula. It does not assume or
prove existence of a traveling root, a particular physical current, or a
freely moving carrier. Those are separately tested. -/

theorem current_advance_adjoint
    {E V : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup V] [NormedSpace ℝ V]
    (G : E × E → E) (H : E × E → V)
    (dG : (E × E) →L[ℝ] E) (dH : (E × E) →L[ℝ] V)
    (r s : ℝ → E) (v w : E) (y : E →L[ℝ] V)
    (hr : HasDerivAt r v 0) (hs : HasDerivAt s w 0)
    (hG : HasFDerivAt G dG (r 0, s 0))
    (hH : HasFDerivAt H dH (r 0, s 0))
    (root : ∀ t, G (r t, s t) = 0)
    (adjoint : ∀ u, dH (0, u) = y (dG (0, u))) :
    HasDerivAt (fun t => H (r t, s t))
      (dH (v, 0) - y (dG (v, 0))) 0 := by
  have hpair := hr.prodMk hs
  have hroot : HasDerivAt (fun t => G (r t, s t)) (dG (v, w)) 0 :=
    HasFDerivAt.comp_hasDerivAt 0 hG hpair
  have hconst : HasDerivAt (fun t => G (r t, s t)) 0 0 := by
    simpa only [root] using (hasDerivAt_const (𝕜 := ℝ) 0 (0 : E))
  have hz : dG (v, w) = 0 := hroot.unique hconst
  have hsplit : (v, w) = (v, 0) + (0, w) := by simp
  have hg : dG (v, 0) + dG (0, w) = 0 := by
    rw [← map_add, ← hsplit, hz]
  have hy : y (dG (v, 0)) + y (dG (0, w)) = 0 := by
    rw [← map_add, hg, map_zero]
  have hd : dH (v, w) = dH (v, 0) - y (dG (v, 0)) := by
    calc
      dH (v, w) = dH (v, 0) + dH (0, w) := by
        rw [← map_add, ← hsplit]
      _ = dH (v, 0) + y (dG (0, w)) := by rw [adjoint]
      _ = dH (v, 0) - y (dG (v, 0)) := by
        calc
          dH (v, 0) + y (dG (0, w)) =
              dH (v, 0) - y (dG (v, 0)) +
                (y (dG (v, 0)) + y (dG (0, w))) := by abel
          _ = _ := by rw [hy, add_zero]
  have hderiv : HasDerivAt (fun t => H (r t, s t)) (dH (v, w)) 0 :=
    HasFDerivAt.comp_hasDerivAt 0 hH hpair
  rw [hd] at hderiv
  exact hderiv

end LeanDirac.V4.CurrentAdvanceSensitivity

#print axioms LeanDirac.V4.CurrentAdvanceSensitivity.current_advance_adjoint
