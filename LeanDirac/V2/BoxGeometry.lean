/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module
public import LeanDirac.LocalizedCarrier

@[expose] public section
namespace LeanDirac.V2
noncomputable section

/- The eight sites with coordinates 0 and 1 carry the cube. Every other
coordinate is mapped to the exterior label 2 in the reference calculation. -/
def boxAxis {L : ℕ} (x : ZMod L) : Fin 3 :=
  if x = 0 then 0 else if x = 1 then 1 else 2

def boxView {L : ℕ} (s : PeriodicCubicSite L) : Site27 :=
  (boxAxis (s 0), boxAxis (s 1), boxAxis (s 2))

private theorem short_cast {L : ℕ} (hL : 3 ≤ L) (a b : ℤ)
    (lo : -2 ≤ b-a) (hi : b-a ≤ 2) :
    ((a : ZMod L) = (b : ZMod L)) ↔ a = b := by
  constructor
  · intro h
    have hd := (ZMod.intCast_eq_intCast_iff_dvd_sub a b L).mp h
    have hz : b - a = 0 := Int.eq_zero_of_abs_lt_dvd hd (by
      rw [abs_lt]
      constructor <;> omega)
    omega
  · rintro rfl
    rfl

private theorem box_shift {L : ℕ} (hL : 3 ≤ L) (x : ZMod L) (k : ℤ)
    (hx : (boxAxis x).val < 2) (hk : -1 ≤ k ∧ k ≤ 1) :
    boxAxis (x + (k : ZMod L)) = fin3Shift (boxAxis x) k := by
  have e : x = 0 ∨ x = 1 := by
    by_cases h : x = 0
    · exact Or.inl h
    by_cases h' : x = 1
    · exact Or.inr h'
    simp [boxAxis, h, h'] at hx
  have c1 : (1 : ZMod L) ≠ 0 := by
    simpa using (short_cast hL 1 0 (by decide) (by decide)).not.mpr (by decide)
  have cm : (-1 : ZMod L) ≠ 0 := neg_ne_zero.mpr c1
  have cm1 : (-1 : ZMod L) ≠ 1 := by
    simpa using (short_cast hL (-1) 1 (by decide) (by decide)).not.mpr (by decide)
  have c2 : (2 : ZMod L) ≠ 0 := by
    simpa using (short_cast hL 2 0 (by decide) (by decide)).not.mpr (by decide)
  have c21 : (2 : ZMod L) ≠ 1 := by
    simpa using (short_cast hL 2 1 (by decide) (by decide)).not.mpr (by decide)
  rcases hk with ⟨hl, hh⟩
  rcases e with rfl | rfl <;> interval_cases k <;>
    norm_num [fin3Shift, boxAxis, c1, cm, cm1, c2, c21, fin3Succ, fin3Pred]

theorem box_neighbor_at_carrier {L : ℕ} (hL : 3 ≤ L)
    (s : PeriodicCubicSite L) (hs : Site27IsCarrier (boxView s))
    (d : C19Neighbor) :
    boxView (periodicC19Neighbor L s d) = site27Neighbor (boxView s) d := by
  have bound (j : Fin 3) : -1 ≤ c19Coord j d ∧ c19Coord j d ≤ 1 := by
    fin_cases j <;> fin_cases d <;> decide
  exact Prod.ext (box_shift hL _ _ hs.1 (bound 0)) (Prod.ext
    (box_shift hL _ _ hs.2.1 (bound 1)) (box_shift hL _ _ hs.2.2 (bound 2)))

/-- Exact equality of the entire scalar bond response at each carrier site,
for arbitrary corner data and every periodic box with side at least three. -/
theorem box_laplacian_at_carrier {L : ℕ} (hL : 3 ≤ L)
    (s : PeriodicCubicSite L) (hs : Site27IsCarrier (boxView s))
    (q : Site27 → ℂ) :
    scalarC19Laplacian (periodicC19Geometry L) (q ∘ boxView) s =
      scalarC19Laplacian site27C19Geometry q (boxView s) := by
  unfold scalarC19Laplacian
  apply Finset.sum_congr rfl
  intro d _
  simp only [Function.comp_apply, periodicC19Geometry, site27C19Geometry,
    box_neighbor_at_carrier hL s hs d]

end
end LeanDirac.V2

