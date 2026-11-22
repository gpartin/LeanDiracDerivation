/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module
public import LeanDirac.LocalizedCarrier

@[expose] public section
namespace LeanDirac.V2
open BigOperators Complex
noncomputable section

/-- Independent aligned amplitudes at the eight carrier sites. -/
def cornerState (r : ℝ) (a b : Cell8) : State Site27 where
  psi s := if Site27IsCarrier s then
    c3ScaleComplex (a (site27Corner s)) (tetraCorner (site27Corner s)) else c3Zero
  chi s := if Site27IsCarrier s then
    c3ScaleComplex (b (site27Corner s)) (companion (site27Corner s)) else
    c3ScaleComplex (r : ℂ) haloDirection

/-- Signed area of the independent pair of corner amplitudes. -/
def signedArea (a b : Cell8) := extendCarrierAmplitude (fun i => -(b i * a i))

theorem corner_wedge (r : ℝ) (a b : Cell8) (s : Site27) :
    wedge ((cornerState r a b).chi s) ((cornerState r a b).psi s) =
      c3ScaleComplex (signedArea a b s) unitArea := by
  by_cases h : Site27IsCarrier s
  · simp only [cornerState, h, if_true, signedArea, extendCarrierAmplitude,
      wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
    funext k
    simp [c3ScaleComplex, c3Neg]
    ring
  · simp [cornerState, h, signedArea, extendCarrierAmplitude]

variable {S : Type} [Fintype S] [DecidableEq S]

private def impulse (s : S) (v : C3) : S → C3 := fun k => if k = s then v else c3Zero

private theorem psi_impulse (vel : Velocity S) (s : S) (v : C3) :
    psiMomentumPair vel (impulse s v) = c3RealPair (vel.psiDot s) v := by
  unfold psiMomentumPair
  rw [Finset.sum_eq_single s]
  · simp [impulse]
  · intro k _ hk
    simp [impulse, hk]
  · simp

private theorem chi_impulse (B : ℝ) (vel : Velocity S) (s : S) (v : C3) :
    chiMomentumPair B vel (impulse s v) = B * c3RealPair (vel.chiDot s) v := by
  unfold chiMomentumPair
  rw [Finset.sum_eq_single s]
  · simp [impulse]
  · intro k _ hk
    simp [impulse, hk]
  · simp

omit [Fintype S] [DecidableEq S] in
private theorem area_bond (g : C19Lattice S) (r : ℝ) (st : State S) (q : S → ℂ)
    (hw : ∀ s, wedge (st.chi s) (st.psi s) = c3ScaleComplex (q s) unitArea) (d : State S)
    (e : C19Edge S) :
    exteriorAreaBondFirstVariation r st d (c19Bond g e) =
      c19Weight e.2 / r ^ 2 * (star (scalarC19Difference g
        q e) * scalarC19Difference g
        (fun s => wedgeFirstVariation st d s 0) e).re := by
  simp only [exteriorAreaBondFirstVariation, bondAreaDifference, bondAreaFirstVariation,
    hw, c19Bond]
  congr 1
  simp [c3RealPair, difference, c3ScaleComplex, unitArea, Fin.sum_univ_succ,
    scalarC19Difference, Complex.mul_re, sub_mul, mul_sub]

omit [DecidableEq S] in
theorem corner_potential_variation (g : C19Lattice S) (r : ℝ) (st : State S) (q : S → ℂ)
    (hw : ∀ s, wedge (st.chi s) (st.psi s) = c3ScaleComplex (q s) unitArea) (d : State S) :
    potentialFirstVariation r st d (c19Bond g) =
      (∑ s, sitePotentialFirstVariation r st d s) +
      2 / r ^ 2 * ∑ s, (star (scalarC19Laplacian g
        q s) * wedgeFirstVariation st d s 0).re := by
  unfold potentialFirstVariation
  congr 1
  simp_rw [area_bond g r st q hw]
  have h := c19_scalar_green_real_identity g q
    (fun s => wedgeFirstVariation st d s 0)
  calc
    _ = (1/r^2) * ∑ e : C19Edge S, c19Weight e.2 *
        (star (scalarC19Difference g q e) *
        scalarC19Difference g
          (fun s => wedgeFirstVariation st d s 0) e).re := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro e _
      ring
    _ = _ := by rw [h]; ring

/-- Pointwise identification of the weighted native Psi acceleration, against
every complex three-component direction including exterior and normal ones. -/
theorem psi_acc_pair (g : C19Lattice S) (r B : ℝ) (st : State S) (q : S → ℂ)
    (hw : ∀ s, wedge (st.chi s) (st.psi s) = c3ScaleComplex (q s) unitArea) (s : S) (v : C3) :
    c3RealPair ((accelerationVelocity r B st
      (c19Bond g)).psiDot s) v =
    -((density (st.chi s) + density (st.psi s)) *
      c3RealPair (st.psi s) v +
      2 / r^2 * (star (scalarC19Laplacian g q s) *
        wedge (st.chi s) v 0).re) := by
  rw [← psi_impulse, psi_acceleration_all_directions]
  unfold psiEulerLagrangeForce lagrangianStateFirstVariation
  rw [corner_potential_variation g r st q hw]
  congr 1
  congr 1
  · rw [Finset.sum_eq_single s]
    · simp [sitePotentialFirstVariation, purePsiDirection, impulse]
    · intro k _ hk
      simp [sitePotentialFirstVariation, purePsiDirection, impulse, hk]
    · simp
  · rw [Finset.sum_eq_single s]
    · simp [wedgeFirstVariation, purePsiDirection, impulse]
    · intro k _ hk
      simp [wedgeFirstVariation, purePsiDirection, impulse, hk, c3Zero]
    · simp

/-- Pointwise identification of the weighted native chi acceleration against
the same unrestricted family of ambient test directions. -/
theorem chi_acc_pair (g : C19Lattice S) (r B : ℝ) (hB : B ≠ 0) (st : State S) (q : S → ℂ)
    (hw : ∀ s, wedge (st.chi s) (st.psi s) = c3ScaleComplex (q s) unitArea) (s : S) (v : C3) :
    B * c3RealPair ((accelerationVelocity r B st
      (c19Bond g)).chiDot s) v =
    -((density (st.chi s) - r^2 + density (st.psi s)) *
      c3RealPair (st.chi s) v +
      2 / r^2 * (star (scalarC19Laplacian g q s) *
        wedge v (st.psi s) 0).re) := by
  rw [← chi_impulse, chi_acceleration_all_directions r B _ _ hB]
  unfold chiEulerLagrangeForce lagrangianStateFirstVariation
  rw [corner_potential_variation g r st q hw]
  congr 1
  congr 1
  · rw [Finset.sum_eq_single s]
    · simp [sitePotentialFirstVariation, pureChiDirection, impulse]
    · intro k _ hk
      simp [sitePotentialFirstVariation, pureChiDirection, impulse, hk]
    · simp
  · rw [Finset.sum_eq_single s]
    · simp [wedgeFirstVariation, pureChiDirection, impulse, c3Add, c3Zero]
    · intro k _ hk
      simp [wedgeFirstVariation, pureChiDirection, impulse, hk, c3Add, c3Zero]
    · simp

end
end LeanDirac.V2

