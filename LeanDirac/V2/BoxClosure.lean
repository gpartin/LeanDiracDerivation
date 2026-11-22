/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module
public import LeanDirac.V2.BoxGeometry
public import LeanDirac.V2.CarrierForces
public import LeanDirac.NativeLocality

@[expose] public section
namespace LeanDirac.V2
open BigOperators Complex
noncomputable section

/-- Extend the eight carrier sites into an arbitrary periodic box using
the same spatially constant exterior fields as the reference state. -/
def boxLift {L : ℕ} (st : State Site27) : State (PeriodicCubicSite L) where
  psi s := st.psi (boxView s)
  chi s := st.chi (boxView s)

private theorem pair_ext (x y : C3)
    (h : ∀ v, c3RealPair x v = c3RealPair y v) : x = y := by
  have eq : c3RealPair (c3Sub x y) (c3Sub x y) = 0 := by
    calc
      _ = c3RealPair x (c3Sub x y) - c3RealPair y (c3Sub x y) := by
        simp only [c3RealPair, c3Sub, sub_mul, Complex.sub_re, Finset.sum_sub_distrib]
      _ = 0 := sub_eq_zero.mpr (h _)
  have hz := density_zero_implies_c3_zero _ ((c3_real_pair_self _).symm.trans eq)
  funext k
  exact sub_eq_zero.mp (congrFun hz k)

/-- At a carrier site the entire incident scalar force agrees with the
reference box. At an exterior site both local environments can differ, but
the two exterior-area force components vanish for every test direction. -/
theorem box_corner_acceleration {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B : ℝ) (hB : B ≠ 0) (a b : Cell8) (s : PeriodicCubicSite L) :
    let st := cornerState r a b
    let acc := accelerationVelocity r B (boxLift st) (c19Bond (periodicC19Geometry L))
    let ref := accelerationVelocity r B st (c19Bond site27C19Geometry)
    acc.psiDot s = ref.psiDot (boxView s) ∧ acc.chiDot s = ref.chiDot (boxView s) := by
  let st := cornerState r a b
  let q := signedArea a b
  have hw := corner_wedge r a b
  have hb (t : PeriodicCubicSite L) :
      wedge ((boxLift st).chi t) ((boxLift st).psi t) =
        c3ScaleComplex ((q ∘ boxView) t) unitArea :=
    hw (boxView t)
  constructor
  · apply pair_ext
    intro v
    rw [psi_acc_pair (periodicC19Geometry L) r B (boxLift st) (q ∘ boxView) hb,
      psi_acc_pair site27C19Geometry r B st q hw]
    simp only [boxLift]
    by_cases hi : Site27IsCarrier (boxView s)
    · rw [box_laplacian_at_carrier hL s hi q]
    · simp [st, cornerState, hi, c3ScaleComplex, haloDirection, basis, wedge]
  · apply pair_ext
    intro v
    apply mul_left_cancel₀ hB
    rw [chi_acc_pair (periodicC19Geometry L) r B hB (boxLift st) (q ∘ boxView) hb,
      chi_acc_pair site27C19Geometry r B hB st q hw]
    simp only [boxLift]
    by_cases hi : Site27IsCarrier (boxView s)
    · rw [box_laplacian_at_carrier hL s hi q]
    · simp [st, cornerState, hi, c3Zero, wedge]

/-- Exact commutation of the full synchronous update with box enlargement,
including all complex components of the exterior at every site. -/
theorem box_native_next {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hB : B ≠ 0) (previous : State Site27) (a b : Cell8) :
    nativeNextState r B dt (c19Bond (periodicC19Geometry L))
        (boxLift previous) (boxLift (cornerState r a b)) =
      boxLift (nativeNextState r B dt (c19Bond site27C19Geometry) previous
        (cornerState r a b)) := by
  apply congrArg₂ State.mk <;> funext s k
  · change 2 * _ - _ + _ * _ = 2 * _ - _ + _ * _
    rw [(box_corner_acceleration hL r B hB a b s).1]
    rfl
  · change 2 * _ - _ + _ * _ = 2 * _ - _ + _ * _
    rw [(box_corner_acceleration hL r B hB a b s).2]
    rfl

/-- An arbitrary state which agrees locally with the lifted reference has
the same next fields at this site, even if every more distant field differs.
The first equality uses the native locality theorem shared with DIRAC-035. -/
theorem box_step_from_neighborhood {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hB : B ≠ 0) (p : State Site27) (a b : Cell8)
    (prev curr : State (PeriodicCubicSite L)) (s : PeriodicCubicSite L)
    (hp : prev.psi s = (boxLift p).psi s) (hc : prev.chi s = (boxLift p).chi s)
    (h : C19LocalAgreement (periodicC19Geometry L) curr (boxLift (cornerState r a b)) s) :
    let out := nativeNextState r B dt (c19Bond (periodicC19Geometry L)) prev curr
    let ref := nativeNextState r B dt (c19Bond site27C19Geometry) p (cornerState r a b)
    out.psi s = ref.psi (boxView s) ∧ out.chi s = ref.chi (boxView s) := by
  have eq := native_c19_update_is_local r B dt (periodicC19Geometry L)
    prev (boxLift p) curr (boxLift (cornerState r a b)) s hp hc h
  rw [box_native_next hL r B dt hB] at eq
  exact eq

/-- The reference two-amplitude sector on every cube size. Only the starting
layers are assumed to have this form; every site's next value is dynamical. -/
def InBoxCarrierSector {L : ℕ} (r : ℝ) (st : State (PeriodicCubicSite L)) : Prop :=
  ∃ a b : ℂ, st = boxLift (localizedCarrierState r a b)

theorem box_preserves_localized_carrier {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hB : B ≠ 0) (p c : State (PeriodicCubicSite L))
    (hp : InBoxCarrierSector r p) (hc : InBoxCarrierSector r c) :
    InBoxCarrierSector r (nativeNextState r B dt (c19Bond (periodicC19Geometry L)) p c) := by
  obtain ⟨ap, bp, rfl⟩ := hp
  obtain ⟨ac, bc, rfl⟩ := hc
  obtain ⟨an, bn, hn⟩ := native_next_preserves_localized_carrier_sector r B dt hB
    (localizedCarrierState r ap bp) (localizedCarrierState r ac bc)
    ⟨ap, bp, rfl⟩ ⟨ac, bc, rfl⟩
  refine ⟨an, bn, ?_⟩
  exact (box_native_next hL r B dt hB (localizedCarrierState r ap bp)
    (fun i => ac * localizedTwistMinus i) (fun _ => bc)).trans (congrArg boxLift hn)

/-- On the reference carrier, two matched starting layers determine identical
site fields under box enlargement at every native tick. -/
theorem box_trajectory_matches {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hB : B ≠ 0) (ref : ℕ → State Site27)
    (tr : ℕ → State (PeriodicCubicSite L))
    (hr : ∀ n, ref (n+2) = nativeNextState r B dt (c19Bond site27C19Geometry)
      (ref n) (ref (n+1)))
    (ht : ∀ n, tr (n+2) = nativeNextState r B dt (c19Bond (periodicC19Geometry L))
      (tr n) (tr (n+1)))
    (h0 : InLocalizedCarrierSector r (ref 0)) (h1 : InLocalizedCarrierSector r (ref 1))
    (m0 : tr 0 = boxLift (ref 0)) (m1 : tr 1 = boxLift (ref 1)) (n : ℕ) :
    tr n = boxLift (ref n) := by
  let P := fun k => InLocalizedCarrierSector r (ref k) ∧ tr k = boxLift (ref k)
  have pair (k : ℕ) : P k ∧ P (k+1) := by
    induction k with
    | zero => exact ⟨⟨h0, m0⟩, ⟨h1, m1⟩⟩
    | succ k ih =>
      obtain ⟨⟨hp, ep⟩, hc, ec⟩ := ih
      refine ⟨⟨hc, ec⟩, ?_, ?_⟩
      · rw [hr]
        exact native_next_preserves_localized_carrier_sector r B dt hB _ _ hp hc
      · obtain ⟨a,b,h⟩ := hc
        rw [ht, hr, ep, ec, h]
        exact box_native_next hL r B dt hB _ (fun i => a * localizedTwistMinus i)
          (fun _ => b)
  exact (pair n).1.2

end
end LeanDirac.V2

