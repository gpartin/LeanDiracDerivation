/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/
module
public import LeanDirac.V2.BalancedClock
public import LeanDirac.V2.BoxClosure
public import LeanDirac.V2.Transport

@[expose] public section
namespace LeanDirac.V2
open Complex
noncomputable section

/-- One exact scalar clock, including the current amplitude and time layer. -/
def clockAmplitude (x dt k : ℝ) (n : ℕ) : ℂ :=
  harmonicPhaseStep dt k ^ n * (Real.sqrt x : ℝ)

private theorem clock_norm (x dt k : ℝ) (n : ℕ) (h : 0 ≤ centeredPhaseRateSq dt k) :
    normSq (clockAmplitude x dt k n) = normSq ((Real.sqrt x : ℝ) : ℂ) := by
  simp [clockAmplitude, map_pow, (harmonic_phase_step_laws dt k h).1]

private theorem clock_recurrence (x dt k : ℝ) (n : ℕ) (h : 0 ≤ centeredPhaseRateSq dt k) :
    clockAmplitude x dt k (n+2) = 2 * clockAmplitude x dt k (n+1) -
      clockAmplitude x dt k n + (dt^2 : ℂ) * (-(k : ℂ) * clockAmplitude x dt k (n+1)) := by
  have eq := (harmonic_phase_step_laws dt k h).2
  simp only [clockAmplitude, pow_add, pow_one]
  rw [eq]
  push_cast
  ring

private theorem clock_rates_nonneg (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hd : dt^2*r^2/4 < 1) :
    0 ≤ centeredPhaseRateSq dt (balancedPsiFrequencySq r B dt) ∧
    0 ≤ centeredPhaseRateSq dt (balancedChiFrequencySq r B dt) := by
  obtain ⟨hv,hvu,hu,_,_⟩ := balanced_clock_rates r B dt hr hB hd
  constructor
  · exact mul_nonneg (le_of_lt (hv.trans hvu)) hu.le
  · apply mul_nonneg hv.le
    nlinarith [mul_nonneg (sq_nonneg dt) (sub_nonneg.mpr hvu.le)]

/-- Two independently rotating native phases on the fixed eight-site carrier. -/
def balancedRest27 (r B dt : ℝ) (n : ℕ) : State Site27 :=
  localizedCarrierState r
    (clockAmplitude (balancedAmplitudeSq r B dt) dt (balancedPsiFrequencySq r B dt) n)
    (clockAmplitude (balancedAmplitudeSq r B dt) dt (balancedChiFrequencySq r B dt) n)

private theorem localized_next_coefficients (r B dt : ℝ) (hB : B ≠ 0)
    (aPrev bPrev a b aNext bNext : ℂ)
    (ha : aNext = 2*a-aPrev + (dt^2 : ℂ)*localizedPsiCommonAcceleration r a b)
    (hb : bNext = 2*b-bPrev + (dt^2 : ℂ)*localizedChiCommonAcceleration r B a b) :
    localizedCarrierState r aNext bNext = nativeNextState r B dt (c19Bond site27C19Geometry)
      (localizedCarrierState r aPrev bPrev) (localizedCarrierState r a b) := by
  rw [ha,hb]
  apply congrArg₂ State.mk <;> funext s j
  · change _ = 2 * _ - _ + _ * (accelerationVelocity r B (localizedCarrierState r a b) _).psiDot s j
    rw [localized_psi_acceleration_all_sites]
    by_cases hs : Site27IsCarrier s
    · simp [localizedCarrierState, hs, c3ScaleComplex]
      ring
    · simp [localizedCarrierState, hs, c3Zero]
  · change _ = 2 * _ - _ + _ * (accelerationVelocity r B (localizedCarrierState r a b) _).chiDot s j
    rw [localized_chi_acceleration_all_sites r B hB]
    by_cases hs : Site27IsCarrier s <;> simp [localizedCarrierState, hs, c3ScaleComplex,c3Zero] <;> ring

theorem balanced_rest_native_step {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B) (hd : dt^2*r^2/4 < 1) (n : ℕ) :
    boxLift (balancedRest27 r B dt (n+2)) =
      nativeNextState r B dt (c19Bond (periodicC19Geometry L))
        (boxLift (balancedRest27 r B dt n)) (boxLift (balancedRest27 r B dt (n+1))) := by
  let x := balancedAmplitudeSq r B dt
  let u := balancedPsiFrequencySq r B dt
  let v := balancedChiFrequencySq r B dt
  let a := clockAmplitude x dt u (n+1)
  let b := clockAmplitude x dt v (n+1)
  have nz : B ≠ 0 := by linarith
  obtain ⟨hu,hv⟩ := clock_rates_nonneg r B dt hr hB hd
  have sa := clock_norm x dt u (n+1) hu
  have sb := clock_norm x dt v (n+1) hv
  have ks := balanced_native_stiffnesses r B dt hr hB hd
  have kp : localizedPsiStiffness r a b = u := by
    simpa only [a,b,localizedPsiStiffness,sa,sb,x] using! ks.1
  have kc : localizedChiStiffness r a b / B = v := by
    simpa only [a,b,localizedChiStiffness,sa,sb,x] using! ks.2
  apply Eq.trans _ (box_native_next hL r B dt nz (balancedRest27 r B dt n)
    (fun i => a * localizedTwistMinus i) (fun _ => b)).symm
  apply congrArg boxLift
  apply localized_next_coefficients r B dt nz
  · change clockAmplitude x dt u (n+2) = 2*a - _ + _ * (-(localizedPsiStiffness r a b : ℂ)*a)
    rw [kp]
    exact clock_recurrence x dt u n hu
  · change clockAmplitude x dt v (n+2) = 2*b - _ + _ * (-((localizedChiStiffness r a b / B : ℝ) : ℂ)*b)
    rw [kc]
    exact clock_recurrence x dt v n hv

private theorem phase_momentum (q z : ℂ) (u : C3) :
    c3RealPair (c3ScaleComplex (q*z) u) (phaseTurn (c3ScaleComplex z u)) =
      q.im * normSq z * density u := by
  have turn : phaseTurn (c3ScaleComplex z u) = c3ScaleComplex (I*z) u := by
    funext k
    simp only [phaseTurn, c3ScaleComplex, mul_assoc]
  rw [turn, c3_real_pair_eq_inner_re, c3_inner_scale_left, c3_inner_scale_right, c3_inner_self]
  simp [Complex.mul_re, Complex.mul_im, normSq]
  ring

private theorem initial_imbalance (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hd : dt^2*r^2/4 < 1) (s : Site27) :
    imbalance B (balancedRest27 r B dt 0) (balancedRest27 r B dt 1) s = 0 := by
  let x := balancedAmplitudeSq r B dt
  let u := balancedPsiFrequencySq r B dt
  let v := balancedChiFrequencySq r B dt
  let z : ℂ := (Real.sqrt x : ℝ)
  obtain ⟨hu,hv⟩ := clock_rates_nonneg r B dt hr hB hd
  obtain ⟨_,_,_,_,bal⟩ := balanced_clock_rates r B dt hr hB hd
  have signed : (harmonicPhaseStep dt u).im = B * (harmonicPhaseStep dt v).im := by
    change dt * Real.sqrt (centeredPhaseRateSq dt u) = B * (dt * Real.sqrt (centeredPhaseRateSq dt v))
    rw [show centeredPhaseRateSq dt u = B^2 * centeredPhaseRateSq dt v from bal,
      Real.sqrt_mul (sq_nonneg B), Real.sqrt_sq (show 0 ≤ B by linarith)]
    ring
  by_cases hs : Site27IsCarrier s
  · simp only [imbalance, balancedRest27, localizedCarrierState, hs, if_true, clockAmplitude,
      pow_zero, pow_one, one_mul]
    change c3RealPair (c3ScaleComplex ((harmonicPhaseStep dt u * z)*_) _) _ -
      B*c3RealPair (c3ScaleComplex (harmonicPhaseStep dt v * z) _) _ = 0
    rw [mul_assoc, phase_momentum, phase_momentum, Complex.normSq_mul,
      localized_twist_minus_norm, tetra_corner_density_one, companion_density_one, signed]
    ring
  · have halo := phase_momentum 1 (r : ℂ) haloDirection
    simp only [one_mul, Complex.one_im, zero_mul] at halo
    simp only [imbalance, balancedRest27, localizedCarrierState, hs, if_false, halo,
      mul_zero, sub_zero]
    simp [c3Zero, phaseTurn, c3RealPair]

/-- The explicit two-field history solves the unrestricted action-derived
update on every tick and has zero signed local charge imbalance at every site. -/
theorem balanced_rest_orbit {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B) (hd : dt^2*r^2/4 < 1) :
    let q : ℕ → State (PeriodicCubicSite L) := fun n => boxLift (balancedRest27 r B dt n)
    (∀ n, q (n+2) = nativeNextState r B dt (c19Bond (periodicC19Geometry L))
      (q n) (q (n+1))) ∧ (∀ n s, imbalance B (q n) (q (n+1)) s = 0) := by
  dsimp only
  let q : ℕ → State (PeriodicCubicSite L) := fun n => boxLift (balancedRest27 r B dt n)
  have step := balanced_rest_native_step hL r B dt hr hB hd
  refine ⟨step, ?_⟩
  intro n s
  rw [history_site_imbalance r B dt (c19Bond (periodicC19Geometry L)) (by linarith) q step s n]
  exact initial_imbalance r B dt hr hB hd (boxView s)

/-- With a positive step the exact solution is running. Each core field has
a positive signed phase advance, even though the two weighted momenta cancel. -/
theorem balanced_phases_advance (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hdt : 0 < dt) (hd : dt^2*r^2/4 < 1) :
    0 < (harmonicPhaseStep dt (balancedPsiFrequencySq r B dt)).im ∧
    0 < (harmonicPhaseStep dt (balancedChiFrequencySq r B dt)).im := by
  obtain ⟨hv,hvu,hu,_,_⟩ := balanced_clock_rates r B dt hr hB hd
  constructor
  · exact mul_pos hdt (Real.sqrt_pos.mpr (mul_pos (hv.trans hvu) hu))
  · apply mul_pos hdt (Real.sqrt_pos.mpr (mul_pos hv ?_))
    nlinarith [mul_nonneg (sq_nonneg dt) (sub_nonneg.mpr hvu.le)]

end
end LeanDirac.V2

