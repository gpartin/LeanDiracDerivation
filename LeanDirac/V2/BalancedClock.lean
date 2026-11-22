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

/-- Positive squared Psi frequency chosen from the two native recurrences
by equality of their inertia-weighted squared phase rates. -/
def balancedPsiFrequencySq (r B dt : ℝ) : ℝ :=
  (B * r^2 + dt^2 * r^4 / 4) / (B - 1 + dt^2 * r^2 / 2)

def balancedChiFrequencySq (r B dt : ℝ) : ℝ :=
  (balancedPsiFrequencySq r B dt - r^2) / B

/-- Equal squared amplitude of the two freely evolving native fields. -/
def balancedAmplitudeSq (r B dt : ℝ) : ℝ :=
  balancedPsiFrequencySq r B dt / (2 + 54 / r^2)

/-- The squared sine quadrature per unit time for a stable centered
harmonic recurrence of squared frequency k. -/
def centeredPhaseRateSq (dt k : ℝ) : ℝ := k * (1 - dt^2 * k / 4)

theorem balanced_clock_rates (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hd : dt^2 * r^2 / 4 < 1) :
    let u := balancedPsiFrequencySq r B dt
    let v := balancedChiFrequencySq r B dt
    0 < v ∧ v < u ∧ 0 < 1 - dt^2 * u / 4 ∧
      0 < balancedAmplitudeSq r B dt ∧
      centeredPhaseRateSq dt u = B^2 * centeredPhaseRateSq dt v := by
  let p := dt^2 * r^2 / 4
  let d := B - 1 + dt^2 * r^2 / 2
  let n := B * r^2 + dt^2 * r^4 / 4
  have hu : balancedPsiFrequencySq r B dt = n/d := rfl
  have hp : 0 ≤ p := by dsimp [p]; positivity
  have hd' : 0 < d := by
    dsimp [d]
    positivity
  have hv : balancedChiFrequencySq r B dt = r^2 * (1-p) / (B*d) := by
    rw [balancedChiFrequencySq, hu]
    field_simp [show B ≠ 0 by linarith, ne_of_gt hd']
    dsimp [n, d, p]
    ring
  have huv : balancedPsiFrequencySq r B dt = r^2 + B * balancedChiFrequencySq r B dt := by
    simp only [balancedChiFrequencySq]
    field_simp [show B ≠ 0 by linarith]
    ring
  have hpos : 0 < balancedChiFrequencySq r B dt := by
    rw [hv]
    exact div_pos (mul_pos (sq_pos_of_pos hr) (sub_pos.mpr hd)) (mul_pos (by linarith) hd')
  have hlt : balancedChiFrequencySq r B dt < balancedPsiFrequencySq r B dt := by
    rw [huv]
    nlinarith [sq_pos_of_pos hr, mul_pos (sub_pos.mpr hB) hpos]
  have hspace : 1 - dt^2 * balancedPsiFrequencySq r B dt / 4 =
      ((1-p) * (B-1+p)) / d := by
    rw [hu]
    field_simp [ne_of_gt hd']
    dsimp [n, d, p]
    ring
  refine ⟨hpos, hlt, hspace.symm ▸ ?_, ?_, ?_⟩
  · exact div_pos (mul_pos (sub_pos.mpr hd) (by linarith only [hB,hp])) hd'
  · exact div_pos (lt_trans hpos hlt) (by positivity)
  · dsimp only [centeredPhaseRateSq, balancedChiFrequencySq]
    rw [hu]
    field_simp [show B ≠ 0 by linarith, ne_of_gt hd']
    dsimp [n, d]
    ring

/-- Both squared frequencies in the balance identity are exactly the native
stiffnesses (including the chi inertia) at one explicit pair of amplitudes. -/
theorem balanced_native_stiffnesses (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hd : dt^2 * r^2 / 4 < 1) :
    let z : ℂ := ((Real.sqrt (balancedAmplitudeSq r B dt) : ℝ) : ℂ)
    localizedPsiStiffness r z z = balancedPsiFrequencySq r B dt ∧
      localizedChiStiffness r z z / B = balancedChiFrequencySq r B dt := by
  have hs := (balanced_clock_rates r B dt hr hB hd).2.2.2.1
  have hnorm : Complex.normSq ((Real.sqrt (balancedAmplitudeSq r B dt) : ℝ) : ℂ) =
      balancedAmplitudeSq r B dt := by
    rw [Complex.normSq_ofReal, ← sq, Real.sq_sqrt (le_of_lt hs)]
  have heq : (2 + 54 / r^2) * balancedAmplitudeSq r B dt = balancedPsiFrequencySq r B dt := by
    unfold balancedAmplitudeSq
    field_simp [show 2 + 54 / r^2 ≠ 0 by positivity]
  dsimp only
  simp only [localizedPsiStiffness, localizedChiStiffness, hnorm, balancedChiFrequencySq]
  constructor
  · linarith only [heq]
  · congr 1
    linarith only [heq]

/-- Under the native equal-amplitude stiffness relation the rate balance
fixes the squared Psi frequency uniquely. -/
theorem balanced_frequency_unique (r B dt u v : ℝ) (hB : 1 < B)
    (hnative : B*v = u-r^2)
    (hbalance : centeredPhaseRateSq dt u = B^2 * centeredPhaseRateSq dt v) :
    u = balancedPsiFrequencySq r B dt := by
  have hd : 0 < B - 1 + dt^2 * r^2 / 2 := by positivity
  apply (eq_div_iff (ne_of_gt hd)).mpr
  have cross : (B*v)^2 = (u-r^2)^2 := congrArg (fun x : ℝ => x^2) hnative
  unfold centeredPhaseRateSq at hbalance
  linear_combination -hbalance - B*hnative + dt^2/4*cross

/-- One unit phase step for a homogeneous centered harmonic recurrence. -/
def harmonicPhaseStep (dt k : ℝ) : ℂ :=
  ⟨1 - dt^2 * k / 2, dt * Real.sqrt (centeredPhaseRateSq dt k)⟩

theorem harmonic_phase_step_laws (dt k : ℝ) (h : 0 ≤ centeredPhaseRateSq dt k) :
    let q := harmonicPhaseStep dt k
    Complex.normSq q = 1 ∧ q^2 = ((2-dt^2*k : ℝ) : ℂ)*q-1 := by
  have hs := Real.sq_sqrt h
  unfold centeredPhaseRateSq at hs
  constructor
  · change (1-dt^2*k/2)*(1-dt^2*k/2) + (dt*_)*(dt*_) = 1
    unfold centeredPhaseRateSq
    linear_combination dt^2 * hs
  · apply Complex.ext
    · simp only [sq, Complex.mul_re, Complex.sub_re, Complex.one_re, Complex.ofReal_re,
        Complex.ofReal_im, harmonicPhaseStep, zero_mul, sub_zero]
      unfold centeredPhaseRateSq
      linear_combination -dt^2 * hs
    · simp [harmonicPhaseStep, sq, Complex.mul_im]
      ring

end
end LeanDirac.V2

