/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import Mathlib.Analysis.Calculus.Deriv.Basic
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum

@[expose] public section

namespace LeanDirac.V4.FacePhaseMomentum

noncomputable section

/-! # Local phase momentum of the matched face oscillator

Research lemma for one complex component of the existing V4 face action.
The full C3 action sums these components and the wedge's two phase
generators. The lemma does not assume the Dirac equation or prove a
three-dimensional moving carrier. It is not a V4-006 pass certificate.
-/

/-- Real coordinates of one complex component, with its native phase tangent. -/
abbrev Plane := ℝ × ℝ

def normSq (z : Plane) : ℝ := z.1 ^ 2 + z.2 ^ 2

def inner (z w : Plane) : ℝ := z.1 * w.1 + z.2 * w.2

def cross (z w : Plane) : ℝ := z.1 * w.2 - z.2 * w.1

def sub (z w : Plane) : Plane := (z.1 - w.1, z.2 - w.2)

def scale (a : ℝ) (z : Plane) : Plane := (a * z.1, a * z.2)

def phaseVariation (z : Plane) (s : ℝ) : Plane :=
  (z.1 - s * z.2, z.2 + s * z.1)

/-- The exact matched face oscillator, using `k = B / dt` and
`c = 1 - dt² chi0² / 2`. The two bond differences are oriented i -> j. -/
def intervalAction (k c : ℝ) (li lj ri rj : Plane) : ℝ :=
  let dl := sub lj li
  let dr := sub rj ri
  k / 2 * (c * (normSq dl + normSq dr) - 2 * inner dl dr)

def rightMomentum (k c : ℝ) (li lj ri rj : Plane) : ℝ :=
  k * cross rj (sub (scale c (sub rj ri)) (sub lj li))

def leftMomentum (k c : ℝ) (li lj ri rj : Plane) : ℝ :=
  k * cross lj (sub (scale c (sub lj li)) (sub rj ri))

/-- The coefficient of the endpoint's local phase variation is derived
from the interval polynomial, rather than assigned as a current. -/
theorem right_variation_expansion (k c t : ℝ) (li lj ri rj : Plane) :
    intervalAction k c li lj ri (phaseVariation rj t) =
      intervalAction k c li lj ri rj + t * rightMomentum k c li lj ri rj +
        (k * c / 2) * t ^ 2 * normSq rj := by
  unfold intervalAction phaseVariation rightMomentum normSq cross inner sub scale
  dsimp
  ring

theorem left_variation_expansion (k c t : ℝ) (li lj ri rj : Plane) :
    intervalAction k c li (phaseVariation lj t) ri rj =
      intervalAction k c li lj ri rj + t * leftMomentum k c li lj ri rj +
        (k * c / 2) * t ^ 2 * normSq lj := by
  unfold intervalAction phaseVariation leftMomentum normSq cross inner sub scale
  dsimp
  ring

/-- The ordinary derivative of the phase variation evaluates to the
native right-endpoint momentum. -/
theorem right_momentum_is_variation (k c : ℝ) (li lj ri rj : Plane) :
    HasDerivAt (fun t => intervalAction k c li lj ri (phaseVariation rj t))
      (rightMomentum k c li lj ri rj) 0 := by
  rw [show (fun t => intervalAction k c li lj ri (phaseVariation rj t)) =
    (fun t => intervalAction k c li lj ri rj + t * rightMomentum k c li lj ri rj +
      (k * c / 2) * t ^ 2 * normSq rj) from funext (fun t => right_variation_expansion k c t li lj ri rj)]
  have hconst := hasDerivAt_const (𝕜 := ℝ) 0 (intervalAction k c li lj ri rj)
  have hlinear := (hasDerivAt_id (0 : ℝ)).mul_const (rightMomentum k c li lj ri rj)
  have hquadratic := (HasDerivAt.const_mul (k * c / 2)
    (hasDerivAt_pow 2 (0 : ℝ))).mul_const (normSq rj)
  have h := HasDerivAt.add (HasDerivAt.add hconst hlinear) hquadratic
  convert h using 1
  · funext t
    simp only [Pi.add_apply, id_eq]
  · norm_num

/-- The signed inflow to j is the shared phase variation of both time
endpoints of j. Swapping the face reverses its sign. -/
def inflow (k c : ℝ) (li lj ri rj : Plane) : ℝ :=
  leftMomentum k c li lj ri rj + rightMomentum k c li lj ri rj

theorem inflow_antisymmetric (k c : ℝ) (li lj ri rj : Plane) :
    inflow k c li lj ri rj = -inflow k c lj li rj ri := by
  unfold inflow leftMomentum rightMomentum cross sub scale
  dsimp
  ring

/-- Explicit signed momentum flow: this is the phase derivative of the
same face action, not an independently introduced bond rule. -/
theorem inflow_explicit (k c : ℝ) (li lj ri rj : Plane) :
    inflow k c li lj ri rj = k *
      (cross lj ri + cross rj li - c * (cross lj li + cross rj ri)) := by
  unfold inflow leftMomentum rightMomentum cross sub scale
  dsimp
  ring

/-- Exact two-site exchange. The donor/receiver have opposite phase and
their amplitudes sum to A at both times. The bond retains its rest-clock
oscillation while its endpoint phase momentum migrates from i to j.
The result is local to that bond and makes no claim about its other five
faces or the full two-field action. -/
theorem antiphase_pair_exact_transfer
    (k c s A a0 b0 a1 b1 : ℝ)
    (h0 : a0 + b0 = A) (h1 : a1 + b1 = A) :
    let li : Plane := (a0, 0)
    let lj : Plane := (-b0, 0)
    let ri : Plane := (c * a1, s * a1)
    let rj : Plane := (-c * b1, -s * b1)
    leftMomentum k c li lj ri rj = -k * s * A * b0 ∧
      rightMomentum k c li lj ri rj = k * s * A * b1 ∧
      inflow k c li lj ri rj = k * s * A * (b1 - b0) := by
  have ha0 : a0 = A - b0 := by linarith
  have ha1 : a1 = A - b1 := by linarith
  rw [ha0, ha1]
  dsimp [leftMomentum, rightMomentum, inflow, cross, sub, scale]
  constructor
  · ring
  constructor <;> ring

/-- The shared antiphase difference obeys the centered matched oscillator
at the middle layer, even if the donor and recipient divide its amplitude
differently at the three times. Other incident faces are not included. -/
theorem constant_antiphase_difference_clock_del_zero
    (k c s A aM bM a0 b0 aP bP : ℝ)
    (hM : aM + bM = A) (h0 : a0 + b0 = A) (hP : aP + bP = A) :
    let dM := sub (-c * bM, s * bM) (c * aM, -s * aM)
    let d0 := sub (-b0, 0) (a0, 0)
    let dP := sub (-c * bP, -s * bP) (c * aP, s * aP)
    scale k (sub (sub (scale (2 * c) d0) dM) dP) = (0, 0) := by
  have haM : aM = A - bM := by linarith
  have ha0 : a0 = A - b0 := by linarith
  have haP : aP = A - bP := by linarith
  rw [haM, ha0, haP]
  simp only [sub, scale]
  apply Prod.ext <;> dsimp <;> ring

end
end LeanDirac.V4.FacePhaseMomentum

#print axioms LeanDirac.V4.FacePhaseMomentum.right_momentum_is_variation
#print axioms LeanDirac.V4.FacePhaseMomentum.inflow_antisymmetric
#print axioms LeanDirac.V4.FacePhaseMomentum.antiphase_pair_exact_transfer
#print axioms LeanDirac.V4.FacePhaseMomentum.constant_antiphase_difference_clock_del_zero
