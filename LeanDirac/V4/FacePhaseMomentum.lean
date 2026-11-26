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

/-- The left time endpoint has the same independent variational meaning. -/
theorem left_momentum_is_variation (k c : ℝ) (li lj ri rj : Plane) :
    HasDerivAt (fun t => intervalAction k c li (phaseVariation lj t) ri rj)
      (leftMomentum k c li lj ri rj) 0 := by
  rw [show (fun t => intervalAction k c li (phaseVariation lj t) ri rj) =
    (fun t => intervalAction k c li lj ri rj + t * leftMomentum k c li lj ri rj +
      (k * c / 2) * t ^ 2 * normSq lj) from
      funext (fun t => left_variation_expansion k c t li lj ri rj)]
  have hconst := hasDerivAt_const (𝕜 := ℝ) 0 (intervalAction k c li lj ri rj)
  have hlinear := (hasDerivAt_id (0 : ℝ)).mul_const (leftMomentum k c li lj ri rj)
  have hquadratic := (HasDerivAt.const_mul (k * c / 2)
    (hasDerivAt_pow 2 (0 : ℝ))).mul_const (normSq lj)
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


/-- Addition and multiplication in the real-coordinate complex plane.
The native wedge is a difference of such bilinear products. -/
def add (z w : Plane) : Plane := (z.1 + w.1, z.2 + w.2)
def mul (z w : Plane) : Plane :=
  (z.1 * w.1 - z.2 * w.2, z.1 * w.2 + z.2 * w.1)

/-- The actual centered clock defect, not an independently introduced force. -/
def clockDefect (c : ℝ) (l m r : Plane) : Plane :=
  sub (add l r) (scale (2 * c) m)

/-- For one incident face, the storage change minus the signed inflow is
the phase contraction of the clock defect. Sum both incident bonds for an
axis, then the three axes for the common site's equation. -/
theorem oriented_phase_balance (k c : ℝ) (li lj mi mj ri rj : Plane) :
    rightMomentum k c mi mj ri rj -
      rightMomentum k c li lj mi mj -
      inflow k c mi mj ri rj =
    k * cross mj (sub (clockDefect c lj mj rj) (clockDefect c li mi ri)) := by
  unfold inflow rightMomentum leftMomentum cross clockDefect add sub scale
  dsimp
  ring

/-- The balance defect is the negative derivative of the actual
two-interval action when only this site's middle phase is varied. -/
theorem oriented_balance_is_negative_variation
    (k c : ℝ) (li lj mi mj ri rj : Plane) :
    HasDerivAt (fun t => intervalAction k c li lj mi (phaseVariation mj t) +
        intervalAction k c mi (phaseVariation mj t) ri rj)
      (-(k * cross mj (sub (clockDefect c lj mj rj) (clockDefect c li mi ri)))) 0 := by
  have h := HasDerivAt.add
    (right_momentum_is_variation k c li lj mi mj)
    (left_momentum_is_variation k c mi mj ri rj)
  convert h using 1
  unfold rightMomentum leftMomentum cross clockDefect add sub scale
  dsimp
  ring

/-- For one complex product, the exact finite-layer defect splits into
the wave defect, the live companion defect, and the product of their
increments. No small-velocity or frozen-companion assumption is used.
Apply this to each product difference in chi wedge Psi. -/
theorem product_clock_defect
    (c : ℝ) (xl x xr yl y yr : Plane) :
    clockDefect c (mul xl yl) (mul x y) (mul xr yr) =
      add (mul x (clockDefect c yl y yr))
        (add (mul (clockDefect 1 xl x xr) y)
          (add (mul (sub xl x) (sub yl y))
            (mul (sub xr x) (sub yr y)))) := by
  unfold clockDefect mul add sub scale
  apply Prod.ext <;> dsimp <;> ring

/-- The corresponding wedge component: two complex products with their
native orientation. This is the exact shared-field curvature split
that contributes to both native phase charges. -/
theorem wedge_component_clock_defect (c : ℝ)
    (xl x xr yl y yr ul u ur vl v vr : Plane) :
    clockDefect c (sub (mul xl yl) (mul ul vl))
      (sub (mul x y) (mul u v)) (sub (mul xr yr) (mul ur vr)) =
    add (sub (mul x (clockDefect c yl y yr)) (mul u (clockDefect c vl v vr)))
      (add (sub (mul (clockDefect 1 xl x xr) y) (mul (clockDefect 1 ul u ur) v))
        (add (sub (mul (sub xl x) (sub yl y)) (mul (sub ul u) (sub vl v)))
          (sub (mul (sub xr x) (sub yr y)) (mul (sub ur u) (sub vr v))))) := by
  unfold clockDefect mul add sub scale
  apply Prod.ext <;> dsimp <;> ring

/-- If the companion is the same at the three native layers, the curvature
term vanishes: its wedge defect is entirely due to the wave clock. -/
theorem held_companion_clock_defect (c : ℝ) (x yl y yr : Plane) :
    clockDefect c (mul x yl) (mul x y) (mul x yr) =
      mul x (clockDefect c yl y yr) := by
  unfold clockDefect mul add sub scale
  apply Prod.ext <;> dsimp <;> ring


/-- Native infinitesimal phase rotation and complex conjugation. -/
def phaseTangent (z : Plane) : Plane := (-z.2, z.1)
def conjugate (z : Plane) : Plane := (z.1, -z.2)

/-- Both fields generate the same phase tangent on their bilinear product.
A wedge component is a difference of two such products. -/
theorem product_phase_generators (x y : Plane) :
    mul (phaseTangent x) y = phaseTangent (mul x y) ∧
    mul x (phaseTangent y) = phaseTangent (mul x y) := by
  unfold mul phaseTangent
  constructor <;> apply Prod.ext <;> dsimp <;> ring

/-- In the Hermitian parallel overlap, the chi generator has the opposite
sign to the Psi generator. Thus parallel contributions can change the
difference charge, not its sum. -/
theorem conjugate_product_phase_generators (x y : Plane) :
    mul (conjugate (phaseTangent x)) y =
      scale (-1) (phaseTangent (mul (conjugate x) y)) ∧
    mul (conjugate x) (phaseTangent y) =
      phaseTangent (mul (conjugate x) y) := by
  unfold mul conjugate phaseTangent scale
  constructor <;> apply Prod.ext <;> dsimp <;> ring

/-- Exact two-charge split once the identical wedge and opposite parallel
phase tangents are derived. A wedge-only response cannot alter the
difference of the two field charges. -/
theorem two_charge_split (p chi w s : ℝ) :
    (p+w+s) - (chi+w-s) = p-chi+2*s ∧
    (p+w+s) + (chi+w-s) = p+chi+2*w := by
  constructor <;> ring


/-- Exact change in signed face current between two consecutive native
intervals, split into the centered clock term and its clock defects. This
relates current decay to the same three-layer field motion. -/
theorem inflow_change_clock_split (k c : ℝ)
    (li lj mi mj ri rj : Plane) :
    inflow k c mi mj ri rj - inflow k c li lj mi mj =
      k * ((1 - c^2) * (cross mj (sub ri li) + cross (sub rj lj) mi) -
        c/2 * (cross (sub rj lj) (clockDefect c li mi ri) +
          cross (clockDefect c lj mj rj) (sub ri li))) := by
  unfold inflow leftMomentum rightMomentum cross clockDefect add sub scale
  dsimp
  ring

end
end LeanDirac.V4.FacePhaseMomentum

#print axioms LeanDirac.V4.FacePhaseMomentum.right_momentum_is_variation
#print axioms LeanDirac.V4.FacePhaseMomentum.inflow_antisymmetric
#print axioms LeanDirac.V4.FacePhaseMomentum.antiphase_pair_exact_transfer
#print axioms LeanDirac.V4.FacePhaseMomentum.constant_antiphase_difference_clock_del_zero

#print axioms LeanDirac.V4.FacePhaseMomentum.oriented_phase_balance
#print axioms LeanDirac.V4.FacePhaseMomentum.wedge_component_clock_defect
#print axioms LeanDirac.V4.FacePhaseMomentum.held_companion_clock_defect

#print axioms LeanDirac.V4.FacePhaseMomentum.product_phase_generators
#print axioms LeanDirac.V4.FacePhaseMomentum.conjugate_product_phase_generators
#print axioms LeanDirac.V4.FacePhaseMomentum.two_charge_split

#print axioms LeanDirac.V4.FacePhaseMomentum.left_momentum_is_variation
#print axioms LeanDirac.V4.FacePhaseMomentum.oriented_balance_is_negative_variation

#print axioms LeanDirac.V4.FacePhaseMomentum.inflow_change_clock_split
