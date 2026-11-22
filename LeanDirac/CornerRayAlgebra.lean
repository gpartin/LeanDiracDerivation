/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.Dirac004DiscreteAction

@[expose] public section

namespace LeanDirac

/-! Linear algebra of the corner rays for the native two-field action. -/

noncomputable section
open BigOperators Complex

abbrev Cell8 := Fin 8 → ℂ

def c3ScaleComplex (a : ℂ) (u : C3) : C3 :=
  fun component => a * u component

def c3Inner (u v : C3) : ℂ :=
  ∑ component, star (u component) * v component

def ComplexNormalTo (u eta : C3) : Prop := c3Inner u eta = 0

@[simp] theorem wedge_companion_tetra (corner : CubeCorner) :
    wedge (companion corner) (tetraCorner corner) = c3Neg unitArea := by
  have h := tetra_wedge_companion_constant corner
  funext component
  have hc := congrFun h component
  fin_cases component <;> simp [wedge, c3Neg] at hc ⊢ <;>
    rw [← hc] <;> ring

theorem c3_inner_unitArea_wedge_planar_companion
    (u eta : C3) (hu2 : u 2 = 0) :
    c3Inner unitArea
      (wedge (fun
        | 0 => -star (u 1)
        | 1 => star (u 0)
        | _ => 0) eta) = -c3Inner u eta := by
  apply Complex.ext <;>
    simp [c3Inner, Fin.sum_univ_succ, unitArea, wedge, hu2] <;> ring

theorem c3_inner_unitArea_wedge_planar
    (u eta : C3) (hu2 : u 2 = 0) :
    c3Inner unitArea (wedge eta u) =
      -c3Inner (fun
        | 0 => -star (u 1)
        | 1 => star (u 0)
        | _ => 0) eta := by
  apply Complex.ext <;>
    simp [c3Inner, Fin.sum_univ_succ, unitArea, wedge, hu2] <;> ring

theorem c3_inner_unitArea_wedge_companion
    (corner : CubeCorner) (eta : C3) :
    c3Inner unitArea (wedge (companion corner) eta) =
      -c3Inner (tetraCorner corner) eta := by
  apply c3_inner_unitArea_wedge_planar_companion
  fin_cases corner <;>
    simp [tetraCorner, tetraRay0, tetraRay1, tetraRay2, tetraRay3,
      c3Of, c3Neg]

theorem c3_inner_unitArea_wedge_tetra
    (corner : CubeCorner) (eta : C3) :
    c3Inner unitArea (wedge eta (tetraCorner corner)) =
      -c3Inner (companion corner) eta := by
  apply c3_inner_unitArea_wedge_planar
  fin_cases corner <;>
    simp [tetraCorner, tetraRay0, tetraRay1, tetraRay2, tetraRay3,
      c3Of, c3Neg]

theorem c3_real_pair_eq_inner_re (u v : C3) :
    c3RealPair u v = (c3Inner u v).re := by
  simp [c3RealPair, c3Inner]

theorem c3_inner_scale_left (a : ℂ) (u v : C3) :
    c3Inner (c3ScaleComplex a u) v = star a * c3Inner u v := by
  unfold c3Inner c3ScaleComplex
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro component _
  simp
  ring

theorem wedge_scale_left (a : ℂ) (u v : C3) :
    wedge (c3ScaleComplex a u) v = c3ScaleComplex a (wedge u v) := by
  funext component
  fin_cases component <;> simp [wedge, c3ScaleComplex] <;> ring

theorem wedge_scale_right (a : ℂ) (u v : C3) :
    wedge u (c3ScaleComplex a v) = c3ScaleComplex a (wedge u v) := by
  funext component
  fin_cases component <;> simp [wedge, c3ScaleComplex] <;> ring

theorem c3_inner_sub_right (u v z : C3) :
    c3Inner u (difference v z) = c3Inner u z - c3Inner u v := by
  unfold c3Inner difference
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro component _
  ring

theorem c3_inner_scale_left_general (a : ℂ) (u v : C3) :
    c3Inner (c3ScaleComplex a u) v = star a * c3Inner u v :=
  c3_inner_scale_left a u v

def c3Sub (u v : C3) : C3 := fun component => u component - v component

def rayResidual (u z : C3) : C3 :=
  c3Sub z (c3ScaleComplex (c3Inner u z) u)

theorem c3_inner_self (u : C3) :
    c3Inner u u = (density u : ℂ) := by
  apply Complex.ext
  · simp [c3Inner, density, Fin.sum_univ_succ, Complex.normSq]
  · simp [c3Inner, density, Fin.sum_univ_succ, Complex.normSq]
    ring

theorem c3_inner_sub_right_general (u v z : C3) :
    c3Inner u (c3Sub v z) = c3Inner u v - c3Inner u z := by
  unfold c3Inner c3Sub
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro component _
  ring

theorem c3_inner_scale_right (a : ℂ) (u v : C3) :
    c3Inner u (c3ScaleComplex a v) = a * c3Inner u v := by
  unfold c3Inner c3ScaleComplex
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro component _
  ring

theorem ray_residual_normal (u z : C3) (hnorm : density u = 1) :
    ComplexNormalTo u (rayResidual u z) := by
  unfold ComplexNormalTo rayResidual
  rw [c3_inner_sub_right_general, c3_inner_scale_right,
    c3_inner_self, hnorm]
  norm_num

theorem c3_real_pair_self (u : C3) : c3RealPair u u = density u := by
  rw [c3_real_pair_eq_inner_re, c3_inner_self]
  simp

theorem c3_real_pair_scale_left_zero (a : ℂ) (u v : C3)
    (h : c3Inner u v = 0) :
    c3RealPair (c3ScaleComplex a u) v = 0 := by
  rw [c3_real_pair_eq_inner_re, c3_inner_scale_left, h]
  simp

theorem ray_plus_residual (u z : C3) :
    c3Add (c3ScaleComplex (c3Inner u z) u) (rayResidual u z) = z := by
  funext component
  simp [c3Add, c3ScaleComplex, rayResidual, c3Sub]

theorem c3_real_pair_with_residual (u z : C3)
    (hnorm : density u = 1) :
    c3RealPair z (rayResidual u z) = density (rayResidual u z) := by
  calc
    c3RealPair z (rayResidual u z) =
        c3RealPair
          (c3Add (c3ScaleComplex (c3Inner u z) u) (rayResidual u z))
          (rayResidual u z) := by rw [ray_plus_residual]
    _ = c3RealPair (c3ScaleComplex (c3Inner u z) u) (rayResidual u z) +
        c3RealPair (rayResidual u z) (rayResidual u z) := by
          change c3RealPair
            (fun component =>
              c3ScaleComplex (c3Inner u z) u component +
                rayResidual u z component)
            (rayResidual u z) = _
          rw [c3_real_pair_add_left]
    _ = density (rayResidual u z) := by
      rw [c3_real_pair_scale_left_zero _ _ _
          (ray_residual_normal u z hnorm),
        c3_real_pair_self]
      ring

theorem density_zero_implies_c3_zero (u : C3) (h : density u = 0) :
    u = c3Zero := by
  have hsum := h
  simp [density, Fin.sum_univ_succ, Complex.normSq] at hsum
  funext component
  fin_cases component <;> apply Complex.ext <;> simp [c3Zero] <;>
    nlinarith [sq_nonneg (u 0).re, sq_nonneg (u 0).im,
      sq_nonneg (u 1).re, sq_nonneg (u 1).im,
      sq_nonneg (u 2).re, sq_nonneg (u 2).im]

theorem companion_density_one (corner : CubeCorner) :
    density (companion corner) = 1 := by
  fin_cases corner <;>
    simp [density, Fin.sum_univ_succ, companion, tetraCorner,
      tetraRay0, tetraRay1, tetraRay2, tetraRay3, c3Of, c3Neg,
      tetraA, tetraB, omega, omegaBar, Complex.normSq] <;>
    field_simp [sqrt_three_ne_zero] <;> ring_nf <;>
    nlinarith [sqrt_three_sq, sqrt_two_sq, sqrt_three_pow_four,
      sqrt_three_pow_five, sqrt_three_pow_six, sqrt_three_pow_eight,
      sqrt_two_pow_four]

theorem density_aligned_complex_scale (z : ℂ) (u : C3)
    (hu : density u = 1) :
    density (c3ScaleComplex z u) = Complex.normSq z := by
  unfold density c3ScaleComplex
  simp_rw [Complex.normSq_mul]
  change (∑ i, Complex.normSq (u i)) = 1 at hu
  rw [← Finset.mul_sum, hu, mul_one]

end
end LeanDirac

