/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.NativeAction

@[expose] public section

namespace LeanDirac

/-!
# C19 Fourier symbol from the native geometry

This module contains only Fourier analysis of the canonical C19 bonds.  It
imports the native action and geometry, but no prepared rays, background,
Dirac stage, spinor, or Clifford matrix.
-/

noncomputable section
open BigOperators Complex

set_option maxHeartbeats 1000000

abbrev LatticeMomentum := Fin 3 → ℝ
abbrev InfiniteCubicSite := Fin 3 → ℤ

def infiniteC19Neighbor (site : InfiniteCubicSite)
    (direction : C19Neighbor) : InfiniteCubicSite :=
  fun axis => site axis + c19Coord axis direction

def infiniteC19Bond (site : InfiniteCubicSite)
    (direction : C19Neighbor) : Bond InfiniteCubicSite where
  source := site
  target := infiniteC19Neighbor site direction
  weight := c19Weight direction

def latticePhase (momentum : LatticeMomentum)
    (site : InfiniteCubicSite) : ℝ :=
  ∑ axis, momentum axis * (site axis : ℝ)

def c19PhaseAngle (momentum : LatticeMomentum)
    (direction : C19Neighbor) : ℝ :=
  ∑ axis, momentum axis * (c19Coord axis direction : ℝ)

def c19Character (momentum : LatticeMomentum)
    (direction : C19Neighbor) : ℂ :=
  Complex.exp (I * (c19PhaseAngle momentum direction : ℂ))

def cubicLatticeCharacter (momentum : LatticeMomentum)
    (site : InfiniteCubicSite) : ℂ :=
  Complex.exp (I * (latticePhase momentum site : ℂ))

theorem lattice_phase_neighbor (momentum : LatticeMomentum)
    (site : InfiniteCubicSite) (direction : C19Neighbor) :
    latticePhase momentum (infiniteC19Neighbor site direction) =
      latticePhase momentum site + c19PhaseAngle momentum direction := by
  unfold latticePhase infiniteC19Neighbor c19PhaseAngle
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro axis _
  push_cast
  ring

theorem cubic_character_neighbor (momentum : LatticeMomentum)
    (site : InfiniteCubicSite) (direction : C19Neighbor) :
    cubicLatticeCharacter momentum (infiniteC19Neighbor site direction) =
      cubicLatticeCharacter momentum site * c19Character momentum direction := by
  unfold cubicLatticeCharacter c19Character
  rw [lattice_phase_neighbor]
  push_cast
  rw [mul_add, Complex.exp_add]

theorem cubic_character_normSq_one (momentum : LatticeMomentum)
    (site : InfiniteCubicSite) :
    Complex.normSq (cubicLatticeCharacter momentum site) = 1 := by
  rw [Complex.normSq_eq_norm_sq]
  simp [cubicLatticeCharacter, Complex.norm_exp_I_mul_ofReal]

def c19IntegerWeight (direction : C19Neighbor) : ℤ :=
  if direction.val < 6 then 2 else 1

theorem c19_weight_is_integer_weight (direction : C19Neighbor) :
    c19Weight direction = (c19IntegerWeight direction : ℝ) := by
  unfold c19Weight c19IntegerWeight
  split <;> norm_num

def c19FirstMoment (axis : Fin 3) : ℤ :=
  ∑ direction, c19IntegerWeight direction * c19Coord axis direction

def c19SecondMoment (axis other : Fin 3) : ℤ :=
  ∑ direction, c19IntegerWeight direction *
    c19Coord axis direction * c19Coord other direction

theorem c19_first_moment_zero (axis : Fin 3) :
    c19FirstMoment axis = 0 := by
  fin_cases axis <;> decide

theorem c19_second_moment_isotropic (axis other : Fin 3) :
    c19SecondMoment axis other = if axis = other then 12 else 0 := by
  fin_cases axis <;> fin_cases other <;> decide

def c19TransportSymbol (momentum : LatticeMomentum) : ℝ :=
  ∑ direction,
    c19Weight direction *
      Complex.normSq (1 - c19Character momentum direction)

theorem c19_transport_symbol_nonnegative (momentum : LatticeMomentum) :
    0 ≤ c19TransportSymbol momentum := by
  unfold c19TransportSymbol
  exact Finset.sum_nonneg (fun direction _ =>
    mul_nonneg (le_of_lt (c19_weight_positive direction))
      (Complex.normSq_nonneg _))

theorem c19_character_zero (direction : C19Neighbor) :
    c19Character (fun _ => 0) direction = 1 := by
  simp [c19Character, c19PhaseAngle]

theorem c19_transport_symbol_zero :
    c19TransportSymbol (fun _ => 0) = 0 := by
  simp [c19TransportSymbol, c19_character_zero]

def c19ContinuumSpeed : ℝ := Real.sqrt 12

theorem c19_continuum_speed_square : c19ContinuumSpeed^2 = 12 := by
  norm_num [c19ContinuumSpeed]

theorem normSq_one_sub_exp_I (angle : ℝ) :
    Complex.normSq
        (1 - Complex.exp (Complex.I * (angle : ℂ))) =
      2 * (1 - Real.cos angle) := by
  rw [mul_comm, Complex.exp_ofReal_mul_I]
  rw [Complex.normSq_apply]
  simp [Complex.cos_ofReal_re, Complex.sin_ofReal_re]
  nlinarith [Real.sin_sq_add_cos_sq angle]

private def c19ClosedPhase
    (momentum : LatticeMomentum) (direction : C19Neighbor) : ℝ :=
  match direction.val with
  | 0 => momentum 0
  | 1 => -momentum 0
  | 2 => momentum 1
  | 3 => -momentum 1
  | 4 => momentum 2
  | 5 => -momentum 2
  | 6 => -momentum 0 + -momentum 1
  | 7 => -momentum 0 + momentum 1
  | 8 => momentum 0 + -momentum 1
  | 9 => momentum 0 + momentum 1
  | 10 => -momentum 0 + -momentum 2
  | 11 => -momentum 0 + momentum 2
  | 12 => momentum 0 + -momentum 2
  | 13 => momentum 0 + momentum 2
  | 14 => -momentum 1 + -momentum 2
  | 15 => -momentum 1 + momentum 2
  | 16 => momentum 1 + -momentum 2
  | _ => momentum 1 + momentum 2

private theorem c19_phase_angle_closed
    (momentum : LatticeMomentum) (direction : C19Neighbor) :
    c19PhaseAngle momentum direction = c19ClosedPhase momentum direction := by
  fin_cases direction <;>
    simp [c19PhaseAngle, c19ClosedPhase, c19Coord, Fin.sum_univ_succ]

def c19QuadraticSymbol (momentum : LatticeMomentum) : ℝ :=
  12 * (momentum 0 ^ 2 + momentum 1 ^ 2 + momentum 2 ^ 2)

theorem c19_quadratic_symbol_is_phase_square_sum
    (momentum : LatticeMomentum) :
    c19QuadraticSymbol momentum =
      ∑ direction,
        c19Weight direction * c19PhaseAngle momentum direction ^ 2 := by
  unfold c19QuadraticSymbol c19PhaseAngle
  simp only [Fin.sum_univ_succ]
  simp [c19Weight, c19Coord]
  ring

def c19SpatialRemainder (momentum : LatticeMomentum) : ℝ :=
  c19TransportSymbol momentum - c19QuadraticSymbol momentum

theorem c19_spatial_remainder_is_bond_error_sum
    (momentum : LatticeMomentum) :
    c19SpatialRemainder momentum =
      ∑ direction, c19Weight direction *
        (2 * (1 - Real.cos (c19PhaseAngle momentum direction)) -
          c19PhaseAngle momentum direction ^ 2) := by
  rw [c19SpatialRemainder, c19_quadratic_symbol_is_phase_square_sum]
  unfold c19TransportSymbol
  simp_rw [c19Character, normSq_one_sub_exp_I]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro direction _
  ring

private theorem abs_le_two_mul_of_abs_le
    (value radius : ℝ) (hradius : 0 ≤ radius)
    (hvalue : |value| ≤ radius) :
    |value| ≤ 2 * radius := by
  linarith

private theorem abs_add_le_two_mul
    (first second radius : ℝ)
    (hfirst : |first| ≤ radius) (hsecond : |second| ≤ radius) :
    |first + second| ≤ 2 * radius := by
  calc
    |first + second| ≤ |first| + |second| := abs_add_le first second
    _ ≤ radius + radius := add_le_add hfirst hsecond
    _ = 2 * radius := by ring

theorem c19_phase_angle_abs_le_two_mul
    (momentum : LatticeMomentum) (radius : ℝ) (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (direction : C19Neighbor) :
    |c19PhaseAngle momentum direction| ≤ 2 * radius := by
  have h0 := hmomentum (0 : Fin 3)
  have h1 := hmomentum (1 : Fin 3)
  have h2 := hmomentum (2 : Fin 3)
  have hn0 : |-momentum 0| ≤ radius := by simpa using h0
  have hn1 : |-momentum 1| ≤ radius := by simpa using h1
  have hn2 : |-momentum 2| ≤ radius := by simpa using h2
  rw [c19_phase_angle_closed]
  fin_cases direction
  · exact abs_le_two_mul_of_abs_le _ _ hradius h0
  · exact abs_le_two_mul_of_abs_le _ _ hradius hn0
  · exact abs_le_two_mul_of_abs_le _ _ hradius h1
  · exact abs_le_two_mul_of_abs_le _ _ hradius hn1
  · exact abs_le_two_mul_of_abs_le _ _ hradius h2
  · exact abs_le_two_mul_of_abs_le _ _ hradius hn2
  · exact abs_add_le_two_mul _ _ _ hn0 hn1
  · exact abs_add_le_two_mul _ _ _ hn0 h1
  · exact abs_add_le_two_mul _ _ _ h0 hn1
  · exact abs_add_le_two_mul _ _ _ h0 h1
  · exact abs_add_le_two_mul _ _ _ hn0 hn2
  · exact abs_add_le_two_mul _ _ _ hn0 h2
  · exact abs_add_le_two_mul _ _ _ h0 hn2
  · exact abs_add_le_two_mul _ _ _ h0 h2
  · exact abs_add_le_two_mul _ _ _ hn1 hn2
  · exact abs_add_le_two_mul _ _ _ hn1 h2
  · exact abs_add_le_two_mul _ _ _ h1 hn2
  · exact abs_add_le_two_mul _ _ _ h1 h2

theorem cosine_phase_error_bound
    (angle radius : ℝ) (_hradius : 0 ≤ radius)
    (hangle : |angle| ≤ 2 * radius) (hsmall : 2 * radius ≤ 1) :
    |2 * (1 - Real.cos angle) - angle^2| ≤ (5/3 : ℝ) * radius^4 := by
  have hunit : |angle| ≤ 1 := hangle.trans hsmall
  have hcos := Real.cos_bound hunit
  have hpow : |angle|^4 ≤ (2 * radius)^4 :=
    pow_le_pow_left₀ (abs_nonneg angle) hangle 4
  calc
    |2 * (1 - Real.cos angle) - angle^2| =
        2 * |Real.cos angle - (1 - angle^2 / 2)| := by
          rw [show 2 * (1 - Real.cos angle) - angle^2 =
            -2 * (Real.cos angle - (1 - angle^2 / 2)) by ring]
          simp
    _ ≤ 2 * (|angle|^4 * (5/96 : ℝ)) := by nlinarith
    _ ≤ 2 * ((2 * radius)^4 * (5/96 : ℝ)) := by nlinarith
    _ = (5/3 : ℝ) * radius^4 := by ring

theorem c19_spatial_remainder_bound
    (momentum : LatticeMomentum) (radius : ℝ) (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (hsmall : 2 * radius ≤ 1) :
    |c19SpatialRemainder momentum| ≤ 40 * radius^4 := by
  rw [c19_spatial_remainder_is_bond_error_sum]
  calc
    |∑ direction, c19Weight direction *
        (2 * (1 - Real.cos (c19PhaseAngle momentum direction)) -
          c19PhaseAngle momentum direction ^ 2)| ≤
      ∑ direction, |c19Weight direction *
        (2 * (1 - Real.cos (c19PhaseAngle momentum direction)) -
          c19PhaseAngle momentum direction ^ 2)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ direction, c19Weight direction * ((5/3 : ℝ) * radius^4) := by
      apply Finset.sum_le_sum
      intro direction _
      rw [abs_mul, abs_of_pos (c19_weight_positive direction)]
      exact mul_le_mul_of_nonneg_left
        (cosine_phase_error_bound
          (c19PhaseAngle momentum direction) radius hradius
          (c19_phase_angle_abs_le_two_mul momentum radius hradius
            hmomentum direction) hsmall)
        (le_of_lt (c19_weight_positive direction))
    _ = 40 * radius^4 := by
      have hweights : ∑ direction, c19Weight direction = 24 := by
        norm_num [c19Weight, Fin.sum_univ_succ]
      rw [← Finset.sum_mul, hweights]
      ring

theorem c19_symbol_exact_quadratic_plus_remainder
    (momentum : LatticeMomentum) :
    c19TransportSymbol momentum =
      c19QuadraticSymbol momentum + c19SpatialRemainder momentum := by
  simp [c19SpatialRemainder]

theorem c19_spatial_remainder_zero_at_origin :
    c19SpatialRemainder (fun _ => 0) = 0 := by
  simp [c19SpatialRemainder, c19QuadraticSymbol,
    c19_transport_symbol_zero]

end
end LeanDirac

#print axioms LeanDirac.c19_second_moment_isotropic
#print axioms LeanDirac.c19_spatial_remainder_bound
