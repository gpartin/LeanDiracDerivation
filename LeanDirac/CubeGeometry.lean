import LeanDirac.Foundation

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-! Explicit eight-corner tetrahedral geometry used by the prepared state. -/

noncomputable section
open BigOperators Complex

abbrev CubeCorner := Fin 8
abbrev C19Neighbor := Fin 18

def c19Coord (axis : Fin 3) (neighbor : C19Neighbor) : ℤ :=
  match neighbor.val, axis.val with
  | 0, 0 => 1 | 1, 0 => -1 | 2, 1 => 1 | 3, 1 => -1
  | 4, 2 => 1 | 5, 2 => -1
  | 6, 0 => -1 | 6, 1 => -1
  | 7, 0 => -1 | 7, 1 => 1
  | 8, 0 => 1 | 8, 1 => -1
  | 9, 0 => 1 | 9, 1 => 1
  | 10, 0 => -1 | 10, 2 => -1
  | 11, 0 => -1 | 11, 2 => 1
  | 12, 0 => 1 | 12, 2 => -1
  | 13, 0 => 1 | 13, 2 => 1
  | 14, 1 => -1 | 14, 2 => -1
  | 15, 1 => -1 | 15, 2 => 1
  | 16, 1 => 1 | 16, 2 => -1
  | 17, 1 => 1 | 17, 2 => 1
  | _, _ => 0

def c19Weight (neighbor : C19Neighbor) : ℝ :=
  if neighbor.val < 6 then 2 else 1

theorem c19_weight_positive (neighbor : C19Neighbor) :
    0 < c19Weight neighbor := by
  unfold c19Weight
  split <;> norm_num

def tetraA : ℝ := 1 / Real.sqrt 3
def tetraB : ℝ := Real.sqrt 2 / Real.sqrt 3
def omega : ℂ := (-1/2 : ℝ) + (Real.sqrt 3 / 2 : ℝ) * I
def omegaBar : ℂ := (-1/2 : ℝ) - (Real.sqrt 3 / 2 : ℝ) * I

def c3Of (x y z : ℂ) : C3
  | 0 => x | 1 => y | _ => z

def tetraRay0 : C3 := c3Of 1 0 0
def tetraRay1 : C3 := c3Of tetraA tetraB 0
def tetraRay2 : C3 := c3Of tetraA (tetraB * omega) 0
def tetraRay3 : C3 := c3Of tetraA (tetraB * omegaBar) 0
def c3Neg (u : C3) : C3 := fun component => -u component

def tetraCorner (corner : CubeCorner) : C3 :=
  match corner.val with
  | 0 => tetraRay0 | 1 => tetraRay1
  | 2 => tetraRay2 | 3 => tetraRay3
  | 4 => c3Neg tetraRay3 | 5 => c3Neg tetraRay2
  | 6 => c3Neg tetraRay1 | _ => c3Neg tetraRay0

def overlap (u v : C3) : ℂ :=
  ⟨(∑ component : Fin 3,
      ((u component).re * (v component).re +
        (u component).im * (v component).im)),
    (∑ component : Fin 3,
      ((u component).re * (v component).im -
        (u component).im * (v component).re))⟩

def faceProduct (a b c d : CubeCorner) : ℂ :=
  overlap (tetraCorner a) (tetraCorner b) *
    overlap (tetraCorner b) (tetraCorner c) *
    overlap (tetraCorner c) (tetraCorner d) *
    overlap (tetraCorner d) (tetraCorner a)

theorem sqrt_three_ne_zero : Real.sqrt 3 ≠ 0 := by positivity
theorem sqrt_three_sq : (Real.sqrt 3)^2 = 3 := by norm_num
theorem sqrt_two_sq : (Real.sqrt 2)^2 = 2 := by norm_num
theorem sqrt_three_pow_four : (Real.sqrt 3)^4 = 9 := by
  calc
    (Real.sqrt 3)^4 = ((Real.sqrt 3)^2)^2 := by ring
    _ = 9 := by rw [sqrt_three_sq]; norm_num
theorem sqrt_three_pow_five : (Real.sqrt 3)^5 = 9 * Real.sqrt 3 := by
  calc
    (Real.sqrt 3)^5 = (Real.sqrt 3)^4 * Real.sqrt 3 := by ring
    _ = 9 * Real.sqrt 3 := by rw [sqrt_three_pow_four]
theorem sqrt_three_pow_six : (Real.sqrt 3)^6 = 27 := by
  calc
    (Real.sqrt 3)^6 = ((Real.sqrt 3)^2)^3 := by ring
    _ = 27 := by rw [sqrt_three_sq]; norm_num
theorem sqrt_three_pow_eight : (Real.sqrt 3)^8 = 81 := by
  calc
    (Real.sqrt 3)^8 = ((Real.sqrt 3)^4)^2 := by ring
    _ = 81 := by rw [sqrt_three_pow_four]; norm_num
theorem sqrt_two_pow_four : (Real.sqrt 2)^4 = 4 := by
  calc
    (Real.sqrt 2)^4 = ((Real.sqrt 2)^2)^2 := by ring
    _ = 4 := by rw [sqrt_two_sq]; norm_num

theorem tetra_corner_density_one (corner : CubeCorner) :
    density (tetraCorner corner) = 1 := by
  fin_cases corner <;>
    simp [density, Fin.sum_univ_succ, tetraCorner, tetraRay0,
      tetraRay1, tetraRay2, tetraRay3, c3Of, c3Neg, omega, omegaBar,
      tetraA, tetraB, Complex.normSq] <;>
    field_simp [sqrt_three_ne_zero] <;> ring_nf <;>
    nlinarith [sqrt_three_sq, sqrt_two_sq, sqrt_three_pow_four,
      sqrt_three_pow_five, sqrt_three_pow_six, sqrt_three_pow_eight,
      sqrt_two_pow_four]

theorem all_face_products_negative :
    faceProduct 0 2 3 1 = (-1/9 : ℂ) ∧
    faceProduct 4 6 7 5 = (-1/9 : ℂ) ∧
    faceProduct 0 4 5 1 = (-1/9 : ℂ) ∧
    faceProduct 2 6 7 3 = (-1/9 : ℂ) ∧
    faceProduct 0 4 6 2 = (-1/9 : ℂ) ∧
    faceProduct 1 5 7 3 = (-1/9 : ℂ) := by
  repeat' apply And.intro
  all_goals apply Complex.ext <;>
    simp [faceProduct, overlap, Fin.sum_univ_succ, tetraCorner,
      tetraRay0, tetraRay1, tetraRay2, tetraRay3, c3Of, c3Neg,
      omega, omegaBar, tetraA, tetraB] <;>
    field_simp [sqrt_three_ne_zero] <;> ring_nf <;>
    nlinarith [sqrt_three_sq, sqrt_two_sq, sqrt_three_pow_four,
      sqrt_three_pow_five, sqrt_three_pow_six, sqrt_three_pow_eight,
      sqrt_two_pow_four]

end
end LeanDirac

