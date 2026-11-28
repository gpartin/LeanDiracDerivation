/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import Mathlib.Analysis.Distribution.SchwartzSpace.Deriv
public import Mathlib.Analysis.Distribution.AEEqOfIntegralContDiff
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Tactic

@[expose] public section

namespace LeanDirac

/-! The two native complex-C3 fields and their ordinary positive kinetic law. -/

noncomputable section
open BigOperators Complex

abbrev C3 := Fin 3 → ℂ

def chi0 : ℝ := 19

structure State (Site : Type) where
  psi : Site → C3
  chi : Site → C3

structure Velocity (Site : Type) where
  psiDot : Site → C3
  chiDot : Site → C3

structure Bond (Site : Type) where
  source : Site
  target : Site
  weight : ℝ

def density (v : C3) : ℝ :=
  ∑ component, Complex.normSq (v component)

/-!
The oriented plane order is `(01, 02, 12)`.  The Euclidean cubical Hodge map
sends vector coordinates `(x, y, z)` to bivector coordinates `(z, -y, x)`.
The V3.6 action uses the one Euclidean Clifford metric on these two grades,
so corresponding unit vector and unit bivector blades carry equal kinetic
weight. This fixes the relative chi kinetic normalization to one. The two
native fields remain distinct; the grade map is a representation used for the
shared metric, not a field identification.
-/

def cubicalVectorToBivector (v : C3) : C3
  | 0 => v 2
  | 1 => -v 1
  | _ => v 0

def cubicalBivectorToVector (b : C3) : C3
  | 0 => b 2
  | 1 => -b 1
  | _ => b 0

theorem cube_bivector_to_vector_left_inverse (v : C3) :
    cubicalBivectorToVector (cubicalVectorToBivector v) = v := by
  funext component
  fin_cases component <;>
    simp [cubicalVectorToBivector, cubicalBivectorToVector]

theorem cube_vector_to_bivector_left_inverse (b : C3) :
    cubicalVectorToBivector (cubicalBivectorToVector b) = b := by
  funext component
  fin_cases component <;>
    simp [cubicalVectorToBivector, cubicalBivectorToVector]

theorem cube_vector_bivector_density_isometry (v : C3) :
    density (cubicalVectorToBivector v) = density v := by
  simp [density, cubicalVectorToBivector, Fin.sum_univ_three]
  ring

theorem cube_bivector_vector_density_isometry (b : C3) :
    density (cubicalBivectorToVector b) = density b := by
  simp [density, cubicalBivectorToVector, Fin.sum_univ_three]
  ring

structure CubicalVectorBivectorFiber where
  vector : C3
  bivector : C3

def cubicalCliffordKineticDensity
    (fiber : CubicalVectorBivectorFiber) : ℝ :=
  density fiber.vector + density fiber.bivector

/-- A fixed unit vector used only to evaluate the relative grade weight of the
standard Euclidean cubical Clifford metric. -/
def cubicalReferenceUnitVector : C3
  | 0 => 1
  | _ => 0

/-- V3.6 live action value computed as the bivector/vector norm ratio for
corresponding unit blades in the standard Euclidean cubical Clifford metric. -/
def chiKineticNormalization : ℝ :=
  density (cubicalVectorToBivector cubicalReferenceUnitVector) /
    density cubicalReferenceUnitVector

/-- `kappa` remains the reciprocal parameterization `chi0 / B`. -/
def kappa : ℝ := chi0 / chiKineticNormalization

theorem chi_kinetic_normalization_eq_one :
    chiKineticNormalization = 1 := by
  rw [chiKineticNormalization,
    cube_vector_bivector_density_isometry cubicalReferenceUnitVector]
  norm_num [density, cubicalReferenceUnitVector, Fin.sum_univ_three,
    Complex.normSq]

theorem kappa_eq_nineteen : kappa = 19 := by
  rw [kappa, chi_kinetic_normalization_eq_one]
  norm_num [chi0]

def inner (u v : C3) : ℂ :=
  ∑ component, star (u component) * v component

def difference (u v : C3) : C3 :=
  fun component => v component - u component

def kinetic [Fintype Site] (B : ℝ) (velocity : Velocity Site) : ℝ :=
  (∑ site, density (velocity.psiDot site)) / 2 +
    B * (∑ site, density (velocity.chiDot site)) / 2

def v36ResearchKinetic [Fintype Site] (velocity : Velocity Site) : ℝ :=
  kinetic chiKineticNormalization velocity

theorem v36_research_kinetic_has_equal_grade_weights [Fintype Site]
    (velocity : Velocity Site) :
    v36ResearchKinetic velocity =
      (∑ site, density (velocity.psiDot site)) / 2 +
        (∑ site, density (velocity.chiDot site)) / 2 := by
  rw [v36ResearchKinetic, kinetic, chi_kinetic_normalization_eq_one]
  ring

theorem density_nonnegative (v : C3) : 0 ≤ density v := by
  unfold density
  exact Finset.sum_nonneg (fun component _ =>
    Complex.normSq_nonneg (v component))

theorem kinetic_nonnegative [Fintype Site]
    (B : ℝ) (velocity : Velocity Site) (hB : 0 ≤ B) :
    0 ≤ kinetic B velocity := by
  unfold kinetic
  have hp : 0 ≤ ∑ site, density (velocity.psiDot site) :=
    Finset.sum_nonneg (fun site _ =>
      density_nonnegative (velocity.psiDot site))
  have hx : 0 ≤ ∑ site, density (velocity.chiDot site) :=
    Finset.sum_nonneg (fun site _ =>
      density_nonnegative (velocity.chiDot site))
  positivity

theorem state_has_exactly_two_field_families (state : State Site) :
    state = ⟨state.psi, state.chi⟩ := by
  cases state
  rfl

end
end LeanDirac

