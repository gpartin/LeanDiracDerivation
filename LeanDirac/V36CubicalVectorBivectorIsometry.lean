import LeanDirac.V35CompleteCovariantObservables

/-!
# V3.6 Stage 1: cubical vector--bivector kinetic geometry

The oriented plane order is `(01, 02, 12)`.  The Euclidean Hodge map sends
vector coordinates `(x, y, z)` to bivector coordinates `(z, -y, x)`.
-/

namespace LeanDirac

noncomputable section

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

theorem cube_corresponding_unit_blades_have_equal_weight
    (selected : Fin 3) :
    density (cubicalVectorToBivector (basis selected)) =
      density (basis selected) :=
  cube_vector_bivector_density_isometry (basis selected)

theorem cube_equal_grade_metric_recovers_live_B :
    chiKineticNormalization = 1 :=
  chi_kinetic_normalization_eq_one

theorem cube_every_unit_blade_has_live_relative_weight
    (selected : Fin 3) :
    chiKineticNormalization * density (basis selected) =
      density (cubicalVectorToBivector (basis selected)) := by
  rw [chi_kinetic_normalization_eq_one, one_mul]
  exact (cube_corresponding_unit_blades_have_equal_weight selected).symm

def cubeAntisymmetricProduct (chi psi : C3) : C3 :=
  cubicalBivectorToVector (wedge chi psi)

theorem cube_antisymmetric_product_components (chi psi : C3) :
    cubeAntisymmetricProduct chi psi = fun component =>
      match component with
      | 0 => chi 1 * psi 2 - chi 2 * psi 1
      | 1 => chi 2 * psi 0 - chi 0 * psi 2
      | 2 => chi 0 * psi 1 - chi 1 * psi 0 := by
  funext component
  fin_cases component <;>
    simp [cubeAntisymmetricProduct, cubicalBivectorToVector, wedge]

structure V36CubicalVectorBivectorIsometryCertificate : Prop where
  vectorRoundTrip : ∀ v,
    cubicalBivectorToVector (cubicalVectorToBivector v) = v
  bivectorRoundTrip : ∀ b,
    cubicalVectorToBivector (cubicalBivectorToVector b) = b
  vectorNorm : ∀ v, density (cubicalVectorToBivector v) = density v
  bivectorNorm : ∀ b, density (cubicalBivectorToVector b) = density b
  liveWeight : chiKineticNormalization = 1
  allUnitBlades : ∀ selected,
    chiKineticNormalization * density (basis selected) =
      density (cubicalVectorToBivector (basis selected))
  nativeInteraction : ∀ chi psi,
    cubeAntisymmetricProduct chi psi = fun component =>
      match component with
      | 0 => chi 1 * psi 2 - chi 2 * psi 1
      | 1 => chi 2 * psi 0 - chi 0 * psi 2
      | 2 => chi 0 * psi 1 - chi 1 * psi 0

theorem v36_cubical_vector_bivector_isometry :
    V36CubicalVectorBivectorIsometryCertificate where
  vectorRoundTrip := cube_bivector_to_vector_left_inverse
  bivectorRoundTrip := cube_vector_to_bivector_left_inverse
  vectorNorm := cube_vector_bivector_density_isometry
  bivectorNorm := cube_bivector_vector_density_isometry
  liveWeight := cube_equal_grade_metric_recovers_live_B
  allUnitBlades := cube_every_unit_blade_has_live_relative_weight
  nativeInteraction := cube_antisymmetric_product_components

#print axioms v36_cubical_vector_bivector_isometry

end
end LeanDirac

