import LeanDirac.V33FrameworkLock

/-!
# V3.3 Stage 2: complete orthonormal spinor basis
-/

namespace LeanDirac

noncomputable section

theorem v33_standard_basis_orthonormal (i j : Fin 4) :
    cubicalCell4Inner (standardCell4Basis i) (standardCell4Basis j) =
      if i = j then 1 else 0 := by
  fin_cases i <;> fin_cases j <;>
    simp [cubicalCell4Inner, standardCell4Basis]

theorem v33_standard_basis_resolves (z : CubicalCell4) :
    (fun component =>
      ∑ i : Fin 4, z i * standardCell4Basis i component) = z := by
  funext component
  fin_cases component <;>
    simp [standardCell4Basis]

structure V33SpinorBasisCertificate : Prop where
  orthonormal : ∀ i j : Fin 4,
    cubicalCell4Inner (standardCell4Basis i) (standardCell4Basis j) =
      if i = j then 1 else 0
  resolves : ∀ z : CubicalCell4,
    (fun component =>
      ∑ i : Fin 4, z i * standardCell4Basis i component) = z

theorem v33_spinor_basis : V33SpinorBasisCertificate where
  orthonormal := v33_standard_basis_orthonormal
  resolves := v33_standard_basis_resolves

#print axioms v33_spinor_basis

end
end LeanDirac
