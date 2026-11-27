import LeanDirac.V35GeneratedLorentzTransformations
import LeanDirac.V34NonforwardGordon

/-!
# V3.5 Stage 5: the complete covariant bilinear family

All operators are compositions of the cubical gamma operators already present
in V3.4.  In particular, gamma-five is derived here rather than supplied as a
new matrix.
-/

namespace LeanDirac

noncomputable section

def v35GammaFive (z : CubicalCell4) : CubicalCell4 :=
  fun component => Complex.I *
    v34Gamma 0 (v34Gamma 1 (v34Gamma 2 (v34Gamma 3 z))) component

def v35ScalarBilinear (left right : CubicalCell4) : ℂ :=
  v34DiracBilinear left (fun z => z) right

def v35PseudoscalarBilinear (left right : CubicalCell4) : ℂ :=
  v34DiracBilinear left v35GammaFive right

def v35VectorBilinear
    (mu : Fin 4) (left right : CubicalCell4) : ℂ :=
  v34DiracBilinear left (v34Gamma mu) right

def v35AxialBilinear
    (mu : Fin 4) (left right : CubicalCell4) : ℂ :=
  v34DiracBilinear left (fun z => v34Gamma mu (v35GammaFive z)) right

def v35TensorBilinear
    (mu nu : Fin 4) (left right : CubicalCell4) : ℂ :=
  v34DiracBilinear left (v34Sigma mu nu) right

theorem v35_gamma_five_square (z : CubicalCell4) :
    v35GammaFive (v35GammaFive z) = z := by
  funext component
  fin_cases component <;>
    simp [v35GammaFive, v34Gamma, cubicalGammaTime,
      cubicalGammaSpace, cubicalAlpha, cubicalStandardBeta,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ] <;>
    ring_nf <;> norm_num [Complex.I_sq]

theorem v35_gamma_five_anticommutes
    (mu : Fin 4) (z : CubicalCell4) :
    (fun component =>
      v35GammaFive (v34Gamma mu z) component +
        v34Gamma mu (v35GammaFive z) component) = 0 := by
  funext component
  fin_cases mu <;> fin_cases component <;>
    simp [v35GammaFive, v34Gamma, cubicalGammaTime,
      cubicalGammaSpace, cubicalAlpha, cubicalStandardBeta,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ]

theorem v35_tensor_bilinear_antisymmetric
    (mu nu : Fin 4) (left right : CubicalCell4) :
    v35TensorBilinear nu mu left right =
      -v35TensorBilinear mu nu left right := by
  simp [v35TensorBilinear, v34DiracBilinear, v34Sigma,
    Fin.sum_univ_four]
  ring

structure V35CompleteBilinearCertificate : Prop where
  gammaFiveSquare : ∀ z, v35GammaFive (v35GammaFive z) = z
  gammaFiveAnticommutes : ∀ mu z,
    (fun component =>
      v35GammaFive (v34Gamma mu z) component +
        v34Gamma mu (v35GammaFive z) component) = 0
  tensorAntisymmetric : ∀ mu nu left right,
    v35TensorBilinear nu mu left right =
      -v35TensorBilinear mu nu left right

theorem v35_complete_bilinears : V35CompleteBilinearCertificate where
  gammaFiveSquare := v35_gamma_five_square
  gammaFiveAnticommutes := v35_gamma_five_anticommutes
  tensorAntisymmetric := v35_tensor_bilinear_antisymmetric

#print axioms v35_complete_bilinears

end
end LeanDirac
