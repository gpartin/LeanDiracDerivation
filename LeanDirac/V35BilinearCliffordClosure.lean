import LeanDirac.V35CompleteBilinears

/-!
# V3.5 Stage 6: sixteen-channel Clifford closure

The five covariant classes contain 1 + 1 + 4 + 4 + 6 channels.  The six
tensor labels below are the unordered index pairs; antisymmetry from Stage 5
supplies every reversed pair and makes the diagonal vanish.
-/

namespace LeanDirac

noncomputable section

inductive V35BilinearChannel where
  | scalar
  | pseudoscalar
  | vector (mu : Fin 4)
  | axial (mu : Fin 4)
  | tensor (pair : Fin 6)
  deriving DecidableEq, Fintype

def v35TensorPair : Fin 6 → Fin 4 × Fin 4
  | 0 => (0, 1)
  | 1 => (0, 2)
  | 2 => (0, 3)
  | 3 => (1, 2)
  | 4 => (1, 3)
  | _ => (2, 3)

def v35BilinearOperator : V35BilinearChannel → CubicalCell4 → CubicalCell4
  | .scalar => fun z => z
  | .pseudoscalar => v35GammaFive
  | .vector mu => v34Gamma mu
  | .axial mu => fun z => v34Gamma mu (v35GammaFive z)
  | .tensor pair => v34Sigma (v35TensorPair pair).1 (v35TensorPair pair).2

def v35GammaLower (mu : Fin 4) (z : CubicalCell4) : CubicalCell4 :=
  if mu = 0 then v34Gamma mu z else fun component => -v34Gamma mu z component

def v35AxialUpper (mu : Fin 4) (z : CubicalCell4) : CubicalCell4 :=
  v34Gamma mu (v35GammaFive z)

def v35AxialLower (mu : Fin 4) (z : CubicalCell4) : CubicalCell4 :=
  v35GammaLower mu (v35GammaFive z)

def v35SigmaLower (mu nu : Fin 4) (z : CubicalCell4) : CubicalCell4 :=
  if (mu = 0) = (nu = 0) then v34Sigma mu nu z
  else fun component => -v34Sigma mu nu z component

/-- Inverse of the number of complex channels in the derived spinor. -/
def v35SpinorChannelNormalization : ℂ :=
  ((Fintype.card (Fin 4) : ℕ) : ℂ)⁻¹

/-- Inverse of the two orderings of each off-diagonal antisymmetric pair. -/
def v35TensorPairNormalization : ℂ :=
  ((Fintype.card (Fin 2) : ℕ) : ℂ)⁻¹

theorem v35_spinor_channel_normalization_eq_quarter :
    v35SpinorChannelNormalization = (1 / 4 : ℂ) := by
  norm_num [v35SpinorChannelNormalization]

theorem v35_tensor_pair_normalization_eq_half :
    v35TensorPairNormalization = (1 / 2 : ℂ) := by
  norm_num [v35TensorPairNormalization]

theorem v35_bilinear_channel_count : Fintype.card V35BilinearChannel = 16 := by
  decide

theorem v35_sigma_diagonal_zero (mu : Fin 4) (z : CubicalCell4) :
    v34Sigma mu mu z = 0 := by
  funext component
  simp [v34Sigma]

theorem v35_sigma_antisymmetric (mu nu : Fin 4) (z : CubicalCell4) :
    v34Sigma nu mu z = fun component => -v34Sigma mu nu z component := by
  funext component
  simp [v34Sigma]
  ring

def v35FierzEntry (a b c d : Fin 4) : Prop :=
    standardCell4Basis d a * standardCell4Basis b c =
      v35SpinorChannelNormalization *
        (standardCell4Basis b a * standardCell4Basis d c +
          v34OperatorEntry v35GammaFive a b *
            v34OperatorEntry v35GammaFive c d +
          (∑ mu : Fin 4,
            v34OperatorEntry (v34Gamma mu) a b *
              v34OperatorEntry (v35GammaLower mu) c d) -
          (∑ mu : Fin 4,
            v34OperatorEntry (v35AxialUpper mu) a b *
              v34OperatorEntry (v35AxialLower mu) c d) +
          v35TensorPairNormalization * ∑ mu : Fin 4, ∑ nu : Fin 4,
            v34OperatorEntry (v34Sigma mu nu) a b *
              v34OperatorEntry (v35SigmaLower mu nu) c d)

macro "v35_fierz_finish" : tactic =>
  `(tactic|
    simp [v35FierzEntry, v35SpinorChannelNormalization,
      v35TensorPairNormalization, standardCell4Basis, v34OperatorEntry,
      v35GammaFive,
      v35GammaLower, v35AxialUpper, v35AxialLower, v35SigmaLower,
      v34Sigma, v34Gamma, cubicalGammaTime, cubicalGammaSpace,
      cubicalAlpha, cubicalStandardBeta, cubicalStandardAlphaX,
      cubicalStandardAlphaY, cubicalStandardAlphaZ,
      Fin.sum_univ_four] <;>
    ring_nf <;> norm_num [Complex.I_sq])

set_option maxHeartbeats 500000

private theorem v35_fierz_00 (c d : Fin 4) : v35FierzEntry 0 0 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_01 (c d : Fin 4) : v35FierzEntry 0 1 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_02 (c d : Fin 4) : v35FierzEntry 0 2 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_03 (c d : Fin 4) : v35FierzEntry 0 3 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_10 (c d : Fin 4) : v35FierzEntry 1 0 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_11 (c d : Fin 4) : v35FierzEntry 1 1 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_12 (c d : Fin 4) : v35FierzEntry 1 2 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_13 (c d : Fin 4) : v35FierzEntry 1 3 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_20 (c d : Fin 4) : v35FierzEntry 2 0 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_21 (c d : Fin 4) : v35FierzEntry 2 1 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_22 (c d : Fin 4) : v35FierzEntry 2 2 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_23 (c d : Fin 4) : v35FierzEntry 2 3 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_30 (c d : Fin 4) : v35FierzEntry 3 0 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_31 (c d : Fin 4) : v35FierzEntry 3 1 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_32 (c d : Fin 4) : v35FierzEntry 3 2 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish
private theorem v35_fierz_33 (c d : Fin 4) : v35FierzEntry 3 3 c d := by fin_cases c <;> fin_cases d <;> v35_fierz_finish

set_option maxHeartbeats 200000

/-- The Fierz completeness relation for the sixteen matrices generated by the
cubical Clifford root.  The tensor sum runs over all ordered pairs and
therefore carries the conventional factor `1/2`. -/
theorem v35_fierz_completeness
    (a b c d : Fin 4) :
    standardCell4Basis d a * standardCell4Basis b c =
      v35SpinorChannelNormalization *
        (standardCell4Basis b a * standardCell4Basis d c +
          v34OperatorEntry v35GammaFive a b *
            v34OperatorEntry v35GammaFive c d +
          (∑ mu : Fin 4,
            v34OperatorEntry (v34Gamma mu) a b *
              v34OperatorEntry (v35GammaLower mu) c d) -
          (∑ mu : Fin 4,
            v34OperatorEntry (v35AxialUpper mu) a b *
              v34OperatorEntry (v35AxialLower mu) c d) +
          v35TensorPairNormalization * ∑ mu : Fin 4, ∑ nu : Fin 4,
            v34OperatorEntry (v34Sigma mu nu) a b *
              v34OperatorEntry (v35SigmaLower mu nu) c d) := by
  change v35FierzEntry a b c d
  fin_cases a <;> fin_cases b
  all_goals first
    | exact v35_fierz_00 c d | exact v35_fierz_01 c d
    | exact v35_fierz_02 c d | exact v35_fierz_03 c d
    | exact v35_fierz_10 c d | exact v35_fierz_11 c d
    | exact v35_fierz_12 c d | exact v35_fierz_13 c d
    | exact v35_fierz_20 c d | exact v35_fierz_21 c d
    | exact v35_fierz_22 c d | exact v35_fierz_23 c d
    | exact v35_fierz_30 c d | exact v35_fierz_31 c d
    | exact v35_fierz_32 c d | exact v35_fierz_33 c d

structure V35BilinearCliffordClosureCertificate : Prop where
  channelCount : Fintype.card V35BilinearChannel = 16
  spinorNormalization : v35SpinorChannelNormalization = (1 / 4 : ℂ)
  tensorNormalization : v35TensorPairNormalization = (1 / 2 : ℂ)
  gammaFive : V35CompleteBilinearCertificate
  sigmaDiagonal : ∀ mu z, v34Sigma mu mu z = 0
  sigmaAntisymmetric : ∀ mu nu z,
    v34Sigma nu mu z = fun component => -v34Sigma mu nu z component
  fierzCompleteness : ∀ a b c d,
    standardCell4Basis d a * standardCell4Basis b c =
      v35SpinorChannelNormalization *
        (standardCell4Basis b a * standardCell4Basis d c +
          v34OperatorEntry v35GammaFive a b *
            v34OperatorEntry v35GammaFive c d +
          (∑ mu : Fin 4,
            v34OperatorEntry (v34Gamma mu) a b *
              v34OperatorEntry (v35GammaLower mu) c d) -
          (∑ mu : Fin 4,
            v34OperatorEntry (v35AxialUpper mu) a b *
              v34OperatorEntry (v35AxialLower mu) c d) +
          v35TensorPairNormalization * ∑ mu : Fin 4, ∑ nu : Fin 4,
            v34OperatorEntry (v34Sigma mu nu) a b *
              v34OperatorEntry (v35SigmaLower mu nu) c d)

theorem v35_bilinear_clifford_closure :
    V35BilinearCliffordClosureCertificate where
  channelCount := v35_bilinear_channel_count
  spinorNormalization := v35_spinor_channel_normalization_eq_quarter
  tensorNormalization := v35_tensor_pair_normalization_eq_half
  gammaFive := v35_complete_bilinears
  sigmaDiagonal := v35_sigma_diagonal_zero
  sigmaAntisymmetric := v35_sigma_antisymmetric
  fierzCompleteness := v35_fierz_completeness

#print axioms v35_bilinear_clifford_closure

end
end LeanDirac
