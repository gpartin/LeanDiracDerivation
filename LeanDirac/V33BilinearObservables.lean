import LeanDirac.V33SpinSums
import LeanDirac.PositiveFrequencyCurrentBridge

/-!
# V3.3 Stage 5: normalized free-spinor observables
-/

namespace LeanDirac

noncomputable section

def v33ScalarBilinear (z : CubicalCell4) : ℝ :=
  Complex.normSq (z 0) + Complex.normSq (z 1) -
    Complex.normSq (z 2) - Complex.normSq (z 3)

theorem v33_normalized_scalar_bilinear_formula
    (mass energy px py pz : ℝ) (q : UniformCell2) :
    v33ScalarBilinear
        (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
      positiveFrequencyScale mass energy ^ 2 * uniformCell2Density q -
        positiveFrequencyRatio mass energy ^ 2 *
          uniformCell2Density (positivePauliSpatial px py pz q) := by
  simp [v33ScalarBilinear, normalizedPositiveFrequencyEmbedding,
    uniformCell2Density, Complex.normSq, Complex.mul_re, Complex.mul_im,
    Fin.sum_univ_succ]
  ring

theorem v33_normalized_scalar_bilinear
    (mass energy px py pz : ℝ)
    (hpositive : 0 < energy + mass)
    (hshell : energy^2 = mass^2 + px^2 + py^2 + pz^2)
    (q : UniformCell2) :
    v33ScalarBilinear
        (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
      mass * uniformCell2Density q := by
  have hdenom : energy + mass ≠ 0 := ne_of_gt hpositive
  have hscale := positive_frequency_scale_square mass energy (le_of_lt hpositive)
  rw [v33_normalized_scalar_bilinear_formula]
  rw [positive_pauli_spatial_density, positiveFrequencyRatio]
  have hcoefficient :
      positiveFrequencyScale mass energy ^ 2 -
          (positiveFrequencyScale mass energy / (energy + mass)) ^ 2 *
            (px ^ 2 + py ^ 2 + pz ^ 2) = mass := by
    field_simp [hdenom]
    rw [hscale]
    nlinarith
  calc
    positiveFrequencyScale mass energy ^ 2 * uniformCell2Density q -
        (positiveFrequencyScale mass energy / (energy + mass)) ^ 2 *
          ((px ^ 2 + py ^ 2 + pz ^ 2) * uniformCell2Density q) =
      (positiveFrequencyScale mass energy ^ 2 -
        (positiveFrequencyScale mass energy / (energy + mass)) ^ 2 *
          (px ^ 2 + py ^ 2 + pz ^ 2)) * uniformCell2Density q := by ring
    _ = mass * uniformCell2Density q := by rw [hcoefficient]

structure V33BilinearObservableCertificate : Prop where
  density : ∀ mass energy px py pz : ℝ,
    0 < energy + mass →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 → ∀ q : UniformCell2,
      cubicalCell4Density
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
        energy * uniformCell2Density q
  current : ∀ mass energy px py pz : ℝ,
    0 < energy + mass →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 →
      ∀ (q : UniformCell2) (axis : Fin 3),
      cubicalDiracCurrent axis
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
        c19ContinuumSpeed *
          (fun axis : Fin 3 =>
            if axis = 0 then px else if axis = 1 then py else pz) axis *
          uniformCell2Density q
  scalar : ∀ mass energy px py pz : ℝ,
    0 < energy + mass →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 → ∀ q : UniformCell2,
      v33ScalarBilinear
          (normalizedPositiveFrequencyEmbedding mass energy px py pz q) =
        mass * uniformCell2Density q
  nativeOrigin : ∀ mass energy px py pz : ℝ,
    mass ≠ 0 → energy + mass ≠ 0 →
    energy^2 = mass^2 + px^2 + py^2 + pz^2 → ∀ q : UniformCell2,
      ∃ phase : UniformTwoChannelPhase,
        cubicalPhaseReadout (mass : ℂ) (px : ℂ) (py : ℂ) (pz : ℂ) phase =
            normalizedPositiveFrequencyEmbedding mass energy px py pz q ∧
          uniformTwoChannelKleinGordonVector
              (mass : ℂ) (px : ℂ) (py : ℂ) (pz : ℂ) phase =
            uniformTwoChannelScale (-Complex.I * (energy : ℂ)) phase

theorem v33_bilinear_observables : V33BilinearObservableCertificate where
  density := normalized_positive_frequency_density
  current := normalized_positive_frequency_current
  scalar := v33_normalized_scalar_bilinear
  nativeOrigin := normalized_embedding_has_action_positive_preimage

#print axioms v33_bilinear_observables

end
end LeanDirac
