import LeanDirac.V36KineticMetricClassification

/-!
# V3.6 Stage 5: radial-response consequences

The local chi potential already present in the repository fixes the radial
restoring coefficient.  Combining it with a normalization pair yields the
candidate squared small-oscillation frequencies.  No candidate is selected
from numerical performance.
-/

namespace LeanDirac

noncomputable section

def v36RadialScalarState (q : ℝ) : State Unit where
  psi := fun _ => 0
  chi := fun _ component => if component = 0 then q else 0

def v36RadialScalarPotential (q : ℝ) : ℝ :=
  (q^2 - chi0^2)^2 / 4

theorem v36_radial_scalar_state_chi_density (q : ℝ) :
    density ((v36RadialScalarState q).chi ()) = q^2 := by
  simp [v36RadialScalarState, density, Fin.sum_univ_three, Complex.normSq]
  ring

theorem v36_radial_scalar_potential_is_native_site_slice (q : ℝ) :
    sitePotential chi0 (v36RadialScalarState q) () =
      v36RadialScalarPotential q := by
  simp [sitePotential, v36RadialScalarState, v36RadialScalarPotential,
    density, Fin.sum_univ_three, Complex.normSq]
  ring

def v36RadialScalarForce (q : ℝ) : ℝ :=
  -(q^2 - chi0^2) * q

theorem v36_radial_force_expansion (eps : ℝ) :
    v36RadialScalarForce (chi0 + eps) =
      -(2 * chi0^2) * eps - 3 * chi0 * eps^2 - eps^3 := by
  simp [v36RadialScalarForce]
  ring

def v36RadialFrequencySquareFromB (candidateB : ℝ) : ℝ :=
  2 * chi0^2 / candidateB

def v36RadialFrequencySquareFromKappa (candidateKappa : ℝ) : ℝ :=
  2 * chi0 * candidateKappa

theorem v36_frequency_parameterizations_agree
    (candidateKappa candidateB : ℝ) (hB : candidateB ≠ 0)
    (hrelation : V36NormalizationRelation candidateKappa candidateB) :
    v36RadialFrequencySquareFromB candidateB =
      v36RadialFrequencySquareFromKappa candidateKappa := by
  unfold v36RadialFrequencySquareFromB
    v36RadialFrequencySquareFromKappa V36NormalizationRelation at *
  rw [← hrelation]
  field_simp [hB]

theorem v36_legacy_radial_frequency_square :
    v36RadialFrequencySquareFromKappa v36LegacyKappa = 38 / 63 := by
  norm_num [v36RadialFrequencySquareFromKappa, v36LegacyKappa, chi0]

theorem v36_c19_weight_radial_frequency_square :
    v36RadialFrequencySquareFromKappa v36C19WeightKappa = 38 := by
  norm_num [v36RadialFrequencySquareFromKappa, v36C19WeightKappa, chi0]

theorem v36_equal_metric_radial_frequency_square :
    v36RadialFrequencySquareFromKappa v36EqualMetricKappa = 722 := by
  norm_num [v36RadialFrequencySquareFromKappa, v36EqualMetricKappa, chi0]

theorem v36_candidate_radial_frequency_squares_strictly_ordered :
    v36RadialFrequencySquareFromKappa v36LegacyKappa <
      v36RadialFrequencySquareFromKappa v36C19WeightKappa ∧
    v36RadialFrequencySquareFromKappa v36C19WeightKappa <
      v36RadialFrequencySquareFromKappa v36EqualMetricKappa := by
  norm_num [v36RadialFrequencySquareFromKappa, v36LegacyKappa,
    v36C19WeightKappa, v36EqualMetricKappa, chi0]

structure V36RadialResponseCertificate : Prop where
  nativePotentialSlice : ∀ q,
    sitePotential chi0 (v36RadialScalarState q) () =
      v36RadialScalarPotential q
  nativeForce : ∀ eps,
    v36RadialScalarForce (chi0 + eps) =
      -(2 * chi0^2) * eps - 3 * chi0 * eps^2 - eps^3
  parameterBridge : ∀ candidateKappa candidateB,
    candidateB ≠ 0 →
    V36NormalizationRelation candidateKappa candidateB →
    v36RadialFrequencySquareFromB candidateB =
      v36RadialFrequencySquareFromKappa candidateKappa
  legacySquare :
    v36RadialFrequencySquareFromKappa v36LegacyKappa = 38 / 63
  c19WeightSquare :
    v36RadialFrequencySquareFromKappa v36C19WeightKappa = 38
  equalMetricSquare :
    v36RadialFrequencySquareFromKappa v36EqualMetricKappa = 722
  ordering :
    v36RadialFrequencySquareFromKappa v36LegacyKappa <
      v36RadialFrequencySquareFromKappa v36C19WeightKappa ∧
    v36RadialFrequencySquareFromKappa v36C19WeightKappa <
      v36RadialFrequencySquareFromKappa v36EqualMetricKappa

theorem v36_radial_response : V36RadialResponseCertificate where
  nativePotentialSlice := v36_radial_scalar_potential_is_native_site_slice
  nativeForce := v36_radial_force_expansion
  parameterBridge := v36_frequency_parameterizations_agree
  legacySquare := v36_legacy_radial_frequency_square
  c19WeightSquare := v36_c19_weight_radial_frequency_square
  equalMetricSquare := v36_equal_metric_radial_frequency_square
  ordering := v36_candidate_radial_frequency_squares_strictly_ordered

#print axioms v36_radial_response

end
end LeanDirac

