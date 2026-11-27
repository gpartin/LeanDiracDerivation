import LeanDirac.V35FrameworkAndGapLock

/-!
# V3.5 Stage 2: antiparticle residue closure

V3.4 connected the positive spin sum to the positive spectral projector.
This stage proves the corresponding negative-branch statement, including the
momentum reversal required by the antiparticle convention.
-/

namespace LeanDirac

noncomputable section

theorem v35_antiparticle_spin_sum_is_negative_projector_beta
    (mass energy px py pz : ℝ) (henergy : energy ≠ 0) :
    v34AntiparticleCovariantNumerator mass energy px py pz =
      fun row column =>
        (energy : ℂ) * v34OperatorEntry
          (fun z => v32NegativeProjector mass (-px) (-py) (-pz) energy
            (cubicalStandardBeta z)) row column := by
  funext row column
  fin_cases row <;> fin_cases column <;>
    simp [v34AntiparticleCovariantNumerator, v34OperatorEntry,
      v34MomentumSlash, v34CovariantMomentum, v34Gamma,
      cubicalGammaTime, cubicalGammaSpace, cubicalAlpha,
      v32NegativeProjector, v32SpectralProjector,
      v32ContinuumHamiltonian, cubicalStandardHamiltonian,
      cubicalStandardSpatial, cubicalStandardBeta,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ, standardCell4Basis,
      Fin.sum_univ_four] <;>
    field_simp [henergy] <;> ring

theorem v35_antiparticle_spin_sum_is_negative_pole
    (mass energy px py pz : ℝ)
    (hpositive : 0 < energy + mass)
    (hshell : energy ^ 2 = mass ^ 2 + px ^ 2 + py ^ 2 + pz ^ 2)
    (henergy : energy ≠ 0) :
    v34AntiparticleSpinSum mass energy px py pz =
      fun row column =>
        (energy : ℂ) * v34OperatorEntry
          (fun z => v34NegativePoleNumerator mass (-px) (-py) (-pz) energy
            (cubicalStandardBeta z)) row column := by
  rw [v34_antiparticle_covariant_spin_sum mass energy px py pz hpositive hshell]
  exact v35_antiparticle_spin_sum_is_negative_projector_beta
    mass energy px py pz henergy

structure V35AntiparticleResidueClosureCertificate : Prop where
  numeratorProjector : ∀ mass energy px py pz,
    energy ≠ 0 →
      v34AntiparticleCovariantNumerator mass energy px py pz =
        fun row column =>
          (energy : ℂ) * v34OperatorEntry
            (fun z => v32NegativeProjector mass (-px) (-py) (-pz) energy
              (cubicalStandardBeta z)) row column
  spinSumPole : ∀ mass energy px py pz,
    0 < energy + mass →
    energy ^ 2 = mass ^ 2 + px ^ 2 + py ^ 2 + pz ^ 2 →
    energy ≠ 0 →
      v34AntiparticleSpinSum mass energy px py pz =
        fun row column =>
          (energy : ℂ) * v34OperatorEntry
            (fun z => v34NegativePoleNumerator mass (-px) (-py) (-pz) energy
              (cubicalStandardBeta z)) row column

theorem v35_antiparticle_residue_closure :
    V35AntiparticleResidueClosureCertificate where
  numeratorProjector := v35_antiparticle_spin_sum_is_negative_projector_beta
  spinSumPole := v35_antiparticle_spin_sum_is_negative_pole

#print axioms v35_antiparticle_residue_closure

end
end LeanDirac
