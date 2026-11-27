import LeanDirac.V33SpinorBasis

/-!
# V3.3 Stage 3: on-shell projected spinors

The states are obtained by applying the V3.2 spectral projectors to the
derived four-component basis.  They are not independently postulated.
-/

namespace LeanDirac

noncomputable section

def v33PositiveSpinor
    (mass px py pz energy : ℝ) (seed : Fin 4) : CubicalCell4 :=
  v32PositiveProjector mass px py pz energy (standardCell4Basis seed)

def v33NegativeSpinor
    (mass px py pz energy : ℝ) (seed : Fin 4) : CubicalCell4 :=
  v32NegativeProjector mass px py pz energy (standardCell4Basis seed)

theorem v33_positive_spinor_is_eigenstate
    (mass px py pz energy : ℝ)
    (hshell : energy^2 = v32ContinuumEnergySq mass px py pz)
    (hne : energy ≠ 0) (seed : Fin 4) :
    v32ContinuumHamiltonian mass px py pz
        (v33PositiveSpinor mass px py pz energy seed) =
      fun component => (energy : ℂ) *
        v33PositiveSpinor mass px py pz energy seed component := by
  exact v32_positive_projector_is_positive_branch
    mass px py pz energy hshell hne (standardCell4Basis seed)

theorem v33_negative_spinor_is_eigenstate
    (mass px py pz energy : ℝ)
    (hshell : energy^2 = v32ContinuumEnergySq mass px py pz)
    (hne : energy ≠ 0) (seed : Fin 4) :
    v32ContinuumHamiltonian mass px py pz
        (v33NegativeSpinor mass px py pz energy seed) =
      fun component => (-energy : ℂ) *
        v33NegativeSpinor mass px py pz energy seed component := by
  exact v32_negative_projector_is_negative_branch
    mass px py pz energy hshell hne (standardCell4Basis seed)

theorem v33_seed_splits_into_energy_branches
    (mass px py pz energy : ℝ) (seed : Fin 4) :
    (fun component =>
      v33PositiveSpinor mass px py pz energy seed component +
        v33NegativeSpinor mass px py pz energy seed component) =
      standardCell4Basis seed := by
  exact v32_projectors_complete mass px py pz energy
    (standardCell4Basis seed)

structure V33ProjectedSpinorCertificate : Prop where
  positiveEigenstate : ∀ mass px py pz energy,
    energy^2 = v32ContinuumEnergySq mass px py pz → energy ≠ 0 → ∀ seed,
      v32ContinuumHamiltonian mass px py pz
          (v33PositiveSpinor mass px py pz energy seed) =
        fun component => (energy : ℂ) *
          v33PositiveSpinor mass px py pz energy seed component
  negativeEigenstate : ∀ mass px py pz energy,
    energy^2 = v32ContinuumEnergySq mass px py pz → energy ≠ 0 → ∀ seed,
      v32ContinuumHamiltonian mass px py pz
          (v33NegativeSpinor mass px py pz energy seed) =
        fun component => (-energy : ℂ) *
          v33NegativeSpinor mass px py pz energy seed component
  seedCompleteness : ∀ mass px py pz energy seed,
    (fun component =>
      v33PositiveSpinor mass px py pz energy seed component +
        v33NegativeSpinor mass px py pz energy seed component) =
      standardCell4Basis seed

theorem v33_projected_spinors : V33ProjectedSpinorCertificate where
  positiveEigenstate := v33_positive_spinor_is_eigenstate
  negativeEigenstate := v33_negative_spinor_is_eigenstate
  seedCompleteness := v33_seed_splits_into_energy_branches

#print axioms v33_projected_spinors

end
end LeanDirac
