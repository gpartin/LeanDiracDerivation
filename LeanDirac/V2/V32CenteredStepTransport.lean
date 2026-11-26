import LeanDirac.V2.V32TransportObservables
import LeanDirac.V2.UniformVacuumVerletDispersion
import LeanDirac.V2.UniformVacuumVerletStability

/-! # V3.2 Stage 15: exact centered-step spectrum -/

namespace LeanDirac.V2

noncomputable section
open Filter Topology

structure V32CenteredStepTransportCertificate : Prop where
  exactDispersion : ∀ r dt momentum,
    uniformVerletStable r dt momentum →
      Real.cos (dt * uniformVerletAngularFrequency r dt momentum) =
        1 - uniformVerletStabilityParameter r dt momentum / 2
  rootHasUnitNorm : ∀ r dt momentum,
    ‖uniformVerletCharacteristicRoot r dt momentum‖ = 1
  rootSolvesRecurrence : ∀ r dt momentum,
    uniformVerletStable r dt momentum →
      uniformVerletCharacteristicRoot r dt momentum ^ 2 -
        ((2 - uniformVerletStabilityParameter r dt momentum : ℝ) : ℂ) *
          uniformVerletCharacteristicRoot r dt momentum + 1 = 0
  modifiedEnergyInvariant : ∀ r dt momentum phase,
    uniformVerletModifiedEnergy r dt momentum
        (uniformVerletPhaseStep r dt momentum phase) =
      uniformVerletModifiedEnergy r dt momentum phase
  numericalFrequencyConverges : ∀ r momentum,
    Tendsto (fun dt : ℝ => uniformVerletAngularFrequency r dt momentum)
      (nhds 0) (nhds (uniformVacuumModalFrequency r momentum))
  exactStepAngle : ∀ r dt momentum,
    dt * uniformVerletAngularFrequency r dt momentum =
      2 * Real.arcsin
        (dt * uniformVacuumModalFrequency r momentum / 2)

theorem v32_centered_step_transport : V32CenteredStepTransportCertificate where
  exactDispersion := uniform_verlet_exact_dispersion
  rootHasUnitNorm := uniform_verlet_characteristic_root_norm
  rootSolvesRecurrence := uniform_verlet_characteristic_root_equation
  modifiedEnergyInvariant := uniform_verlet_modified_energy_invariant
  numericalFrequencyConverges :=
    uniform_verlet_angular_frequency_tends_to_native
  exactStepAngle := uniform_verlet_step_angle

#print axioms v32_centered_step_transport

end
end LeanDirac.V2
