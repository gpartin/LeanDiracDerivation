import LeanDirac.ShapeStabilityVariation

namespace LeanDirac

/-!
DIRAC-003 revalidation for the exact stabilized DIRAC-002 action.

The covariance correction is varied as an explicit polynomial of the line
parameter.  Its derivative is therefore obtained from the action rather than
declared as a force law.
-/

noncomputable section

def stabilizedC19LagrangianStateFirstVariation [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) : ℝ :=
  lagrangianStateFirstVariation chi0 state direction (c19Bond geometry) -
    chiCovarianceShapeFirstVariation chi0 state direction geometry

theorem stabilized_c19_lagrangian_state_hasDerivAt [Fintype Site]
    (chi0 B : ℝ) (state direction : State Site)
    (velocity : Velocity Site) (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => stabilizedC19Lagrangian chi0 B
        (stateLine state direction eps) velocity geometry)
      (stabilizedC19LagrangianStateFirstVariation
        chi0 state direction geometry) 0 := by
  have hbase := lagrangian_state_hasDerivAt chi0 B state direction
    velocity (c19Bond geometry)
  have hshape := chi_covariance_shape_hasDerivAt
    chi0 state direction geometry
  have h := hbase.sub hshape
  convert h using 1
  · funext eps
    simp [stabilizedC19Lagrangian, stabilizedC19Potential,
      lagrangian, c19Potential]
    ring
  · rfl

theorem stabilized_c19_lagrangian_velocity_hasDerivAt [Fintype Site]
    (chi0 B : ℝ) (state : State Site)
    (velocity direction : Velocity Site) (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => stabilizedC19Lagrangian chi0 B state
        (velocityLine velocity direction eps) geometry)
      (kineticVelocityFirstVariation B velocity direction) 0 := by
  have hkinetic := kinetic_velocity_hasDerivAt B velocity direction
  have hpotential := hasDerivAt_const (𝕜 := ℝ) 0
    (stabilizedC19Potential chi0 state geometry)
  have h := hkinetic.sub hpotential
  convert h using 1
  · funext eps
    rfl
  · simp

structure StabilizedLocalTwoFieldVariationCertificate
    [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : Prop where
  psiState : ∀ etaPsi : Site → C3,
    AnalyticHasDerivAt
      (fun eps : ℝ => stabilizedC19Lagrangian chi0 B
        (stateLine state (purePsiDirection etaPsi) eps) velocity geometry)
      (stabilizedC19LagrangianStateFirstVariation chi0 state
        (purePsiDirection etaPsi) geometry) 0
  chiState : ∀ etaChi : Site → C3,
    AnalyticHasDerivAt
      (fun eps : ℝ => stabilizedC19Lagrangian chi0 B
        (stateLine state (pureChiDirection etaChi) eps) velocity geometry)
      (stabilizedC19LagrangianStateFirstVariation chi0 state
        (pureChiDirection etaChi) geometry) 0
  velocity : ∀ direction : Velocity Site,
    AnalyticHasDerivAt
      (fun eps : ℝ => stabilizedC19Lagrangian chi0 B state
        (velocityLine velocity direction eps) geometry)
      (kineticVelocityFirstVariation B velocity direction) 0

theorem dirac003_stabilized_c19_variations_from_actual_action
    [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) :
    StabilizedLocalTwoFieldVariationCertificate
      chi0 B state velocity geometry where
  psiState := fun etaPsi =>
    stabilized_c19_lagrangian_state_hasDerivAt chi0 B state
      (purePsiDirection etaPsi) velocity geometry
  chiState := fun etaChi =>
    stabilized_c19_lagrangian_state_hasDerivAt chi0 B state
      (pureChiDirection etaChi) velocity geometry
  velocity := fun direction =>
    stabilized_c19_lagrangian_velocity_hasDerivAt
      chi0 B state velocity direction geometry

end
end LeanDirac
