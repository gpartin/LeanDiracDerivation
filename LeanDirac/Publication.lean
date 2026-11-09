import LeanDirac.Dirac003

namespace LeanDirac

/-!
Sole public proof root. DIRAC-003 imports the explicit DIRAC-001/002 gate
modules and is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac003
    [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) :
    (∀ etaPsi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B
          (stateLine state (purePsiDirection etaPsi) eps) velocity geometry)
        (psiEulerLagrangeForce chi0 state (c19Bond geometry) etaPsi) 0) ∧
    (∀ etaChi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B
          (stateLine state (pureChiDirection etaChi) eps) velocity geometry)
        (chiEulerLagrangeForce chi0 state (c19Bond geometry) etaChi) 0) ∧
    (∀ etaPsi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B state
          (velocityLine velocity (purePsiVelocity etaPsi) eps) geometry)
        (psiMomentumPair velocity etaPsi) 0) ∧
    (∀ etaChi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B state
          (velocityLine velocity (pureChiVelocity etaChi) eps) geometry)
        (chiMomentumPair B velocity etaChi) 0) ∧
    state = ⟨state.psi, state.chi⟩ :=
  dirac003_c19_two_reciprocal_action_variations chi0 B state velocity geometry

end LeanDirac

