import LeanDirac.Dirac003

namespace LeanDirac

/-!
Sole public proof root. DIRAC-003 imports the explicit DIRAC-001/002 gate
modules and is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac003
    (L : ℕ) [NeZero L]
    (chi0 B : ℝ) (state : State (PeriodicCubicSite L))
    (velocity : Velocity (PeriodicCubicSite L))
    (trajectory : NativeTrajectory (PeriodicCubicSite L)) (t : ℝ) :
    ((∀ etaPsi : PeriodicCubicSite L → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B
          (stateLine state (purePsiDirection etaPsi) eps) velocity
          (periodicC19Geometry L))
        (psiEulerLagrangeForce chi0 state
          (c19Bond (periodicC19Geometry L)) etaPsi) 0) ∧
    (∀ etaChi : PeriodicCubicSite L → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B
          (stateLine state (pureChiDirection etaChi) eps) velocity
          (periodicC19Geometry L))
        (chiEulerLagrangeForce chi0 state
          (c19Bond (periodicC19Geometry L)) etaChi) 0) ∧
    (∀ etaPsi : PeriodicCubicSite L → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B state
          (velocityLine velocity (purePsiVelocity etaPsi) eps)
          (periodicC19Geometry L))
        (psiMomentumPair velocity etaPsi) 0) ∧
    (∀ etaChi : PeriodicCubicSite L → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => c19Lagrangian chi0 B state
          (velocityLine velocity (pureChiVelocity etaChi) eps)
          (periodicC19Geometry L))
        (chiMomentumPair B velocity etaChi) 0) ∧
    state = ⟨state.psi, state.chi⟩) ∧
    (NativeActionStationaryAt chi0 B
        (c19Bond (periodicC19Geometry L)) trajectory t ↔
      TwoEulerLagrangeEquationFamiliesAt chi0 B
        (c19Bond (periodicC19Geometry L)) trajectory t) := by
  exact ⟨dirac003_periodic_c19_two_reciprocal_action_variations
      L chi0 B state velocity,
    dirac003_periodic_c19_stationarity_iff_two_equations
      L chi0 B trajectory t⟩

end LeanDirac

