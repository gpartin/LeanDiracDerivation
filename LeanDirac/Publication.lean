import LeanDirac.Dirac003

namespace LeanDirac

/-!
Sole public proof root. DIRAC-003 imports the explicit DIRAC-001/002 gate
modules and is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac003
    [Fintype Site]
    (chi0 B : ℝ) (geometry : C19Lattice Site)
    (trajectory : NativeTrajectory Site)
    (hcompatible : TrajectoryVelocityCompatible trajectory)
    (psiData : ∀ etaPsi : Site → C3,
      PsiProjectedActionData chi0 (c19Bond geometry) trajectory etaPsi)
    (chiData : ∀ etaChi : Site → C3,
      ChiProjectedActionData chi0 B (c19Bond geometry) trajectory etaChi)
    (hPsiStationary : ∀ etaPsi : Site → C3,
      HasStationaryPsiAction chi0 B (c19Bond geometry) trajectory etaPsi)
    (hChiStationary : ∀ etaChi : Site → C3,
      HasStationaryChiAction chi0 B (c19Bond geometry) trajectory etaChi)
    (hPsiControl : ∀ (etaPsi : Site → C3) (test : SchwartzMap ℝ ℝ),
      IntegralVariationControl (psiActionIncrementIntegrand
        chi0 B (c19Bond geometry) trajectory etaPsi test))
    (hChiControl : ∀ (etaChi : Site → C3) (test : SchwartzMap ℝ ℝ),
      IntegralVariationControl (chiActionIncrementIntegrand
        chi0 B (c19Bond geometry) trajectory etaChi test)) :
    TrajectoryVelocityCompatible trajectory ∧
      ∀ t : ℝ, TwoEulerLagrangeEquationFamiliesAt
        chi0 B (c19Bond geometry) trajectory t := by
  exact dirac003_c19_equations_from_actual_stationary_action
    chi0 B geometry trajectory hcompatible psiData chiData
    hPsiStationary hChiStationary hPsiControl hChiControl

end LeanDirac

