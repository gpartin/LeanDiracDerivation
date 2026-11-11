import LeanDirac.Dirac003Stabilized
import LeanDirac.Dirac004

namespace LeanDirac

/-! DIRAC-004 centered update rebuilt from the stabilized action variation. -/

noncomputable section
open Complex

def stabilizedPsiEulerLagrangeForce [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (etaPsi : Site → C3) : ℝ :=
  stabilizedC19LagrangianStateFirstVariation chi0 state
    (purePsiDirection etaPsi) geometry

def stabilizedChiEulerLagrangeForce [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (etaChi : Site → C3) : ℝ :=
  stabilizedC19LagrangianStateFirstVariation chi0 state
    (pureChiDirection etaChi) geometry

def stabilizedPsiAcceleration [Fintype Site] [DecidableEq Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Site → C3 := fun site component =>
  (stabilizedPsiEulerLagrangeForce chi0 state geometry
    (realCoordinateDirection site component) : ℂ) +
  (stabilizedPsiEulerLagrangeForce chi0 state geometry
    (imagCoordinateDirection site component) : ℂ) * I

def stabilizedChiAcceleration [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Site → C3 := fun site component =>
  ((stabilizedChiEulerLagrangeForce chi0 state geometry
    (realCoordinateDirection site component) / B : ℝ) : ℂ) +
  ((stabilizedChiEulerLagrangeForce chi0 state geometry
    (imagCoordinateDirection site component) / B : ℝ) : ℂ) * I

def stabilizedAccelerationVelocity [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Velocity Site where
  psiDot := stabilizedPsiAcceleration chi0 state geometry
  chiDot := stabilizedChiAcceleration chi0 B state geometry

def stabilizedNativeNextState [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) : State Site :=
  let acceleration := stabilizedAccelerationVelocity chi0 B current geometry
  { psi := fun site component =>
      2 * current.psi site component - previous.psi site component +
        (dt^2 : ℂ) * acceleration.psiDot site component
    chi := fun site component =>
      2 * current.chi site component - previous.chi site component +
        (dt^2 : ℂ) * acceleration.chiDot site component }

theorem stabilized_native_next_is_reversible
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) :
    stabilizedNativeNextState chi0 B dt geometry
        (stabilizedNativeNextState chi0 B dt geometry previous current)
        current = previous := by
  apply congrArg₂ State.mk <;> funext site component <;>
    simp [stabilizedNativeNextState] <;> ring

theorem centered_difference_stabilized_native_next
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) (hdt : dt ≠ 0) :
    centeredSecondDifference dt previous current
        (stabilizedNativeNextState chi0 B dt geometry previous current) =
      stabilizedAccelerationVelocity chi0 B current geometry := by
  apply congrArg₂ Velocity.mk <;> funext site component <;>
    simp [stabilizedNativeNextState, stabilizedAccelerationVelocity] <;>
    field_simp [hdt] <;> ring

theorem stabilized_psi_acceleration_real_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (component : Fin 3) :
    psiMomentumPair (stabilizedAccelerationVelocity chi0 B state geometry)
        (realCoordinateDirection site component) =
      stabilizedPsiEulerLagrangeForce chi0 state geometry
        (realCoordinateDirection site component) := by
  classical
  unfold psiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [stabilizedAccelerationVelocity, stabilizedPsiAcceleration,
        realCoordinateDirection]
    · intro other _ hne; simp [realCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, realCoordinateDirection, hne]
  · simp

theorem stabilized_psi_acceleration_imag_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (component : Fin 3) :
    psiMomentumPair (stabilizedAccelerationVelocity chi0 B state geometry)
        (imagCoordinateDirection site component) =
      stabilizedPsiEulerLagrangeForce chi0 state geometry
        (imagCoordinateDirection site component) := by
  classical
  unfold psiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [stabilizedAccelerationVelocity, stabilizedPsiAcceleration,
        imagCoordinateDirection]
    · intro other _ hne; simp [imagCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, imagCoordinateDirection, hne]
  · simp

theorem stabilized_chi_acceleration_real_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hB : B ≠ 0) (site : Site) (component : Fin 3) :
    chiMomentumPair B (stabilizedAccelerationVelocity chi0 B state geometry)
        (realCoordinateDirection site component) =
      stabilizedChiEulerLagrangeForce chi0 state geometry
        (realCoordinateDirection site component) := by
  classical
  unfold chiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [stabilizedAccelerationVelocity, stabilizedChiAcceleration,
        realCoordinateDirection]
      field_simp
    · intro other _ hne; simp [realCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, realCoordinateDirection, hne]
  · simp

theorem stabilized_chi_acceleration_imag_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hB : B ≠ 0) (site : Site) (component : Fin 3) :
    chiMomentumPair B (stabilizedAccelerationVelocity chi0 B state geometry)
        (imagCoordinateDirection site component) =
      stabilizedChiEulerLagrangeForce chi0 state geometry
        (imagCoordinateDirection site component) := by
  classical
  unfold chiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [stabilizedAccelerationVelocity, stabilizedChiAcceleration,
        imagCoordinateDirection]
      field_simp
    · intro other _ hne; simp [imagCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, imagCoordinateDirection, hne]
  · simp

structure StabilizedNativeUpdateCertificate
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) : Prop where
  actualActionVariations : ∀ velocity : Velocity Site,
    StabilizedLocalTwoFieldVariationCertificate chi0 B current velocity geometry
  reversible : stabilizedNativeNextState chi0 B dt geometry
    (stabilizedNativeNextState chi0 B dt geometry previous current) current = previous
  centeredEquation : centeredSecondDifference dt previous current
    (stabilizedNativeNextState chi0 B dt geometry previous current) =
      stabilizedAccelerationVelocity chi0 B current geometry
  psiRealEquation : ∀ site component,
    psiMomentumPair (stabilizedAccelerationVelocity chi0 B current geometry)
      (realCoordinateDirection site component) =
    stabilizedPsiEulerLagrangeForce chi0 current geometry
      (realCoordinateDirection site component)
  psiImagEquation : ∀ site component,
    psiMomentumPair (stabilizedAccelerationVelocity chi0 B current geometry)
      (imagCoordinateDirection site component) =
    stabilizedPsiEulerLagrangeForce chi0 current geometry
      (imagCoordinateDirection site component)
  chiRealEquation : ∀ site component,
    chiMomentumPair B (stabilizedAccelerationVelocity chi0 B current geometry)
      (realCoordinateDirection site component) =
    stabilizedChiEulerLagrangeForce chi0 current geometry
      (realCoordinateDirection site component)
  chiImagEquation : ∀ site component,
    chiMomentumPair B (stabilizedAccelerationVelocity chi0 B current geometry)
      (imagCoordinateDirection site component) =
    stabilizedChiEulerLagrangeForce chi0 current geometry
      (imagCoordinateDirection site component)

theorem dirac004_stabilized_c19_reversible_action_update
    [Fintype Site] [DecidableEq Site]
    (dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) (hdt : dt ≠ 0) :
    StabilizedNativeUpdateCertificate chi0 chiKineticNormalization dt
      geometry previous current := by
  have hB : chiKineticNormalization ≠ 0 := by
    rw [chi_kinetic_normalization_eq_1197]; norm_num
  constructor
  · exact fun velocity => dirac003_stabilized_c19_variations_from_actual_action
      chi0 chiKineticNormalization current velocity geometry
  · exact stabilized_native_next_is_reversible
      chi0 chiKineticNormalization dt geometry previous current
  · exact centered_difference_stabilized_native_next
      chi0 chiKineticNormalization dt geometry previous current hdt
  · exact fun site component => stabilized_psi_acceleration_real_coordinate
      chi0 chiKineticNormalization current geometry site component
  · exact fun site component => stabilized_psi_acceleration_imag_coordinate
      chi0 chiKineticNormalization current geometry site component
  · exact fun site component => stabilized_chi_acceleration_real_coordinate
      chi0 chiKineticNormalization current geometry hB site component
  · exact fun site component => stabilized_chi_acceleration_imag_coordinate
      chi0 chiKineticNormalization current geometry hB site component

end
end LeanDirac
