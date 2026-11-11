import LeanDirac.ChiIncidenceVariation
import LeanDirac.Dirac004

namespace LeanDirac

/-!
DIRAC-004 for the incidence action admitted by DIRAC-002.  The centered
update is reconstructed coordinate by coordinate from the derivative proved
in DIRAC-003; no independent force is supplied.
-/

noncomputable section
open Complex

def incidencePsiEulerLagrangeForce [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (etaPsi : Site → C3) : ℝ :=
  incidenceC19LagrangianStateFirstVariation chi0 state
    (purePsiDirection etaPsi) geometry

def incidenceChiEulerLagrangeForce [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (etaChi : Site → C3) : ℝ :=
  incidenceC19LagrangianStateFirstVariation chi0 state
    (pureChiDirection etaChi) geometry

def incidencePsiAcceleration [Fintype Site] [DecidableEq Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Site → C3 := fun site component =>
  (incidencePsiEulerLagrangeForce chi0 state geometry
    (realCoordinateDirection site component) : ℂ) +
  (incidencePsiEulerLagrangeForce chi0 state geometry
    (imagCoordinateDirection site component) : ℂ) * I

def incidenceChiAcceleration [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Site → C3 := fun site component =>
  ((incidenceChiEulerLagrangeForce chi0 state geometry
    (realCoordinateDirection site component) / B : ℝ) : ℂ) +
  ((incidenceChiEulerLagrangeForce chi0 state geometry
    (imagCoordinateDirection site component) / B : ℝ) : ℂ) * I

def incidenceAccelerationVelocity [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    Velocity Site where
  psiDot := incidencePsiAcceleration chi0 state geometry
  chiDot := incidenceChiAcceleration chi0 B state geometry

def incidenceNativeNextState [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) : State Site :=
  let acceleration := incidenceAccelerationVelocity chi0 B current geometry
  { psi := fun site component =>
      2 * current.psi site component - previous.psi site component +
        (dt^2 : ℂ) * acceleration.psiDot site component
    chi := fun site component =>
      2 * current.chi site component - previous.chi site component +
        (dt^2 : ℂ) * acceleration.chiDot site component }

theorem incidence_native_next_is_reversible
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) :
    incidenceNativeNextState chi0 B dt geometry
        (incidenceNativeNextState chi0 B dt geometry previous current)
        current = previous := by
  apply congrArg₂ State.mk <;> funext site component <;>
    simp [incidenceNativeNextState] <;> ring

theorem centered_difference_incidence_native_next
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) (hdt : dt ≠ 0) :
    centeredSecondDifference dt previous current
        (incidenceNativeNextState chi0 B dt geometry previous current) =
      incidenceAccelerationVelocity chi0 B current geometry := by
  apply congrArg₂ Velocity.mk <;> funext site component <;>
    simp [incidenceNativeNextState, incidenceAccelerationVelocity] <;>
    field_simp [hdt] <;> ring

theorem incidence_psi_acceleration_real_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (component : Fin 3) :
    psiMomentumPair (incidenceAccelerationVelocity chi0 B state geometry)
        (realCoordinateDirection site component) =
      incidencePsiEulerLagrangeForce chi0 state geometry
        (realCoordinateDirection site component) := by
  classical
  unfold psiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [incidenceAccelerationVelocity, incidencePsiAcceleration,
        realCoordinateDirection]
    · intro other _ hne; simp [realCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, realCoordinateDirection, hne]
  · simp

theorem incidence_psi_acceleration_imag_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (component : Fin 3) :
    psiMomentumPair (incidenceAccelerationVelocity chi0 B state geometry)
        (imagCoordinateDirection site component) =
      incidencePsiEulerLagrangeForce chi0 state geometry
        (imagCoordinateDirection site component) := by
  classical
  unfold psiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [incidenceAccelerationVelocity, incidencePsiAcceleration,
        imagCoordinateDirection]
    · intro other _ hne; simp [imagCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, imagCoordinateDirection, hne]
  · simp

theorem incidence_chi_acceleration_real_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hB : B ≠ 0) (site : Site) (component : Fin 3) :
    chiMomentumPair B (incidenceAccelerationVelocity chi0 B state geometry)
        (realCoordinateDirection site component) =
      incidenceChiEulerLagrangeForce chi0 state geometry
        (realCoordinateDirection site component) := by
  classical
  unfold chiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [incidenceAccelerationVelocity, incidenceChiAcceleration,
        realCoordinateDirection]
      field_simp
    · intro other _ hne; simp [realCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, realCoordinateDirection, hne]
  · simp

theorem incidence_chi_acceleration_imag_coordinate
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hB : B ≠ 0) (site : Site) (component : Fin 3) :
    chiMomentumPair B (incidenceAccelerationVelocity chi0 B state geometry)
        (imagCoordinateDirection site component) =
      incidenceChiEulerLagrangeForce chi0 state geometry
        (imagCoordinateDirection site component) := by
  classical
  unfold chiMomentumPair
  rw [Finset.sum_eq_single site]
  · unfold c3RealPair
    rw [Finset.sum_eq_single component]
    · simp [incidenceAccelerationVelocity, incidenceChiAcceleration,
        imagCoordinateDirection]
      field_simp
    · intro other _ hne; simp [imagCoordinateDirection, hne]
    · simp
  · intro other _ hne; simp [c3RealPair, imagCoordinateDirection, hne]
  · simp

structure IncidenceNativeUpdateCertificate
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) : Prop where
  actualActionVariations : ∀ velocity : Velocity Site,
    IncidenceLocalTwoFieldVariationCertificate chi0 B current velocity geometry
  reversible : incidenceNativeNextState chi0 B dt geometry
    (incidenceNativeNextState chi0 B dt geometry previous current) current = previous
  centeredEquation : centeredSecondDifference dt previous current
    (incidenceNativeNextState chi0 B dt geometry previous current) =
      incidenceAccelerationVelocity chi0 B current geometry
  psiRealEquation : ∀ site component,
    psiMomentumPair (incidenceAccelerationVelocity chi0 B current geometry)
      (realCoordinateDirection site component) =
    incidencePsiEulerLagrangeForce chi0 current geometry
      (realCoordinateDirection site component)
  psiImagEquation : ∀ site component,
    psiMomentumPair (incidenceAccelerationVelocity chi0 B current geometry)
      (imagCoordinateDirection site component) =
    incidencePsiEulerLagrangeForce chi0 current geometry
      (imagCoordinateDirection site component)
  chiRealEquation : ∀ site component,
    chiMomentumPair B (incidenceAccelerationVelocity chi0 B current geometry)
      (realCoordinateDirection site component) =
    incidenceChiEulerLagrangeForce chi0 current geometry
      (realCoordinateDirection site component)
  chiImagEquation : ∀ site component,
    chiMomentumPair B (incidenceAccelerationVelocity chi0 B current geometry)
      (imagCoordinateDirection site component) =
    incidenceChiEulerLagrangeForce chi0 current geometry
      (imagCoordinateDirection site component)

theorem dirac004_incidence_c19_reversible_action_update
    [Fintype Site] [DecidableEq Site]
    (dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) (hdt : dt ≠ 0) :
    IncidenceNativeUpdateCertificate chi0 chiKineticNormalization dt
      geometry previous current := by
  have hB : chiKineticNormalization ≠ 0 := by
    rw [chi_kinetic_normalization_eq_1197]; norm_num
  constructor
  · exact fun velocity => incidence_c19_variations_from_actual_action
      chi0 chiKineticNormalization current velocity geometry
  · exact incidence_native_next_is_reversible
      chi0 chiKineticNormalization dt geometry previous current
  · exact centered_difference_incidence_native_next
      chi0 chiKineticNormalization dt geometry previous current hdt
  · exact fun site component => incidence_psi_acceleration_real_coordinate
      chi0 chiKineticNormalization current geometry site component
  · exact fun site component => incidence_psi_acceleration_imag_coordinate
      chi0 chiKineticNormalization current geometry site component
  · exact fun site component => incidence_chi_acceleration_real_coordinate
      chi0 chiKineticNormalization current geometry hB site component
  · exact fun site component => incidence_chi_acceleration_imag_coordinate
      chi0 chiKineticNormalization current geometry hB site component

end
end LeanDirac
