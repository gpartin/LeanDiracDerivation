import LeanDirac.Dirac003

namespace LeanDirac

/-!
Independent definitional contracts for the DIRAC-003 proof boundary.

These definitions deliberately repeat the expected action-increment and
stationarity shapes without referring to the corresponding DIRAC-003
definitions.  The `Iff.rfl`/`rfl` proofs below therefore fail if a target
equation is added directly, hidden behind an alias, or expanded inline in a
stationarity predicate.
-/

noncomputable section

def ExpectedPsiActionIncrementIntegrand
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3)
    (test : SchwartzMap ℝ ℝ) (eps t : ℝ) : ℝ :=
  let direction := scaleC3Field (test t) etaPsi
  let velocityDirection := scaleC3Field (deriv test t) etaPsi
  lagrangian chi0 B
      (stateLine (trajectory.state t) (purePsiDirection direction) eps)
      (velocityLine (trajectory.velocity t)
        (purePsiVelocity velocityDirection) eps) bond -
    lagrangian chi0 B (trajectory.state t) (trajectory.velocity t) bond

def ExpectedChiActionIncrementIntegrand
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3)
    (test : SchwartzMap ℝ ℝ) (eps t : ℝ) : ℝ :=
  let direction := scaleC3Field (test t) etaChi
  let velocityDirection := scaleC3Field (deriv test t) etaChi
  lagrangian chi0 B
      (stateLine (trajectory.state t) (pureChiDirection direction) eps)
      (velocityLine (trajectory.velocity t)
        (pureChiVelocity velocityDirection) eps) bond -
    lagrangian chi0 B (trajectory.state t) (trajectory.velocity t) bond

def ExpectedPsiActionIncrement
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3)
    (test : SchwartzMap ℝ ℝ) (eps : ℝ) : ℝ :=
  ∫ t : ℝ,
    ExpectedPsiActionIncrementIntegrand
      chi0 B bond trajectory etaPsi test eps t

def ExpectedChiActionIncrement
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3)
    (test : SchwartzMap ℝ ℝ) (eps : ℝ) : ℝ :=
  ∫ t : ℝ,
    ExpectedChiActionIncrementIntegrand
      chi0 B bond trajectory etaChi test eps t

def ExpectedStationaryPsiAction
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3) : Prop :=
  ∀ test : SchwartzMap ℝ ℝ,
    AnalyticHasDerivAt
      (ExpectedPsiActionIncrement chi0 B bond trajectory etaPsi test) 0 0

def ExpectedStationaryChiAction
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3) : Prop :=
  ∀ test : SchwartzMap ℝ ℝ,
    AnalyticHasDerivAt
      (ExpectedChiActionIncrement chi0 B bond trajectory etaChi test) 0 0

theorem psi_action_increment_integrand_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3) :
    psiActionIncrementIntegrand chi0 B bond trajectory etaPsi =
      ExpectedPsiActionIncrementIntegrand
        chi0 B bond trajectory etaPsi := by
  rfl

theorem chi_action_increment_integrand_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3) :
    chiActionIncrementIntegrand chi0 B bond trajectory etaChi =
      ExpectedChiActionIncrementIntegrand
        chi0 B bond trajectory etaChi := by
  rfl

theorem psi_action_increment_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3) :
    psiActionIncrement chi0 B bond trajectory etaPsi =
      ExpectedPsiActionIncrement chi0 B bond trajectory etaPsi := by
  rfl

theorem chi_action_increment_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3) :
    chiActionIncrement chi0 B bond trajectory etaChi =
      ExpectedChiActionIncrement chi0 B bond trajectory etaChi := by
  rfl

theorem psi_stationarity_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaPsi : Site → C3) :
    HasStationaryPsiAction chi0 B bond trajectory etaPsi ↔
      ExpectedStationaryPsiAction chi0 B bond trajectory etaPsi := by
  rfl

theorem chi_stationarity_definition_is_exact
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (etaChi : Site → C3) :
    HasStationaryChiAction chi0 B bond trajectory etaChi ↔
      ExpectedStationaryChiAction chi0 B bond trajectory etaChi := by
  rfl

end
end LeanDirac
