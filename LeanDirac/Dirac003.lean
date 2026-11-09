import LeanDirac.Dirac002

namespace LeanDirac

/-!
DIRAC-003 derives the two reciprocal weak Euler--Lagrange equation families
from the frozen DIRAC-002 action.  No time integrator or Dirac claim is made.
-/

noncomputable section
open BigOperators

abbrev AnalyticHasDerivAt (f : ℝ → ℝ) (f' x : ℝ) : Prop :=
  @HasDerivAt ℝ inferInstance ℝ
    Real.normedAddCommGroup.toAddCommGroup
    RCLike.toInnerProductSpaceReal.toModule
    inferInstance inferInstance f f' x

def c3RealPair (u v : C3) : ℝ :=
  ∑ component, (u component * star (v component)).re

def c3Add (u v : C3) : C3 := fun component => u component + v component

def c3ScaleReal (a : ℝ) (u : C3) : C3 :=
  fun component => (a : ℂ) * u component

def c3QuadraticLine (u v q : C3) (eps : ℝ) : C3 :=
  c3Add (c3Add u (c3ScaleReal eps v)) (c3ScaleReal (eps^2) q)

theorem density_quadratic_line_exact (u v q : C3) (eps : ℝ) :
    density (c3QuadraticLine u v q eps) =
      density u + 2 * eps * c3RealPair u v +
      eps^2 * (density v + 2 * c3RealPair u q) +
      2 * eps^3 * c3RealPair v q + eps^4 * density q := by
  simp [density, c3QuadraticLine, c3Add, c3ScaleReal, c3RealPair,
    Fin.sum_univ_succ, Complex.normSq]
  simp only [← Complex.ofReal_pow, Complex.ofReal_re, Complex.ofReal_im]
  ring

theorem density_quadratic_line_hasDerivAt (u v q : C3) :
    AnalyticHasDerivAt (fun eps : ℝ => density (c3QuadraticLine u v q eps))
      (2 * c3RealPair u v) 0 := by
  have hfun :
      (fun eps : ℝ => density (c3QuadraticLine u v q eps)) =
      (fun eps : ℝ =>
        density u + 2 * eps * c3RealPair u v +
        eps^2 * (density v + 2 * c3RealPair u q) +
        2 * eps^3 * c3RealPair v q + eps^4 * density q) := by
    funext eps
    exact density_quadratic_line_exact u v q eps
  rw [hfun]
  have hconst := hasDerivAt_const (𝕜 := ℝ) 0 (density u)
  have hlinear := (HasDerivAt.const_mul (2 : ℝ)
    (hasDerivAt_id (0 : ℝ))).mul_const (c3RealPair u v)
  have hquad := (hasDerivAt_pow 2 (0 : ℝ)).mul_const
    (density v + 2 * c3RealPair u q)
  have hcubic := (HasDerivAt.const_mul (2 : ℝ)
    (hasDerivAt_pow 3 (0 : ℝ))).mul_const (c3RealPair v q)
  have hquartic := (hasDerivAt_pow 4 (0 : ℝ)).mul_const (density q)
  have hsum := (((hconst.add hlinear).add hquad).add hcubic).add hquartic
  convert hsum using 1
  · funext eps
    simp only [Pi.add_apply, id_eq]
  · norm_num

def c3Zero : C3 := fun _ => 0

@[simp] theorem c3_scale_zero (a : ℝ) : c3ScaleReal a c3Zero = c3Zero := by
  funext component
  simp [c3ScaleReal, c3Zero]

@[simp] theorem c3_zero_scale (u : C3) : c3ScaleReal 0 u = c3Zero := by
  funext component
  simp [c3ScaleReal, c3Zero]

@[simp] theorem c3_add_zero_right (u : C3) : c3Add u c3Zero = u := by
  funext component
  simp [c3Add, c3Zero]

def c3Line (u v : C3) (eps : ℝ) : C3 :=
  c3Add u (c3ScaleReal eps v)

def stateLine (state direction : State Site) (eps : ℝ) : State Site where
  psi site := c3Line (state.psi site) (direction.psi site) eps
  chi site := c3Line (state.chi site) (direction.chi site) eps

def wedgeFirstVariation (state direction : State Site) (site : Site) : C3 :=
  c3Add
    (wedge (direction.chi site) (state.psi site))
    (wedge (state.chi site) (direction.psi site))

def wedgeSecondVariation (direction : State Site) (site : Site) : C3 :=
  wedge (direction.chi site) (direction.psi site)

theorem wedge_state_line_exact (state direction : State Site)
    (site : Site) (eps : ℝ) :
    wedge ((stateLine state direction eps).chi site)
      ((stateLine state direction eps).psi site) =
      c3QuadraticLine
        (wedge (state.chi site) (state.psi site))
        (wedgeFirstVariation state direction site)
        (wedgeSecondVariation direction site) eps := by
  funext component
  fin_cases component <;>
    simp [stateLine, c3Line, c3Add, c3ScaleReal, c3QuadraticLine,
      wedgeFirstVariation, wedgeSecondVariation, wedge] <;>
    ring

def bondAreaDifference (state : State Site) (edge : Bond Site) : C3 :=
  difference
    (wedge (state.chi edge.source) (state.psi edge.source))
    (wedge (state.chi edge.target) (state.psi edge.target))

def bondAreaFirstVariation (state direction : State Site)
    (edge : Bond Site) : C3 :=
  difference
    (wedgeFirstVariation state direction edge.source)
    (wedgeFirstVariation state direction edge.target)

def bondAreaSecondVariation (direction : State Site)
    (edge : Bond Site) : C3 :=
  difference
    (wedgeSecondVariation direction edge.source)
    (wedgeSecondVariation direction edge.target)

theorem bond_area_state_line_exact (state direction : State Site)
    (edge : Bond Site) (eps : ℝ) :
    bondAreaDifference (stateLine state direction eps) edge =
      c3QuadraticLine
        (bondAreaDifference state edge)
        (bondAreaFirstVariation state direction edge)
        (bondAreaSecondVariation direction edge) eps := by
  funext component
  simp only [bondAreaDifference, difference,
    wedge_state_line_exact, c3QuadraticLine, c3Add, c3ScaleReal,
    bondAreaFirstVariation, bondAreaSecondVariation]
  ring

def exteriorAreaBondFirstVariation (chi0 : ℝ)
    (state direction : State Site) (edge : Bond Site) : ℝ :=
  edge.weight / chi0^2 *
    c3RealPair (bondAreaDifference state edge)
      (bondAreaFirstVariation state direction edge)

theorem exterior_area_bond_hasDerivAt (chi0 : ℝ)
    (state direction : State Site) (edge : Bond Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => exteriorAreaBond chi0 (stateLine state direction eps) edge)
      (exteriorAreaBondFirstVariation chi0 state direction edge) 0 := by
  have h := HasDerivAt.const_mul (edge.weight / (2 * chi0^2))
    (density_quadratic_line_hasDerivAt
      (bondAreaDifference state edge)
      (bondAreaFirstVariation state direction edge)
      (bondAreaSecondVariation direction edge))
  convert h using 1
  · funext eps
    unfold exteriorAreaBond
    change edge.weight / (2 * chi0^2) *
      density (bondAreaDifference (stateLine state direction eps) edge) = _
    rw [bond_area_state_line_exact]
  · unfold exteriorAreaBondFirstVariation
    ring

def sitePotentialFirstVariation (chi0 : ℝ)
    (state direction : State Site) (site : Site) : ℝ :=
  let rhoPsi := density (state.psi site)
  let rhoChi := density (state.chi site)
  (rhoChi - chi0^2 + rhoPsi) *
      c3RealPair (state.chi site) (direction.chi site) +
    (rhoChi + rhoPsi) *
      c3RealPair (state.psi site) (direction.psi site)

theorem site_potential_hasDerivAt (chi0 : ℝ)
    (state direction : State Site) (site : Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => sitePotential chi0 (stateLine state direction eps) site)
      (sitePotentialFirstVariation chi0 state direction site) 0 := by
  have hChi := density_quadratic_line_hasDerivAt
    (state.chi site) (direction.chi site) c3Zero
  have hPsi := density_quadratic_line_hasDerivAt
    (state.psi site) (direction.psi site) c3Zero
  have hChi' : AnalyticHasDerivAt
      (fun eps : ℝ => density ((stateLine state direction eps).chi site))
      (2 * c3RealPair (state.chi site) (direction.chi site)) 0 := by
    convert hChi using 1
    funext eps
    simp [stateLine, c3Line, c3QuadraticLine]
  have hPsi' : AnalyticHasDerivAt
      (fun eps : ℝ => density ((stateLine state direction eps).psi site))
      (2 * c3RealPair (state.psi site) (direction.psi site)) 0 := by
    convert hPsi using 1
    funext eps
    simp [stateLine, c3Line, c3QuadraticLine]
  have h :=
    ((((hChi'.sub_const (chi0^2)).pow 2).div_const 4).add
      ((hChi'.mul hPsi').div_const 2)).add
      ((hPsi'.pow 2).div_const 4)
  convert h using 1
  · funext eps
    simp [sitePotential, Pi.add_apply, Pi.mul_apply, Pi.pow_apply]
  · unfold sitePotentialFirstVariation
    dsimp
    simp [stateLine, c3Line]
    ring

def potentialFirstVariation [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (state direction : State Site)
    (bond : Edge → Bond Site) : ℝ :=
  (∑ site, sitePotentialFirstVariation chi0 state direction site) +
    ∑ edge, exteriorAreaBondFirstVariation chi0 state direction (bond edge)

theorem potential_hasDerivAt [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (state direction : State Site)
    (bond : Edge → Bond Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => potential chi0 (stateLine state direction eps) bond)
      (potentialFirstVariation chi0 state direction bond) 0 := by
  have hsites := HasDerivAt.fun_sum
    (u := Finset.univ)
    (fun site _ => site_potential_hasDerivAt chi0 state direction site)
  have hedges := HasDerivAt.fun_sum
    (u := Finset.univ)
    (fun edge _ => exterior_area_bond_hasDerivAt
      chi0 state direction (bond edge))
  have h := hsites.add hedges
  convert h using 1
  · funext eps
    simp [potential]
  · rfl

def lagrangianStateFirstVariation [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (state direction : State Site)
    (bond : Edge → Bond Site) : ℝ :=
  -potentialFirstVariation chi0 state direction bond

theorem lagrangian_state_hasDerivAt [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state direction : State Site)
    (velocity : Velocity Site) (bond : Edge → Bond Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ =>
        lagrangian chi0 B (stateLine state direction eps) velocity bond)
      (lagrangianStateFirstVariation chi0 state direction bond) 0 := by
  have hkinetic := hasDerivAt_const (𝕜 := ℝ) 0 (kinetic B velocity)
  have hpotential := potential_hasDerivAt chi0 state direction bond
  have h := hkinetic.sub hpotential
  convert h using 1
  · funext eps
    rfl
  · simp [lagrangianStateFirstVariation]

def purePsiDirection (etaPsi : Site → C3) : State Site where
  psi := etaPsi
  chi := fun _ => c3Zero

def pureChiDirection (etaChi : Site → C3) : State Site where
  psi := fun _ => c3Zero
  chi := etaChi

def psiEulerLagrangeForce [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (state : State Site) (bond : Edge → Bond Site)
    (etaPsi : Site → C3) : ℝ :=
  lagrangianStateFirstVariation chi0 state
    (purePsiDirection etaPsi) bond

def chiEulerLagrangeForce [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (state : State Site) (bond : Edge → Bond Site)
    (etaChi : Site → C3) : ℝ :=
  lagrangianStateFirstVariation chi0 state
    (pureChiDirection etaChi) bond

theorem psi_force_is_full_action_variation
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) (etaPsi : Site → C3) :
    AnalyticHasDerivAt
      (fun eps : ℝ => lagrangian chi0 B
        (stateLine state (purePsiDirection etaPsi) eps) velocity bond)
      (psiEulerLagrangeForce chi0 state bond etaPsi) 0 := by
  exact lagrangian_state_hasDerivAt chi0 B state
    (purePsiDirection etaPsi) velocity bond

theorem chi_force_is_full_action_variation
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) (etaChi : Site → C3) :
    AnalyticHasDerivAt
      (fun eps : ℝ => lagrangian chi0 B
        (stateLine state (pureChiDirection etaChi) eps) velocity bond)
      (chiEulerLagrangeForce chi0 state bond etaChi) 0 := by
  exact lagrangian_state_hasDerivAt chi0 B state
    (pureChiDirection etaChi) velocity bond

def psiMomentumPair [Fintype Site]
    (velocity : Velocity Site) (etaPsi : Site → C3) : ℝ :=
  ∑ site, c3RealPair (velocity.psiDot site) (etaPsi site)

def chiMomentumPair [Fintype Site]
    (B : ℝ) (velocity : Velocity Site) (etaChi : Site → C3) : ℝ :=
  B * ∑ site, c3RealPair (velocity.chiDot site) (etaChi site)

def velocityLine (velocity direction : Velocity Site) (eps : ℝ) :
    Velocity Site where
  psiDot site := c3Line (velocity.psiDot site) (direction.psiDot site) eps
  chiDot site := c3Line (velocity.chiDot site) (direction.chiDot site) eps

def purePsiVelocity (etaPsi : Site → C3) : Velocity Site where
  psiDot := etaPsi
  chiDot := fun _ => c3Zero

def pureChiVelocity (etaChi : Site → C3) : Velocity Site where
  psiDot := fun _ => c3Zero
  chiDot := etaChi

theorem density_line_hasDerivAt (u v : C3) :
    AnalyticHasDerivAt (fun eps : ℝ => density (c3Line u v eps))
      (2 * c3RealPair u v) 0 := by
  convert density_quadratic_line_hasDerivAt u v c3Zero using 1
  funext eps
  simp [c3Line, c3QuadraticLine]

def kineticVelocityFirstVariation [Fintype Site]
    (B : ℝ) (velocity direction : Velocity Site) : ℝ :=
  psiMomentumPair velocity direction.psiDot +
    chiMomentumPair B velocity direction.chiDot

theorem kinetic_velocity_hasDerivAt [Fintype Site]
    (B : ℝ) (velocity direction : Velocity Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => kinetic B (velocityLine velocity direction eps))
      (kineticVelocityFirstVariation B velocity direction) 0 := by
  have hPsi := HasDerivAt.fun_sum
    (u := Finset.univ)
    (fun site _ => density_line_hasDerivAt
      (velocity.psiDot site) (direction.psiDot site))
  have hChi := HasDerivAt.fun_sum
    (u := Finset.univ)
    (fun site _ => density_line_hasDerivAt
      (velocity.chiDot site) (direction.chiDot site))
  have h := (hPsi.div_const 2).add
    ((HasDerivAt.const_mul B hChi).div_const 2)
  convert h using 1
  · funext eps
    simp [kinetic, velocityLine]
  · unfold kineticVelocityFirstVariation psiMomentumPair chiMomentumPair
    rw [← Finset.mul_sum, ← Finset.mul_sum]
    ring

theorem lagrangian_velocity_hasDerivAt
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity direction : Velocity Site)
    (bond : Edge → Bond Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ =>
        lagrangian chi0 B state (velocityLine velocity direction eps) bond)
      (kineticVelocityFirstVariation B velocity direction) 0 := by
  have hkinetic := kinetic_velocity_hasDerivAt B velocity direction
  have hpotential := hasDerivAt_const (𝕜 := ℝ) 0 (potential chi0 state bond)
  have h := hkinetic.sub hpotential
  convert h using 1
  · funext eps
    rfl
  · ring

theorem psi_momentum_is_full_action_velocity_variation
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) (etaPsi : Site → C3) :
    AnalyticHasDerivAt
      (fun eps : ℝ => lagrangian chi0 B state
        (velocityLine velocity (purePsiVelocity etaPsi) eps) bond)
      (psiMomentumPair velocity etaPsi) 0 := by
  convert lagrangian_velocity_hasDerivAt chi0 B state velocity
    (purePsiVelocity etaPsi) bond using 1
  simp [kineticVelocityFirstVariation, purePsiVelocity,
    psiMomentumPair, chiMomentumPair, c3RealPair, c3Zero]

theorem chi_momentum_is_full_action_velocity_variation
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) (etaChi : Site → C3) :
    AnalyticHasDerivAt
      (fun eps : ℝ => lagrangian chi0 B state
        (velocityLine velocity (pureChiVelocity etaChi) eps) bond)
      (chiMomentumPair B velocity etaChi) 0 := by
  convert lagrangian_velocity_hasDerivAt chi0 B state velocity
    (pureChiVelocity etaChi) bond using 1
  simp [kineticVelocityFirstVariation, pureChiVelocity,
    psiMomentumPair, chiMomentumPair, c3RealPair, c3Zero]

structure NativeTrajectory (Site : Type) where
  state : ℝ → State Site
  velocity : ℝ → Velocity Site

def HasStateVelocityAt (trajectory : NativeTrajectory Site) (t : ℝ) : Prop :=
  ∀ site component,
    AnalyticHasDerivAt
      (fun tau => ((trajectory.state tau).psi site component).re)
      ((trajectory.velocity t).psiDot site component).re t ∧
    AnalyticHasDerivAt
      (fun tau => ((trajectory.state tau).psi site component).im)
      ((trajectory.velocity t).psiDot site component).im t ∧
    AnalyticHasDerivAt
      (fun tau => ((trajectory.state tau).chi site component).re)
      ((trajectory.velocity t).chiDot site component).re t ∧
    AnalyticHasDerivAt
      (fun tau => ((trajectory.state tau).chi site component).im)
      ((trajectory.velocity t).chiDot site component).im t

def PsiEulerLagrangeEquationAt [Fintype Site] [Fintype Edge]
    (chi0 : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) : Prop :=
  ∀ etaPsi : Site → C3,
    AnalyticHasDerivAt
      (fun tau => psiMomentumPair (trajectory.velocity tau) etaPsi)
      (psiEulerLagrangeForce chi0 (trajectory.state t) bond etaPsi) t

def ChiEulerLagrangeEquationAt [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) : Prop :=
  ∀ etaChi : Site → C3,
    AnalyticHasDerivAt
      (fun tau => chiMomentumPair B (trajectory.velocity tau) etaChi)
      (chiEulerLagrangeForce chi0 (trajectory.state t) bond etaChi) t

structure TwoEulerLagrangeEquationFamiliesAt
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) : Prop where
  stateVelocity : HasStateVelocityAt trajectory t
  psiEquation : PsiEulerLagrangeEquationAt chi0 bond trajectory t
  chiEquation : ChiEulerLagrangeEquationAt chi0 B bond trajectory t

structure NativeActionStationaryAt
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) : Prop where
  stateVelocity : HasStateVelocityAt trajectory t
  psiFirstVariation : ∀ etaPsi : Site → C3,
    AnalyticHasDerivAt
      (fun tau => psiMomentumPair (trajectory.velocity tau) etaPsi)
      (lagrangianStateFirstVariation chi0 (trajectory.state t)
        (purePsiDirection etaPsi) bond) t
  chiFirstVariation : ∀ etaChi : Site → C3,
    AnalyticHasDerivAt
      (fun tau => chiMomentumPair B (trajectory.velocity tau) etaChi)
      (lagrangianStateFirstVariation chi0 (trajectory.state t)
        (pureChiDirection etaChi) bond) t

theorem native_action_stationarity_iff_two_equations
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (bond : Edge → Bond Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) :
    NativeActionStationaryAt chi0 B bond trajectory t ↔
      TwoEulerLagrangeEquationFamiliesAt chi0 B bond trajectory t := by
  constructor
  · intro stationary
    exact
      { stateVelocity := stationary.stateVelocity
        psiEquation := by
          intro etaPsi
          exact stationary.psiFirstVariation etaPsi
        chiEquation := by
          intro etaChi
          exact stationary.chiFirstVariation etaChi }
  · intro equations
    exact
      { stateVelocity := equations.stateVelocity
        psiFirstVariation := by
          intro etaPsi
          exact equations.psiEquation etaPsi
        chiFirstVariation := by
          intro etaChi
          exact equations.chiEquation etaChi }

theorem dirac003_two_reciprocal_action_variations
    [Fintype Site] [Fintype Edge]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (bond : Edge → Bond Site) :
    (∀ etaPsi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => lagrangian chi0 B
          (stateLine state (purePsiDirection etaPsi) eps) velocity bond)
        (psiEulerLagrangeForce chi0 state bond etaPsi) 0) ∧
    (∀ etaChi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => lagrangian chi0 B
          (stateLine state (pureChiDirection etaChi) eps) velocity bond)
        (chiEulerLagrangeForce chi0 state bond etaChi) 0) ∧
    (∀ etaPsi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => lagrangian chi0 B state
          (velocityLine velocity (purePsiVelocity etaPsi) eps) bond)
        (psiMomentumPair velocity etaPsi) 0) ∧
    (∀ etaChi : Site → C3,
      AnalyticHasDerivAt
        (fun eps : ℝ => lagrangian chi0 B state
          (velocityLine velocity (pureChiVelocity etaChi) eps) bond)
        (chiMomentumPair B velocity etaChi) 0) ∧
    state = ⟨state.psi, state.chi⟩ := by
  exact ⟨psi_force_is_full_action_variation chi0 B state velocity bond,
    chi_force_is_full_action_variation chi0 B state velocity bond,
    psi_momentum_is_full_action_velocity_variation
      chi0 B state velocity bond,
    chi_momentum_is_full_action_velocity_variation
      chi0 B state velocity bond,
    state_has_exactly_two_field_families state⟩

theorem dirac003_c19_two_reciprocal_action_variations
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
    state = ⟨state.psi, state.chi⟩ := by
  simpa [c19Lagrangian] using
    (dirac003_two_reciprocal_action_variations
      (Site := Site) (Edge := C19Edge Site)
      chi0 B state velocity (c19Bond geometry))

theorem dirac003_c19_stationarity_iff_two_equations
    [Fintype Site]
    (chi0 B : ℝ) (geometry : C19Lattice Site)
    (trajectory : NativeTrajectory Site) (t : ℝ) :
    NativeActionStationaryAt chi0 B (c19Bond geometry) trajectory t ↔
      TwoEulerLagrangeEquationFamiliesAt
        chi0 B (c19Bond geometry) trajectory t :=
  native_action_stationarity_iff_two_equations
    chi0 B (c19Bond geometry) trajectory t

theorem dirac003_periodic_c19_two_reciprocal_action_variations
    (L : ℕ) [NeZero L]
    (chi0 B : ℝ) (state : State (PeriodicCubicSite L))
    (velocity : Velocity (PeriodicCubicSite L)) :
    (∀ etaPsi : PeriodicCubicSite L → C3,
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
    state = ⟨state.psi, state.chi⟩ :=
  dirac003_c19_two_reciprocal_action_variations
    chi0 B state velocity (periodicC19Geometry L)

theorem dirac003_periodic_c19_stationarity_iff_two_equations
    (L : ℕ) [NeZero L] (chi0 B : ℝ)
    (trajectory : NativeTrajectory (PeriodicCubicSite L)) (t : ℝ) :
    NativeActionStationaryAt chi0 B
        (c19Bond (periodicC19Geometry L)) trajectory t ↔
      TwoEulerLagrangeEquationFamiliesAt chi0 B
        (c19Bond (periodicC19Geometry L)) trajectory t :=
  dirac003_c19_stationarity_iff_two_equations
    chi0 B (periodicC19Geometry L) trajectory t

end
end LeanDirac
