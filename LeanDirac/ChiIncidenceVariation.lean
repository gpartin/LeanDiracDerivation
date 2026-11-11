import LeanDirac.ChiIncidence
import LeanDirac.Dirac003Stabilized

namespace LeanDirac

set_option maxHeartbeats 4000000
set_option maxRecDepth 4000

/-!
The new incidence term is an explicit finite polynomial in the two native
fields.  This module proves differentiability of that actual polynomial and
uses its unique derivative as the additional reciprocal state variation.
No force is declared independently of the action.
-/

noncomputable section
open BigOperators Complex

def c3QuarticLine (u0 u1 u2 u3 u4 : C3) (eps : ℝ) : C3 :=
  fun component =>
    u0 component + (eps : ℂ) * u1 component +
      (eps^2 : ℝ) * u2 component + (eps^3 : ℝ) * u3 component +
      (eps^4 : ℝ) * u4 component

def densityQuarticRemainder (u0 u1 u2 u3 u4 : C3) (eps : ℝ) : ℝ :=
  density u1 + 2 * c3RealPair u0 u2 +
    eps * (2 * c3RealPair u0 u3 + 2 * c3RealPair u1 u2) +
    eps^2 * (density u2 + 2 * c3RealPair u0 u4 +
      2 * c3RealPair u1 u3) +
    eps^3 * (2 * c3RealPair u1 u4 + 2 * c3RealPair u2 u3) +
    eps^4 * (density u3 + 2 * c3RealPair u2 u4) +
    eps^5 * (2 * c3RealPair u3 u4) + eps^6 * density u4

theorem density_quartic_line_exact
    (u0 u1 u2 u3 u4 : C3) (eps : ℝ) :
    density (c3QuarticLine u0 u1 u2 u3 u4 eps) =
      density u0 + 2 * eps * c3RealPair u0 u1 +
        eps^2 * densityQuarticRemainder u0 u1 u2 u3 u4 eps := by
  simp [density, c3QuarticLine, c3RealPair, densityQuarticRemainder,
    Fin.sum_univ_succ, Complex.normSq]
  simp only [← Complex.ofReal_pow, Complex.ofReal_re, Complex.ofReal_im]
  ring_nf

theorem density_quartic_line_hasDerivAt
    (u0 u1 u2 u3 u4 : C3) :
    AnalyticHasDerivAt
      (fun eps : ℝ => density (c3QuarticLine u0 u1 u2 u3 u4 eps))
      (2 * c3RealPair u0 u1) 0 := by
  rw [show
      (fun eps : ℝ => density (c3QuarticLine u0 u1 u2 u3 u4 eps)) =
      (fun eps : ℝ => density u0 + 2 * eps * c3RealPair u0 u1 +
        eps^2 * densityQuarticRemainder u0 u1 u2 u3 u4 eps) by
        funext eps
        exact density_quartic_line_exact u0 u1 u2 u3 u4 eps]
  have hconst := hasDerivAt_const (𝕜 := ℝ) 0 (density u0)
  have hlinear := (HasDerivAt.const_mul (2 : ℝ)
    (hasDerivAt_id (0 : ℝ))).mul_const (c3RealPair u0 u1)
  have hremainder : DifferentiableAt ℝ
      (densityQuarticRemainder u0 u1 u2 u3 u4) 0 := by
    unfold densityQuarticRemainder
    fun_prop
  have hquadratic := (hasDerivAt_pow 2 (0 : ℝ)).mul
    hremainder.hasDerivAt
  convert (hconst.add hlinear).add hquadratic using 1
  · funext eps
    simp only [Pi.add_apply, Pi.mul_apply, id_eq]
  · norm_num

def incidenceOverlap0 (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (neighbor : C19Neighbor) : ℂ :=
  inner (state.chi site) (state.chi (geometry.neighbor site neighbor))

def incidenceOverlap1 (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site)
    (neighbor : C19Neighbor) : ℂ :=
  inner (direction.chi site) (state.chi (geometry.neighbor site neighbor)) +
    inner (state.chi site) (direction.chi (geometry.neighbor site neighbor))

def incidenceOverlap2 (direction : State Site)
    (geometry : C19Lattice Site) (site : Site)
    (neighbor : C19Neighbor) : ℂ :=
  inner (direction.chi site)
    (direction.chi (geometry.neighbor site neighbor))

def incidenceTrack0 (state : State Site) (geometry : C19Lattice Site)
    (site : Site) (neighbor : C19Neighbor) : C3 :=
  wedge (state.chi (geometry.neighbor site neighbor))
    (state.psi (geometry.neighbor site neighbor))

def incidenceTrack1 (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site)
    (neighbor : C19Neighbor) : C3 :=
  wedgeFirstVariation state direction (geometry.neighbor site neighbor)

def incidenceTrack2 (direction : State Site)
    (geometry : C19Lattice Site) (site : Site)
    (neighbor : C19Neighbor) : C3 :=
  wedgeSecondVariation direction (geometry.neighbor site neighbor)

theorem incidence_overlap_state_line_exact
    (state direction : State Site) (geometry : C19Lattice Site)
    (site : Site) (neighbor : C19Neighbor) (eps : ℝ) :
    inner
        ((stateLine state direction eps).chi site)
        ((stateLine state direction eps).chi
          (geometry.neighbor site neighbor)) =
      incidenceOverlap0 state geometry site neighbor +
        (eps : ℂ) * incidenceOverlap1 state direction geometry site neighbor +
        ((eps^2 : ℝ) : ℂ) *
          incidenceOverlap2 direction geometry site neighbor := by
  simp [incidenceOverlap0, incidenceOverlap1, incidenceOverlap2,
    stateLine, c3Line, c3Add, c3ScaleReal, inner,
    Finset.mul_sum]
  rw [mul_add, Finset.mul_sum, Finset.mul_sum]
  repeat' rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro component _
  ring

def incidenceCoefficient (chi0 : ℝ) (neighbor : C19Neighbor) : ℂ :=
  ((3 * c19FaceIncidenceSign neighbor) / (2 * chi0^2) : ℝ)

def incidenceResidualCoefficient0 [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : C3 := fun component =>
  ∑ neighbor, incidenceCoefficient chi0 neighbor *
    incidenceOverlap0 state geometry site neighbor *
      incidenceTrack0 state geometry site neighbor component

def incidenceResidualCoefficient1 [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) : C3 := fun component =>
  ∑ neighbor, incidenceCoefficient chi0 neighbor *
    (incidenceOverlap1 state direction geometry site neighbor *
        incidenceTrack0 state geometry site neighbor component +
      incidenceOverlap0 state geometry site neighbor *
        incidenceTrack1 state direction geometry site neighbor component)

def incidenceResidualCoefficient2 [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) : C3 := fun component =>
  ∑ neighbor, incidenceCoefficient chi0 neighbor *
    (incidenceOverlap2 direction geometry site neighbor *
        incidenceTrack0 state geometry site neighbor component +
      incidenceOverlap1 state direction geometry site neighbor *
        incidenceTrack1 state direction geometry site neighbor component +
      incidenceOverlap0 state geometry site neighbor *
        incidenceTrack2 direction geometry site neighbor component)

def incidenceResidualCoefficient3 [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) : C3 := fun component =>
  ∑ neighbor, incidenceCoefficient chi0 neighbor *
    (incidenceOverlap2 direction geometry site neighbor *
        incidenceTrack1 state direction geometry site neighbor component +
      incidenceOverlap1 state direction geometry site neighbor *
        incidenceTrack2 direction geometry site neighbor component)

def incidenceResidualCoefficient4 [Fintype Site]
    (chi0 : ℝ) (direction : State Site)
    (geometry : C19Lattice Site) (site : Site) : C3 := fun component =>
  ∑ neighbor, incidenceCoefficient chi0 neighbor *
    incidenceOverlap2 direction geometry site neighbor *
      incidenceTrack2 direction geometry site neighbor component

theorem chi_incidence_residual_state_line_exact [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) (site : Site) (eps : ℝ) :
    chiIncidenceResidual chi0 (stateLine state direction eps) geometry site =
      c3QuarticLine
        (incidenceResidualCoefficient0 chi0 state geometry site)
        (incidenceResidualCoefficient1 chi0 state direction geometry site)
        (incidenceResidualCoefficient2 chi0 state direction geometry site)
        (incidenceResidualCoefficient3 chi0 state direction geometry site)
        (incidenceResidualCoefficient4 chi0 direction geometry site) eps := by
  funext component
  unfold chiIncidenceResidual
  simp_rw [incidence_overlap_state_line_exact]
  simp_rw [wedge_state_line_exact]
  simp [incidenceCoefficient,
    incidenceResidualCoefficient0, incidenceResidualCoefficient1,
    incidenceResidualCoefficient2, incidenceResidualCoefficient3,
    incidenceResidualCoefficient4, incidenceOverlap0, incidenceOverlap1,
    incidenceOverlap2, incidenceTrack0, incidenceTrack1, incidenceTrack2,
    c3QuarticLine, c3QuadraticLine, c3Add, c3ScaleReal,
    wedgeFirstVariation, wedgeSecondVariation,
    Finset.mul_sum]
  repeat' rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro neighbor _
  ring_nf

def chiIncidenceEnergyFirstVariation [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) : ℝ :=
  ∑ site, c3RealPair
    (incidenceResidualCoefficient0 chi0 state geometry site)
    (incidenceResidualCoefficient1 chi0 state direction geometry site)

theorem chi_incidence_energy_differentiableAt [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) :
    DifferentiableAt ℝ (fun eps : ℝ =>
      chiIncidenceEnergy chi0 (stateLine state direction eps) geometry) 0 := by
  unfold chiIncidenceEnergy chiIncidenceSiteEnergy
  rw [show (fun eps : ℝ =>
      ∑ site,
        c3EuclideanSq
          (chiIncidenceResidual chi0 (stateLine state direction eps)
            geometry site) / 2) =
      (fun eps : ℝ =>
        ∑ site,
          density
            (c3QuarticLine
              (incidenceResidualCoefficient0 chi0 state geometry site)
              (incidenceResidualCoefficient1 chi0 state direction geometry site)
              (incidenceResidualCoefficient2 chi0 state direction geometry site)
              (incidenceResidualCoefficient3 chi0 state direction geometry site)
              (incidenceResidualCoefficient4 chi0 direction geometry site) eps) / 2) by
        funext eps
        apply Finset.sum_congr rfl
        intro site _
        rw [c3_euclidean_sq_eq_density,
          chi_incidence_residual_state_line_exact]]
  exact (HasDerivAt.fun_sum (u := Finset.univ) (fun site _ =>
    (density_quartic_line_hasDerivAt
      (incidenceResidualCoefficient0 chi0 state geometry site)
      (incidenceResidualCoefficient1 chi0 state direction geometry site)
      (incidenceResidualCoefficient2 chi0 state direction geometry site)
      (incidenceResidualCoefficient3 chi0 state direction geometry site)
      (incidenceResidualCoefficient4 chi0 direction geometry site)).div_const 2)).differentiableAt

theorem chi_incidence_energy_hasDerivAt [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ =>
        chiIncidenceEnergy chi0 (stateLine state direction eps) geometry)
      (chiIncidenceEnergyFirstVariation chi0 state direction geometry) 0 := by
  unfold chiIncidenceEnergy chiIncidenceSiteEnergy
  rw [show (fun eps : ℝ =>
      ∑ site,
        c3EuclideanSq
          (chiIncidenceResidual chi0 (stateLine state direction eps)
            geometry site) / 2) =
      (fun eps : ℝ =>
        ∑ site,
          density
            (c3QuarticLine
              (incidenceResidualCoefficient0 chi0 state geometry site)
              (incidenceResidualCoefficient1 chi0 state direction geometry site)
              (incidenceResidualCoefficient2 chi0 state direction geometry site)
              (incidenceResidualCoefficient3 chi0 state direction geometry site)
              (incidenceResidualCoefficient4 chi0 direction geometry site) eps) / 2) by
        funext eps
        apply Finset.sum_congr rfl
        intro site _
        rw [c3_euclidean_sq_eq_density,
          chi_incidence_residual_state_line_exact]]
  have h := HasDerivAt.fun_sum (u := Finset.univ) (fun site _ =>
    (density_quartic_line_hasDerivAt
      (incidenceResidualCoefficient0 chi0 state geometry site)
      (incidenceResidualCoefficient1 chi0 state direction geometry site)
      (incidenceResidualCoefficient2 chi0 state direction geometry site)
      (incidenceResidualCoefficient3 chi0 state direction geometry site)
      (incidenceResidualCoefficient4 chi0 direction geometry site)).div_const 2)
  simpa [chiIncidenceEnergyFirstVariation] using h

def incidenceC19LagrangianStateFirstVariation [Fintype Site]
    (chi0 : ℝ) (state direction : State Site)
    (geometry : C19Lattice Site) : ℝ :=
  stabilizedC19LagrangianStateFirstVariation chi0 state direction geometry -
    chiIncidenceEnergyFirstVariation chi0 state direction geometry

theorem incidence_c19_lagrangian_state_hasDerivAt [Fintype Site]
    (chi0 B : ℝ) (state direction : State Site)
    (velocity : Velocity Site) (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => incidenceC19Lagrangian chi0 B
        (stateLine state direction eps) velocity geometry)
      (incidenceC19LagrangianStateFirstVariation
        chi0 state direction geometry) 0 := by
  have hbase := stabilized_c19_lagrangian_state_hasDerivAt
    chi0 B state direction velocity geometry
  have hincidence := chi_incidence_energy_hasDerivAt
    chi0 state direction geometry
  have h := hbase.sub hincidence
  convert h using 1
  · funext eps
    simp [incidenceC19Lagrangian, incidenceC19Potential,
      stabilizedC19Lagrangian]
    ring
  · rfl

theorem incidence_c19_lagrangian_velocity_hasDerivAt [Fintype Site]
    (chi0 B : ℝ) (state : State Site)
    (velocity direction : Velocity Site) (geometry : C19Lattice Site) :
    AnalyticHasDerivAt
      (fun eps : ℝ => incidenceC19Lagrangian chi0 B state
        (velocityLine velocity direction eps) geometry)
      (kineticVelocityFirstVariation B velocity direction) 0 := by
  have hkinetic := kinetic_velocity_hasDerivAt B velocity direction
  have hpotential := hasDerivAt_const (𝕜 := ℝ) 0
    (incidenceC19Potential chi0 state geometry)
  have h := hkinetic.sub hpotential
  convert h using 1
  · funext eps
    rfl
  · simp

structure IncidenceLocalTwoFieldVariationCertificate [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : Prop where
  psiState : ∀ etaPsi : Site → C3,
    AnalyticHasDerivAt
      (fun eps : ℝ => incidenceC19Lagrangian chi0 B
        (stateLine state (purePsiDirection etaPsi) eps) velocity geometry)
      (incidenceC19LagrangianStateFirstVariation chi0 state
        (purePsiDirection etaPsi) geometry) 0
  chiState : ∀ etaChi : Site → C3,
    AnalyticHasDerivAt
      (fun eps : ℝ => incidenceC19Lagrangian chi0 B
        (stateLine state (pureChiDirection etaChi) eps) velocity geometry)
      (incidenceC19LagrangianStateFirstVariation chi0 state
        (pureChiDirection etaChi) geometry) 0
  velocity : ∀ direction : Velocity Site,
    AnalyticHasDerivAt
      (fun eps : ℝ => incidenceC19Lagrangian chi0 B state
        (velocityLine velocity direction eps) geometry)
      (kineticVelocityFirstVariation B velocity direction) 0

theorem incidence_c19_variations_from_actual_action [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) :
    IncidenceLocalTwoFieldVariationCertificate
      chi0 B state velocity geometry where
  psiState := fun etaPsi =>
    incidence_c19_lagrangian_state_hasDerivAt chi0 B state
      (purePsiDirection etaPsi) velocity geometry
  chiState := fun etaChi =>
    incidence_c19_lagrangian_state_hasDerivAt chi0 B state
      (pureChiDirection etaChi) velocity geometry
  velocity := fun direction =>
    incidence_c19_lagrangian_velocity_hasDerivAt
      chi0 B state velocity direction geometry

end
end LeanDirac
