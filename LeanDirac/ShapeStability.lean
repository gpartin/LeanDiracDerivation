import LeanDirac.NativeAction

namespace LeanDirac

/-!
The action-level DIRAC-009 stability correction.

For each site, complex `chi` builds the Hermitian covariance of its complete
C19 shell.  The positive square below penalizes failure of the local `chi`
direction to be an eigenline of that covariance.  It contains no prepared
ray, corner label, spinor matrix, projector, or runtime constraint.

The relative C19 weights sum to 24.  On the prepared cubic frame the relevant
covariance eigenvalue is 8, so the dimensionless normalization is `1 / 8`.
The overall coefficient is the unit coefficient of the norm square; it is not
fitted to a Dirac residual or a lifetime target.
-/

noncomputable section
open BigOperators Complex

def chiCovarianceAction [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : C3 :=
  fun component =>
    ∑ neighbor : C19Neighbor,
      ((c19Weight neighbor / (8 * chi0^2) : ℝ) : ℂ) *
        state.chi (geometry.neighbor site neighbor) component *
        inner (state.chi (geometry.neighbor site neighbor)) (state.chi site)

def chiCovarianceShapeResidual [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : C3 :=
  wedge (state.chi site) (chiCovarianceAction chi0 state geometry site)

def c3EuclideanSq (v : C3) : ℝ :=
  ∑ component,
    ((Complex.reCLM (v component))^2 + (Complex.imCLM (v component))^2)

theorem c3_euclidean_sq_eq_density (v : C3) :
    c3EuclideanSq v = density v := by
  simp [c3EuclideanSq, density, Complex.normSq_apply, pow_two]

def chiCovarianceShapeSiteEnergy [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : ℝ :=
  c3EuclideanSq (chiCovarianceShapeResidual chi0 state geometry site) / 2

def chiCovarianceShapeEnergy [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) : ℝ :=
  ∑ site, chiCovarianceShapeSiteEnergy chi0 state geometry site

def stabilizedC19Potential [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) : ℝ :=
  c19Potential chi0 state geometry +
    chiCovarianceShapeEnergy chi0 state geometry

def stabilizedC19Lagrangian [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : ℝ :=
  kinetic B velocity - stabilizedC19Potential chi0 state geometry

def stabilizedC19Hamiltonian [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : ℝ :=
  kinetic B velocity + stabilizedC19Potential chi0 state geometry

theorem chi_covariance_shape_site_energy_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) :
    0 ≤ chiCovarianceShapeSiteEnergy chi0 state geometry site := by
  unfold chiCovarianceShapeSiteEnergy
  rw [c3_euclidean_sq_eq_density]
  exact div_nonneg (density_nonnegative _) (by norm_num)

theorem chi_covariance_shape_energy_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    0 ≤ chiCovarianceShapeEnergy chi0 state geometry := by
  unfold chiCovarianceShapeEnergy
  exact Finset.sum_nonneg (fun site _ =>
    chi_covariance_shape_site_energy_nonnegative chi0 state geometry site)

theorem stabilized_c19_potential_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hchi : 0 < chi0) :
    0 ≤ stabilizedC19Potential chi0 state geometry := by
  exact add_nonneg
    (potential_nonnegative chi0 state (c19Bond geometry) hchi
      (fun edge => le_of_lt (c19_bond_weight_positive geometry edge)))
    (chi_covariance_shape_energy_nonnegative chi0 state geometry)

theorem stabilized_c19_hamiltonian_nonnegative [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) (hchi : 0 < chi0) (hB : 0 ≤ B) :
    0 ≤ stabilizedC19Hamiltonian chi0 B state velocity geometry := by
  exact add_nonneg (kinetic_nonnegative B velocity hB)
    (stabilized_c19_potential_nonnegative chi0 state geometry hchi)

theorem covariance_track_normalization_unique
    (normalization : ℝ) (htrack : 8 * normalization = 1) :
    normalization = 1 / 8 := by
  linarith

theorem relative_to_canonical_c19_covariance_prefactor :
    (6 : ℝ) / 8 = 3 / 4 := by
  norm_num

end
end LeanDirac
