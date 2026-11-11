import LeanDirac.ShapeStability

namespace LeanDirac

/-!
# Native complex-chi cubical incidence energy

The first six C19 neighbors are the two oriented faces in each of the three
coordinate directions.  Their forward/backward incidence sign, multiplied by
the native complex-chi overlap, transports the composite oriented area
`chi wedge psi`.  The only new energy is the positive norm square of that
native residual.  It introduces no spinor, matrix, projector, or independent
edge field.
-/

noncomputable section
open BigOperators Complex

def c19FaceIncidenceSign (neighbor : C19Neighbor) : ℝ :=
  match neighbor.val with
  | 0 => 1 | 1 => -1 | 2 => 1 | 3 => -1 | 4 => 1 | 5 => -1
  | _ => 0

theorem c19_face_incidence_reverse (neighbor : C19Neighbor) :
    c19FaceIncidenceSign (c19Reverse neighbor) =
      -c19FaceIncidenceSign neighbor := by
  fin_cases neighbor <;> norm_num [c19FaceIncidenceSign, c19Reverse]

def chiIncidenceResidual
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : C3 :=
  fun component =>
    ∑ neighbor : C19Neighbor,
      (((3 * c19FaceIncidenceSign neighbor) / (2 * chi0^2) : ℝ) : ℂ) *
        inner (state.chi site)
          (state.chi (geometry.neighbor site neighbor)) *
        wedge
          (state.chi (geometry.neighbor site neighbor))
          (state.psi (geometry.neighbor site neighbor)) component

/-!
The scalar component symbol of the same residual with an explicit neighbour
phase.  This is not a new energy or force.  It is the exact Bloch reduction
of one component of `chiIncidenceResidual`; setting every phase to one and
using the corresponding component of `chi wedge Psi` recovers the native
residual definition below.
-/
def chiIncidenceComponentSymbol
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (phase : Site → C19Neighbor → ℂ) (amplitude : Site → ℂ)
    (site : Site) : ℂ :=
  ∑ neighbor : C19Neighbor,
    (((3 * c19FaceIncidenceSign neighbor) / (2 * chi0^2) : ℝ) : ℂ) *
      inner (state.chi site)
        (state.chi (geometry.neighbor site neighbor)) *
      phase site neighbor * amplitude (geometry.neighbor site neighbor)

theorem chi_incidence_component_symbol_is_native_residual
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (component : Fin 3) (site : Site) :
    chiIncidenceComponentSymbol chi0 state geometry
        (fun _ _ => 1)
        (fun target => wedge (state.chi target) (state.psi target) component)
        site =
      chiIncidenceResidual chi0 state geometry site component := by
  unfold chiIncidenceComponentSymbol chiIncidenceResidual
  apply Finset.sum_congr rfl
  intro neighbor _
  ring

def chiIncidenceSiteEnergy [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) : ℝ :=
  c3EuclideanSq (chiIncidenceResidual chi0 state geometry site) / 2

def chiIncidenceEnergy [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) : ℝ :=
  ∑ site, chiIncidenceSiteEnergy chi0 state geometry site

def incidenceC19Potential [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) : ℝ :=
  stabilizedC19Potential chi0 state geometry +
    chiIncidenceEnergy chi0 state geometry

def incidenceC19Lagrangian [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : ℝ :=
  kinetic B velocity - incidenceC19Potential chi0 state geometry

def incidenceC19Hamiltonian [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) : ℝ :=
  kinetic B velocity + incidenceC19Potential chi0 state geometry

theorem chi_incidence_site_energy_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (site : Site) :
    0 ≤ chiIncidenceSiteEnergy chi0 state geometry site := by
  unfold chiIncidenceSiteEnergy
  rw [c3_euclidean_sq_eq_density]
  exact div_nonneg (density_nonnegative _) (by norm_num)

theorem chi_incidence_energy_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site) :
    0 ≤ chiIncidenceEnergy chi0 state geometry := by
  unfold chiIncidenceEnergy
  exact Finset.sum_nonneg (fun site _ =>
    chi_incidence_site_energy_nonnegative chi0 state geometry site)

theorem incidence_c19_potential_nonnegative [Fintype Site]
    (chi0 : ℝ) (state : State Site) (geometry : C19Lattice Site)
    (hchi : 0 < chi0) :
    0 ≤ incidenceC19Potential chi0 state geometry := by
  exact add_nonneg
    (stabilized_c19_potential_nonnegative chi0 state geometry hchi)
    (chi_incidence_energy_nonnegative chi0 state geometry)

theorem incidence_c19_hamiltonian_nonnegative [Fintype Site]
    (chi0 B : ℝ) (state : State Site) (velocity : Velocity Site)
    (geometry : C19Lattice Site) (hchi : 0 < chi0) (hB : 0 ≤ B) :
    0 ≤ incidenceC19Hamiltonian chi0 B state velocity geometry := by
  exact add_nonneg (kinetic_nonnegative B velocity hB)
    (incidence_c19_potential_nonnegative chi0 state geometry hchi)

theorem chi_incidence_coefficient_fixed_by_tight_frame
    (coefficient overlapMagnitude : ℝ)
    (hoverlap : overlapMagnitude = 1 / 3)
    (hnormalize : coefficient * overlapMagnitude = 1) :
    coefficient = 3 := by
  rw [hoverlap] at hnormalize
  linarith

end
end LeanDirac
