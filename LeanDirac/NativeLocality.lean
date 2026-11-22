/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.Dirac004DiscreteAction

@[expose] public section

namespace LeanDirac

/-! Locality of both native forces and of the simultaneous two-layer update. -/

noncomputable section
open Complex

/-- A directed C19 bond is incident to `site` when `site` is either endpoint. -/
def C19IncidentAt (geometry : C19Lattice Site)
    (site : Site) (edge : C19Edge Site) : Prop :=
  edge.1 = site ∨ geometry.neighbor edge.1 edge.2 = site

/-- A bond variation vanishes whenever both native field directions vanish
at both endpoints. -/
theorem exterior_area_bond_variation_zero_off_direction_support
    (chi0 : ℝ) (state direction : State Site) (edge : Bond Site)
    (hPsiSource : direction.psi edge.source = c3Zero)
    (hPsiTarget : direction.psi edge.target = c3Zero)
    (hChiSource : direction.chi edge.source = c3Zero)
    (hChiTarget : direction.chi edge.target = c3Zero) :
    exteriorAreaBondFirstVariation chi0 state direction edge = 0 := by
  simp [exteriorAreaBondFirstVariation, bondAreaFirstVariation,
    wedgeFirstVariation, hPsiSource, hPsiTarget, hChiSource,
    hChiTarget, c3RealPair, c3Add, c3Zero, difference, wedge]

/-- A nonincident C19 bond contributes no real-coordinate psi force at the
selected site. -/
theorem nonincident_c19_bond_has_zero_psi_real_variation
    [DecidableEq Site] (chi0 : ℝ) (state : State Site)
    (geometry : C19Lattice Site) (edge : C19Edge Site)
    (site : Site) (component : Fin 3)
    (hnot : ¬ C19IncidentAt geometry site edge) :
    exteriorAreaBondFirstVariation chi0 state
        (purePsiDirection (realCoordinateDirection site component))
        (c19Bond geometry edge) = 0 := by
  have hsource : edge.1 ≠ site := by
    intro h
    exact hnot (Or.inl h)
  have htarget : geometry.neighbor edge.1 edge.2 ≠ site := by
    intro h
    exact hnot (Or.inr h)
  apply exterior_area_bond_variation_zero_off_direction_support
  · funext fieldComponent
    change realCoordinateDirection site component edge.1 fieldComponent = 0
    simp [realCoordinateDirection, hsource]
  · funext fieldComponent
    change realCoordinateDirection site component
      (geometry.neighbor edge.1 edge.2) fieldComponent = 0
    simp [realCoordinateDirection, htarget]
  · rfl
  · rfl

/-- A nonincident C19 bond contributes no imaginary-coordinate psi force. -/
theorem nonincident_c19_bond_has_zero_psi_imag_variation
    [DecidableEq Site] (chi0 : ℝ) (state : State Site)
    (geometry : C19Lattice Site) (edge : C19Edge Site)
    (site : Site) (component : Fin 3)
    (hnot : ¬ C19IncidentAt geometry site edge) :
    exteriorAreaBondFirstVariation chi0 state
        (purePsiDirection (imagCoordinateDirection site component))
        (c19Bond geometry edge) = 0 := by
  have hsource : edge.1 ≠ site := fun h => hnot (Or.inl h)
  have htarget : geometry.neighbor edge.1 edge.2 ≠ site :=
    fun h => hnot (Or.inr h)
  apply exterior_area_bond_variation_zero_off_direction_support
  · funext fieldComponent
    change imagCoordinateDirection site component edge.1 fieldComponent = 0
    simp [imagCoordinateDirection, hsource]
  · funext fieldComponent
    change imagCoordinateDirection site component
      (geometry.neighbor edge.1 edge.2) fieldComponent = 0
    simp [imagCoordinateDirection, htarget]
  · rfl
  · rfl

/-- A nonincident C19 bond contributes no real-coordinate chi force. -/
theorem nonincident_c19_bond_has_zero_chi_real_variation
    [DecidableEq Site] (chi0 : ℝ) (state : State Site)
    (geometry : C19Lattice Site) (edge : C19Edge Site)
    (site : Site) (component : Fin 3)
    (hnot : ¬ C19IncidentAt geometry site edge) :
    exteriorAreaBondFirstVariation chi0 state
        (pureChiDirection (realCoordinateDirection site component))
        (c19Bond geometry edge) = 0 := by
  have hsource : edge.1 ≠ site := fun h => hnot (Or.inl h)
  have htarget : geometry.neighbor edge.1 edge.2 ≠ site :=
    fun h => hnot (Or.inr h)
  apply exterior_area_bond_variation_zero_off_direction_support
  · rfl
  · rfl
  · funext fieldComponent
    change realCoordinateDirection site component edge.1 fieldComponent = 0
    simp [realCoordinateDirection, hsource]
  · funext fieldComponent
    change realCoordinateDirection site component
      (geometry.neighbor edge.1 edge.2) fieldComponent = 0
    simp [realCoordinateDirection, htarget]

/-- A nonincident C19 bond contributes no imaginary-coordinate chi force. -/
theorem nonincident_c19_bond_has_zero_chi_imag_variation
    [DecidableEq Site] (chi0 : ℝ) (state : State Site)
    (geometry : C19Lattice Site) (edge : C19Edge Site)
    (site : Site) (component : Fin 3)
    (hnot : ¬ C19IncidentAt geometry site edge) :
    exteriorAreaBondFirstVariation chi0 state
        (pureChiDirection (imagCoordinateDirection site component))
        (c19Bond geometry edge) = 0 := by
  have hsource : edge.1 ≠ site := fun h => hnot (Or.inl h)
  have htarget : geometry.neighbor edge.1 edge.2 ≠ site :=
    fun h => hnot (Or.inr h)
  apply exterior_area_bond_variation_zero_off_direction_support
  · rfl
  · rfl
  · funext fieldComponent
    change imagCoordinateDirection site component edge.1 fieldComponent = 0
    simp [imagCoordinateDirection, hsource]
  · funext fieldComponent
    change imagCoordinateDirection site component
      (geometry.neighbor edge.1 edge.2) fieldComponent = 0
    simp [imagCoordinateDirection, htarget]

/-- If a directed C19 bond ends at `site`, its source is the reverse C19
neighbor of `site`. -/
theorem c19_incident_target_source_is_reverse_neighbor
    (geometry : C19Lattice Site) (edge : C19Edge Site) (site : Site)
    (htarget : geometry.neighbor edge.1 edge.2 = site) :
    edge.1 = geometry.neighbor site (c19Reverse edge.2) := by
  rw [← htarget]
  symm
  exact geometry.neighbor_reverse edge.1 edge.2

/-- Two native states agree on the center and all eighteen C19 neighbors of
one selected site. -/
structure C19LocalAgreement (geometry : C19Lattice Site)
    (left right : State Site) (site : Site) : Prop where
  centerPsi : left.psi site = right.psi site
  centerChi : left.chi site = right.chi site
  neighborPsi : ∀ direction,
    left.psi (geometry.neighbor site direction) =
      right.psi (geometry.neighbor site direction)
  neighborChi : ∀ direction,
    left.chi (geometry.neighbor site direction) =
      right.chi (geometry.neighbor site direction)

/-- A state variation is supported at exactly one selected site. -/
def StateDirectionSupportedAt (direction : State Site) (site : Site) : Prop :=
  ∀ other, other ≠ site →
    direction.psi other = c3Zero ∧ direction.chi other = c3Zero

theorem site_potential_first_variation_zero_of_direction_zero
    (chi0 : ℝ) (state direction : State Site) (site : Site)
    (hPsi : direction.psi site = c3Zero)
    (hChi : direction.chi site = c3Zero) :
    sitePotentialFirstVariation chi0 state direction site = 0 := by
  simp [sitePotentialFirstVariation, hPsi, hChi, c3RealPair, c3Zero]

theorem site_potential_first_variation_congr
    (chi0 : ℝ) (left right direction : State Site) (site : Site)
    (hPsi : left.psi site = right.psi site)
    (hChi : left.chi site = right.chi site) :
    sitePotentialFirstVariation chi0 left direction site =
      sitePotentialFirstVariation chi0 right direction site := by
  simp [sitePotentialFirstVariation, hPsi, hChi]

theorem exterior_area_bond_first_variation_congr
    (chi0 : ℝ) (left right direction : State Site) (edge : Bond Site)
    (hPsiSource : left.psi edge.source = right.psi edge.source)
    (hPsiTarget : left.psi edge.target = right.psi edge.target)
    (hChiSource : left.chi edge.source = right.chi edge.source)
    (hChiTarget : left.chi edge.target = right.chi edge.target) :
    exteriorAreaBondFirstVariation chi0 left direction edge =
      exteriorAreaBondFirstVariation chi0 right direction edge := by
  simp [exteriorAreaBondFirstVariation, bondAreaDifference,
    bondAreaFirstVariation, wedgeFirstVariation,
    hPsiSource, hPsiTarget, hChiSource, hChiTarget]

/-- Every endpoint of a C19 bond incident to the selected site lies in the
center-or-neighbor region on which local agreement is assumed. -/
theorem c19_incident_endpoints_agree
    (geometry : C19Lattice Site) (left right : State Site) (site : Site)
    (agreement : C19LocalAgreement geometry left right site)
    (edge : C19Edge Site) (hincident : C19IncidentAt geometry site edge) :
    left.psi edge.1 = right.psi edge.1 ∧
      left.psi (geometry.neighbor edge.1 edge.2) =
        right.psi (geometry.neighbor edge.1 edge.2) ∧
      left.chi edge.1 = right.chi edge.1 ∧
      left.chi (geometry.neighbor edge.1 edge.2) =
        right.chi (geometry.neighbor edge.1 edge.2) := by
  rcases edge with ⟨source, direction⟩
  unfold C19IncidentAt at hincident
  rcases hincident with hsource | htarget
  · change source = site at hsource
    subst source
    exact ⟨agreement.centerPsi, agreement.neighborPsi direction,
      agreement.centerChi, agreement.neighborChi direction⟩
  · change geometry.neighbor source direction = site at htarget
    have hsource :
        source = geometry.neighbor site (c19Reverse direction) := by
      rw [← htarget]
      symm
      exact geometry.neighbor_reverse source direction
    constructor
    · simpa [hsource] using agreement.neighborPsi (c19Reverse direction)
    constructor
    · simpa [htarget] using agreement.centerPsi
    constructor
    · simpa [hsource] using agreement.neighborChi (c19Reverse direction)
    · simpa [htarget] using agreement.centerChi

/-- The complete native Lagrangian variation in a direction supported at one
site depends only on the two fields at that site and its C19 neighbors.  This
is a theorem about the assembled site sum and assembled bond sum, not merely
about one representative bond. -/
theorem c19_full_variation_is_local
    [Fintype Site] [DecidableEq Site]
    (chi0 : ℝ) (geometry : C19Lattice Site)
    (left right direction : State Site) (site : Site)
    (agreement : C19LocalAgreement geometry left right site)
    (supported : StateDirectionSupportedAt direction site) :
    lagrangianStateFirstVariation chi0 left direction (c19Bond geometry) =
      lagrangianStateFirstVariation chi0 right direction
        (c19Bond geometry) := by
  classical
  unfold lagrangianStateFirstVariation potentialFirstVariation
  congr 2
  · apply Finset.sum_congr rfl
    intro other _
    by_cases hsite : other = site
    · subst other
      exact site_potential_first_variation_congr chi0 left right direction site
        agreement.centerPsi agreement.centerChi
    · rcases supported other hsite with ⟨hPsi, hChi⟩
      calc
        sitePotentialFirstVariation chi0 left direction other = 0 :=
          site_potential_first_variation_zero_of_direction_zero
            chi0 left direction other hPsi hChi
        _ = sitePotentialFirstVariation chi0 right direction other :=
          (site_potential_first_variation_zero_of_direction_zero
            chi0 right direction other hPsi hChi).symm
  · apply Finset.sum_congr rfl
    intro edge _
    by_cases hincident : C19IncidentAt geometry site edge
    · rcases c19_incident_endpoints_agree geometry left right site
          agreement edge hincident with
        ⟨hPsiSource, hPsiTarget, hChiSource, hChiTarget⟩
      exact exterior_area_bond_first_variation_congr chi0 left right direction
        (c19Bond geometry edge) hPsiSource hPsiTarget hChiSource hChiTarget
    · have hsource : edge.1 ≠ site := fun h => hincident (Or.inl h)
      have htarget : geometry.neighbor edge.1 edge.2 ≠ site :=
        fun h => hincident (Or.inr h)
      rcases supported edge.1 hsource with
        ⟨hPsiSource, hChiSource⟩
      rcases supported (geometry.neighbor edge.1 edge.2) htarget with
        ⟨hPsiTarget, hChiTarget⟩
      calc
        exteriorAreaBondFirstVariation chi0 left direction
            (c19Bond geometry edge) = 0 :=
          exterior_area_bond_variation_zero_off_direction_support
            chi0 left direction (c19Bond geometry edge)
              hPsiSource hPsiTarget hChiSource hChiTarget
        _ = exteriorAreaBondFirstVariation chi0 right direction
            (c19Bond geometry edge) :=
          (exterior_area_bond_variation_zero_off_direction_support
            chi0 right direction (c19Bond geometry edge)
              hPsiSource hPsiTarget hChiSource hChiTarget).symm

theorem pure_psi_real_direction_supported_at
    [DecidableEq Site] (site : Site) (component : Fin 3) :
    StateDirectionSupportedAt
      (purePsiDirection (realCoordinateDirection site component)) site := by
  intro other hother
  constructor
  · funext fieldComponent
    simp [purePsiDirection, realCoordinateDirection, hother, c3Zero]
  · rfl

theorem pure_psi_imag_direction_supported_at
    [DecidableEq Site] (site : Site) (component : Fin 3) :
    StateDirectionSupportedAt
      (purePsiDirection (imagCoordinateDirection site component)) site := by
  intro other hother
  constructor
  · funext fieldComponent
    simp [purePsiDirection, imagCoordinateDirection, hother, c3Zero]
  · rfl

theorem pure_chi_real_direction_supported_at
    [DecidableEq Site] (site : Site) (component : Fin 3) :
    StateDirectionSupportedAt
      (pureChiDirection (realCoordinateDirection site component)) site := by
  intro other hother
  constructor
  · rfl
  · funext fieldComponent
    simp [pureChiDirection, realCoordinateDirection, hother, c3Zero]

theorem pure_chi_imag_direction_supported_at
    [DecidableEq Site] (site : Site) (component : Fin 3) :
    StateDirectionSupportedAt
      (pureChiDirection (imagCoordinateDirection site component)) site := by
  intro other hother
  constructor
  · rfl
  · funext fieldComponent
    simp [pureChiDirection, imagCoordinateDirection, hother, c3Zero]

/-- The fully assembled accelerations at one site are equal whenever the
current native states agree on that site's complete C19 neighborhood. -/
theorem c19_assembled_acceleration_is_local
    [Fintype Site] [DecidableEq Site]
    (chi0 B : ℝ) (geometry : C19Lattice Site)
    (left right : State Site) (site : Site)
    (agreement : C19LocalAgreement geometry left right site) :
    psiAcceleration chi0 left (c19Bond geometry) site =
        psiAcceleration chi0 right (c19Bond geometry) site ∧
      chiAcceleration chi0 B left (c19Bond geometry) site =
        chiAcceleration chi0 B right (c19Bond geometry) site := by
  constructor <;> funext component
  · have hreal := c19_full_variation_is_local chi0 geometry left right
      (purePsiDirection (realCoordinateDirection site component)) site
      agreement (pure_psi_real_direction_supported_at site component)
    have himag := c19_full_variation_is_local chi0 geometry left right
      (purePsiDirection (imagCoordinateDirection site component)) site
      agreement (pure_psi_imag_direction_supported_at site component)
    exact congrArg₂ (fun real imag : ℝ =>
      (real : ℂ) + (imag : ℂ) * I) hreal himag
  · have hreal := c19_full_variation_is_local chi0 geometry left right
      (pureChiDirection (realCoordinateDirection site component)) site
      agreement (pure_chi_real_direction_supported_at site component)
    have himag := c19_full_variation_is_local chi0 geometry left right
      (pureChiDirection (imagCoordinateDirection site component)) site
      agreement (pure_chi_imag_direction_supported_at site component)
    exact congrArg₂ (fun real imag : ℝ =>
      ((real / B : ℝ) : ℂ) + ((imag / B : ℝ) : ℂ) * I) hreal himag

/-- The actual synchronous centered update at a site has exactly the native
one-tick C19 domain of dependence: its previous layer is read at the center,
and its current layer is read only at the center and eighteen neighbors. -/
theorem native_c19_update_is_local
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (previousLeft previousRight currentLeft currentRight : State Site)
    (site : Site)
    (previousPsi : previousLeft.psi site = previousRight.psi site)
    (previousChi : previousLeft.chi site = previousRight.chi site)
    (agreement : C19LocalAgreement geometry currentLeft currentRight site) :
    (nativeNextState chi0 B dt (c19Bond geometry)
        previousLeft currentLeft).psi site =
        (nativeNextState chi0 B dt (c19Bond geometry)
          previousRight currentRight).psi site ∧
      (nativeNextState chi0 B dt (c19Bond geometry)
        previousLeft currentLeft).chi site =
        (nativeNextState chi0 B dt (c19Bond geometry)
          previousRight currentRight).chi site := by
  rcases c19_assembled_acceleration_is_local chi0 B geometry
      currentLeft currentRight site agreement with ⟨hPsi, hChi⟩
  constructor <;> funext component
  · simp only [nativeNextState, accelerationVelocity]
    rw [agreement.centerPsi, previousPsi, hPsi]
  · simp only [nativeNextState, accelerationVelocity]
    rw [agreement.centerChi, previousChi, hChi]

end
end LeanDirac

