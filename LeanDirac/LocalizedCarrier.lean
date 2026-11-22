/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.CornerRayAlgebra

@[expose] public section

namespace LeanDirac

/-!
# Localized carrier on a periodic cubic lattice

The eight carrier corners are embedded in a periodic cell of side three.
The other nineteen sites carry a freely evolving exterior field. This module
derives the carrier's C19 Dirichlet operator, its exact handed eigenmode and
invariance under the full native centered update. DIRAC-041 and the V2
arbitrary-box construction share these results.
-/

noncomputable section
open BigOperators Complex

abbrev Site27 := Fin 3 × Fin 3 × Fin 3

def fin3Succ : Fin 3 → Fin 3
  | 0 => 1
  | 1 => 2
  | _ => 0

def fin3Pred : Fin 3 → Fin 3
  | 0 => 2
  | 1 => 0
  | _ => 1

def fin3Shift (coordinate : Fin 3) (delta : ℤ) : Fin 3 :=
  if delta = 1 then fin3Succ coordinate
  else if delta = -1 then fin3Pred coordinate
  else coordinate

def site27Neighbor (site : Site27) (direction : C19Neighbor) : Site27 :=
  (fin3Shift site.1 (c19Coord 0 direction),
    fin3Shift site.2.1 (c19Coord 1 direction),
    fin3Shift site.2.2 (c19Coord 2 direction))

theorem site27_neighbor_reverse (site : Site27) (direction : C19Neighbor) :
    site27Neighbor (site27Neighbor site direction) (c19Reverse direction) =
      site := by
  rcases site with ⟨x, y, z⟩
  fin_cases x <;> fin_cases y <;> fin_cases z <;>
    fin_cases direction <;> rfl

def site27C19Geometry : C19Lattice Site27 where
  neighbor := site27Neighbor
  neighbor_reverse := site27_neighbor_reverse

def Site27IsCarrier (site : Site27) : Prop :=
  site.1.val < 2 ∧ site.2.1.val < 2 ∧ site.2.2.val < 2

instance (site : Site27) : Decidable (Site27IsCarrier site) :=
  by
    unfold Site27IsCarrier
    infer_instance

def site27Corner (site : Site27) : CubeCorner :=
  match site.1.val, site.2.1.val, site.2.2.val with
  | 0, 0, 0 => 0
  | 0, 0, 1 => 1
  | 0, 1, 0 => 2
  | 0, 1, 1 => 3
  | 1, 0, 0 => 4
  | 1, 0, 1 => 5
  | 1, 1, 0 => 6
  | 1, 1, 1 => 7
  | _, _, _ => 0

def cornerSite27 (corner : CubeCorner) : Site27 :=
  match corner.val with
  | 0 => (0, 0, 0)
  | 1 => (0, 0, 1)
  | 2 => (0, 1, 0)
  | 3 => (0, 1, 1)
  | 4 => (1, 0, 0)
  | 5 => (1, 0, 1)
  | 6 => (1, 1, 0)
  | _ => (1, 1, 1)

@[simp] theorem corner_site27_is_carrier (corner : CubeCorner) :
    Site27IsCarrier (cornerSite27 corner) := by
  fin_cases corner <;> decide

@[simp] theorem site27_corner_corner_site27 (corner : CubeCorner) :
    site27Corner (cornerSite27 corner) = corner := by
  fin_cases corner <;> rfl

def localizedTwistMinus : Cell8
  | 0 => 1
  | 1 => -I
  | 2 => -1
  | 3 => I
  | 4 => -1
  | 5 => I
  | 6 => 1
  | 7 => -I

def localizedTwistPlus : Cell8 := fun corner => star (localizedTwistMinus corner)

@[simp] theorem localized_twist_minus_norm (corner : CubeCorner) :
    Complex.normSq (localizedTwistMinus corner) = 1 := by
  fin_cases corner <;> simp [localizedTwistMinus, Complex.normSq]

@[simp] theorem localized_twist_plus_norm (corner : CubeCorner) :
    Complex.normSq (localizedTwistPlus corner) = 1 := by
  fin_cases corner <;>
    simp [localizedTwistPlus, localizedTwistMinus, Complex.normSq]

def carrierNeighborAmplitude (amplitude : Cell8)
    (corner : CubeCorner) (direction : C19Neighbor) : ℂ :=
  let target := site27Neighbor (cornerSite27 corner) direction
  if Site27IsCarrier target then amplitude (site27Corner target) else 0

def localizedC19Dirichlet (amplitude : Cell8) : Cell8 := fun corner =>
  ∑ direction : C19Neighbor,
    (c19Weight direction : ℂ) *
      (amplitude corner - carrierNeighborAmplitude amplitude corner direction)

def carrierNeighborTable (amplitude : Cell8)
    (corner : CubeCorner) (direction : C19Neighbor) : ℂ :=
  match corner.val, direction.val with
  | 0, 0 => amplitude 4 | 0, 2 => amplitude 2 | 0, 4 => amplitude 1
  | 0, 9 => amplitude 6 | 0, 13 => amplitude 5 | 0, 17 => amplitude 3
  | 1, 0 => amplitude 5 | 1, 2 => amplitude 3 | 1, 5 => amplitude 0
  | 1, 9 => amplitude 7 | 1, 12 => amplitude 4 | 1, 16 => amplitude 2
  | 2, 0 => amplitude 6 | 2, 3 => amplitude 0 | 2, 4 => amplitude 3
  | 2, 8 => amplitude 4 | 2, 13 => amplitude 7 | 2, 15 => amplitude 1
  | 3, 0 => amplitude 7 | 3, 3 => amplitude 1 | 3, 5 => amplitude 2
  | 3, 8 => amplitude 5 | 3, 12 => amplitude 6 | 3, 14 => amplitude 0
  | 4, 1 => amplitude 0 | 4, 2 => amplitude 6 | 4, 4 => amplitude 5
  | 4, 7 => amplitude 2 | 4, 11 => amplitude 1 | 4, 17 => amplitude 7
  | 5, 1 => amplitude 1 | 5, 2 => amplitude 7 | 5, 5 => amplitude 4
  | 5, 7 => amplitude 3 | 5, 10 => amplitude 0 | 5, 16 => amplitude 6
  | 6, 1 => amplitude 2 | 6, 3 => amplitude 4 | 6, 4 => amplitude 7
  | 6, 6 => amplitude 0 | 6, 11 => amplitude 3 | 6, 15 => amplitude 5
  | 7, 1 => amplitude 3 | 7, 3 => amplitude 5 | 7, 5 => amplitude 6
  | 7, 6 => amplitude 1 | 7, 10 => amplitude 2 | 7, 14 => amplitude 4
  | _, _ => 0

theorem carrier_neighbor_amplitude_eq_table (amplitude : Cell8)
    (corner : CubeCorner) (direction : C19Neighbor) :
    carrierNeighborAmplitude amplitude corner direction =
      carrierNeighborTable amplitude corner direction := by
  fin_cases corner <;> fin_cases direction <;> rfl

def cubeFlipX : CubeCorner → CubeCorner
  | 0 => 4 | 1 => 5 | 2 => 6 | 3 => 7
  | 4 => 0 | 5 => 1 | 6 => 2 | _ => 3

def cubeFlipY : CubeCorner → CubeCorner
  | 0 => 2 | 1 => 3 | 2 => 0 | 3 => 1
  | 4 => 6 | 5 => 7 | 6 => 4 | _ => 5

def cubeFlipZ : CubeCorner → CubeCorner
  | 0 => 1 | 1 => 0 | 2 => 3 | 3 => 2
  | 4 => 5 | 5 => 4 | 6 => 7 | _ => 6

def localizedC19FlipFormula (amplitude : Cell8) (corner : CubeCorner) : ℂ :=
  24 * amplitude corner -
    2 * (amplitude (cubeFlipX corner) + amplitude (cubeFlipY corner) +
      amplitude (cubeFlipZ corner)) -
    (amplitude (cubeFlipY (cubeFlipX corner)) +
      amplitude (cubeFlipZ (cubeFlipX corner)) +
      amplitude (cubeFlipZ (cubeFlipY corner)))

private theorem localized_c19_flip_formula_0 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 0 =
      localizedC19FlipFormula amplitude 0 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_1 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 1 =
      localizedC19FlipFormula amplitude 1 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_2 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 2 =
      localizedC19FlipFormula amplitude 2 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_3 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 3 =
      localizedC19FlipFormula amplitude 3 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_4 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 4 =
      localizedC19FlipFormula amplitude 4 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_5 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 5 =
      localizedC19FlipFormula amplitude 5 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_6 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 6 =
      localizedC19FlipFormula amplitude 6 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

private theorem localized_c19_flip_formula_7 (amplitude : Cell8) :
    localizedC19Dirichlet amplitude 7 =
      localizedC19FlipFormula amplitude 7 := by
  simp [localizedC19Dirichlet, localizedC19FlipFormula,
    carrier_neighbor_amplitude_eq_table, carrierNeighborTable, c19Weight,
    cubeFlipX, cubeFlipY, cubeFlipZ, Fin.sum_univ_succ]
  ring

theorem localized_c19_flip_formula (amplitude : Cell8) :
    localizedC19Dirichlet amplitude =
      localizedC19FlipFormula amplitude := by
  funext corner
  fin_cases corner
  · exact localized_c19_flip_formula_0 amplitude
  · exact localized_c19_flip_formula_1 amplitude
  · exact localized_c19_flip_formula_2 amplitude
  · exact localized_c19_flip_formula_3 amplitude
  · exact localized_c19_flip_formula_4 amplitude
  · exact localized_c19_flip_formula_5 amplitude
  · exact localized_c19_flip_formula_6 amplitude
  · exact localized_c19_flip_formula_7 amplitude

theorem localized_twist_minus_eigenmode :
    localizedC19Dirichlet localizedTwistMinus =
      fun corner => 27 * localizedTwistMinus corner := by
  rw [localized_c19_flip_formula]
  funext corner
  fin_cases corner <;>
    simp [localizedC19FlipFormula, localizedTwistMinus,
      cubeFlipX, cubeFlipY, cubeFlipZ] <;>
    ring

theorem carrier_neighbor_amplitude_star (amplitude : Cell8)
    (corner : CubeCorner) (direction : C19Neighbor) :
    carrierNeighborAmplitude (fun c => star (amplitude c)) corner direction =
      star (carrierNeighborAmplitude amplitude corner direction) := by
  unfold carrierNeighborAmplitude
  dsimp only
  by_cases hcarrier :
      Site27IsCarrier (site27Neighbor (cornerSite27 corner) direction)
  · simp [hcarrier]
  · simp [hcarrier]

theorem localized_c19_dirichlet_star (amplitude : Cell8) :
    localizedC19Dirichlet (fun corner => star (amplitude corner)) =
      fun corner => star (localizedC19Dirichlet amplitude corner) := by
  funext corner
  change (∑ direction : C19Neighbor,
      (c19Weight direction : ℂ) *
        (star (amplitude corner) -
          carrierNeighborAmplitude (fun c => star (amplitude c))
            corner direction)) =
    star (∑ direction : C19Neighbor,
      (c19Weight direction : ℂ) *
        (amplitude corner - carrierNeighborAmplitude amplitude corner direction))
  change _ = (starRingEnd ℂ) (∑ direction : C19Neighbor,
    (c19Weight direction : ℂ) *
      (amplitude corner - carrierNeighborAmplitude amplitude corner direction))
  rw [map_sum (starRingEnd ℂ)]
  apply Finset.sum_congr rfl
  intro direction _
  rw [map_mul, map_sub, carrier_neighbor_amplitude_star]
  simp

theorem localized_twist_plus_eigenmode :
    localizedC19Dirichlet localizedTwistPlus =
      fun corner => 27 * localizedTwistPlus corner := by
  rw [show localizedTwistPlus = fun corner =>
      star (localizedTwistMinus corner) by rfl,
    localized_c19_dirichlet_star, localized_twist_minus_eigenmode]
  funext corner
  simp

/-! A one-corner phase defect supplies the non-vacuous control.  It has the
same unit magnitude at every corner as the accepted twist, but the exact C19
area-force operator develops a component outside its one-dimensional span. -/

def localizedTwistDefect : Cell8
  | 0 => -1
  | 1 => -I
  | 2 => -1
  | 3 => I
  | 4 => -1
  | 5 => I
  | 6 => 1
  | 7 => -I

@[simp] theorem localized_twist_defect_norm (corner : CubeCorner) :
    Complex.normSq (localizedTwistDefect corner) = 1 := by
  fin_cases corner <;>
    simp [localizedTwistDefect, Complex.normSq]

def HasC19NormalComponent (amplitude : Cell8) : Prop :=
  ¬ ∃ eigenvalue : ℂ,
    localizedC19Dirichlet amplitude =
      fun corner => eigenvalue * amplitude corner

theorem one_corner_phase_defect_has_c19_normal_component :
    HasC19NormalComponent localizedTwistDefect := by
  rintro ⟨eigenvalue, heigen⟩
  rw [localized_c19_flip_formula] at heigen
  have h0 := congrFun heigen (0 : CubeCorner)
  have h6 := congrFun heigen (6 : CubeCorner)
  simp [localizedC19FlipFormula, localizedTwistDefect,
    cubeFlipX, cubeFlipY, cubeFlipZ] at h0 h6
  apply_fun Complex.re at h0 h6
  norm_num at h0 h6
  linarith

def scalarC19Difference (geometry : C19Lattice Site)
    (field : Site → ℂ) (edge : C19Edge Site) : ℂ :=
  field (geometry.neighbor edge.1 edge.2) - field edge.1

def scalarC19Laplacian (geometry : C19Lattice Site)
    (field : Site → ℂ) (site : Site) : ℂ :=
  ∑ direction : C19Neighbor,
    (c19Weight direction : ℂ) *
      (field site - field (geometry.neighbor site direction))

def reverseC19EdgeEquiv (geometry : C19Lattice Site) :
    C19Edge Site ≃ C19Edge Site where
  toFun := reverseC19Edge geometry
  invFun := reverseC19Edge geometry
  left_inv := by
    intro edge
    rcases edge with ⟨site, direction⟩
    apply Prod.ext
    · exact geometry.neighbor_reverse site direction
    · exact c19_reverse_involutive direction
  right_inv := by
    intro edge
    rcases edge with ⟨site, direction⟩
    apply Prod.ext
    · exact geometry.neighbor_reverse site direction
    · exact c19_reverse_involutive direction

theorem c19_weighted_target_sum_eq_source_sum
    [Fintype Site] (geometry : C19Lattice Site) (field : Site → ℂ) :
    (∑ edge : C19Edge Site,
      (c19Weight edge.2 : ℂ) *
        field (geometry.neighbor edge.1 edge.2)) =
      ∑ edge : C19Edge Site,
        (c19Weight edge.2 : ℂ) * field edge.1 := by
  refine Fintype.sum_equiv (reverseC19EdgeEquiv geometry) _ _ ?_
  intro edge
  simp [reverseC19EdgeEquiv, reverseC19Edge, c19_reverse_weight]

theorem c19_weighted_cross_sum_reverse
    [Fintype Site] (geometry : C19Lattice Site)
    (left right : Site → ℂ) :
    (∑ edge : C19Edge Site,
      (c19Weight edge.2 : ℂ) *
        star (left (geometry.neighbor edge.1 edge.2)) * right edge.1) =
      ∑ edge : C19Edge Site,
        (c19Weight edge.2 : ℂ) * star (left edge.1) *
          right (geometry.neighbor edge.1 edge.2) := by
  refine Fintype.sum_equiv (reverseC19EdgeEquiv geometry) _ _ ?_
  intro edge
  simp [reverseC19EdgeEquiv, reverseC19Edge, c19_reverse_weight,
    geometry.neighbor_reverse]

theorem c19_scalar_green_identity [Fintype Site]
    (geometry : C19Lattice Site) (left right : Site → ℂ) :
    (∑ edge : C19Edge Site,
      (c19Weight edge.2 : ℂ) *
        star (scalarC19Difference geometry left edge) *
          scalarC19Difference geometry right edge) =
      2 * ∑ site : Site,
        star (scalarC19Laplacian geometry left site) * right site := by
  let targetTarget : ℂ := ∑ edge : C19Edge Site,
    (c19Weight edge.2 : ℂ) *
      star (left (geometry.neighbor edge.1 edge.2)) *
        right (geometry.neighbor edge.1 edge.2)
  let targetSource : ℂ := ∑ edge : C19Edge Site,
    (c19Weight edge.2 : ℂ) *
      star (left (geometry.neighbor edge.1 edge.2)) * right edge.1
  let sourceTarget : ℂ := ∑ edge : C19Edge Site,
    (c19Weight edge.2 : ℂ) * star (left edge.1) *
      right (geometry.neighbor edge.1 edge.2)
  let sourceSource : ℂ := ∑ edge : C19Edge Site,
    (c19Weight edge.2 : ℂ) * star (left edge.1) * right edge.1
  have htt : targetTarget = sourceSource := by
    dsimp [targetTarget, sourceSource]
    simpa [mul_assoc] using
      c19_weighted_target_sum_eq_source_sum geometry
        (fun site => star (left site) * right site)
  have hts : targetSource = sourceTarget := by
    dsimp [targetSource, sourceTarget]
    exact c19_weighted_cross_sum_reverse geometry left right
  have hleft :
      (∑ edge : C19Edge Site,
        (c19Weight edge.2 : ℂ) *
          star (scalarC19Difference geometry left edge) *
            scalarC19Difference geometry right edge) =
        targetTarget - targetSource - sourceTarget + sourceSource := by
    dsimp [targetTarget, targetSource, sourceTarget, sourceSource]
    simp only [scalarC19Difference, map_sub]
    rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib,
      ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro edge _
    ring
  have hright :
      (∑ site : Site,
        star (scalarC19Laplacian geometry left site) * right site) =
        sourceSource - targetSource := by
    dsimp [sourceSource, targetSource]
    rw [Fintype.sum_prod_type]
    simp only [scalarC19Laplacian, map_sum, map_mul, map_sub]
    rw [Fintype.sum_prod_type]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro site _
    rw [Finset.sum_mul]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro direction _
    simp
    ring
  rw [hleft, htt, ← hts, hright]
  ring

theorem c19_scalar_green_real_identity [Fintype Site]
    (geometry : C19Lattice Site) (left right : Site → ℂ) :
    (∑ edge : C19Edge Site,
      c19Weight edge.2 *
        (star (scalarC19Difference geometry left edge) *
          scalarC19Difference geometry right edge).re) =
      2 * ∑ site : Site,
        (star (scalarC19Laplacian geometry left site) * right site).re := by
  have h := congrArg Complex.re
    (c19_scalar_green_identity geometry left right)
  simpa [Complex.mul_re, mul_add, add_mul, mul_assoc] using h

def haloDirection : C3 := basis 2

def localizedCarrierState (chi0Value : ℝ) (psiAmplitude chiAmplitude : ℂ) :
    State Site27 where
  psi site :=
    if Site27IsCarrier site then
      c3ScaleComplex
        (psiAmplitude * localizedTwistMinus (site27Corner site))
        (tetraCorner (site27Corner site))
    else c3Zero
  chi site :=
    if Site27IsCarrier site then
      c3ScaleComplex chiAmplitude (companion (site27Corner site))
    else c3ScaleComplex (chi0Value : ℂ) haloDirection

theorem localized_carrier_psi_at_carrier (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hcarrier : Site27IsCarrier site) :
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site =
      c3ScaleComplex
        (psiAmplitude * localizedTwistMinus (site27Corner site))
        (tetraCorner (site27Corner site)) := by
  simp [localizedCarrierState, hcarrier]

theorem localized_carrier_chi_at_carrier (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hcarrier : Site27IsCarrier site) :
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site =
      c3ScaleComplex chiAmplitude (companion (site27Corner site)) := by
  simp [localizedCarrierState, hcarrier]

theorem localized_carrier_psi_at_halo (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hexterior : ¬ Site27IsCarrier site) :
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site =
      c3Zero := by
  simp [localizedCarrierState, hexterior]

theorem localized_carrier_chi_at_halo (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hexterior : ¬ Site27IsCarrier site) :
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site =
      c3ScaleComplex (chi0Value : ℂ) haloDirection := by
  simp [localizedCarrierState, hexterior]

def InLocalizedCarrierSector (chi0Value : ℝ) (state : State Site27) : Prop :=
  ∃ psiAmplitude chiAmplitude,
    state = localizedCarrierState chi0Value psiAmplitude chiAmplitude

def localizedAreaCoefficient (psiAmplitude chiAmplitude : ℂ)
    (site : Site27) : ℂ :=
  if Site27IsCarrier site then
    -(chiAmplitude * psiAmplitude *
      localizedTwistMinus (site27Corner site))
  else 0

def extendCarrierAmplitude (amplitude : Cell8) (site : Site27) : ℂ :=
  if Site27IsCarrier site then amplitude (site27Corner site) else 0

theorem extend_carrier_amplitude_corner (amplitude : Cell8)
    (corner : CubeCorner) :
    extendCarrierAmplitude amplitude (cornerSite27 corner) = amplitude corner := by
  simp [extendCarrierAmplitude]

theorem extend_twist_laplacian_at_corner (corner : CubeCorner) :
    scalarC19Laplacian site27C19Geometry
        (extendCarrierAmplitude localizedTwistMinus) (cornerSite27 corner) =
      27 * localizedTwistMinus corner := by
  have hoperator := congrFun localized_twist_minus_eigenmode corner
  rw [← hoperator]
  unfold scalarC19Laplacian localizedC19Dirichlet
  apply Finset.sum_congr rfl
  intro direction _
  rw [extend_carrier_amplitude_corner]
  congr 2

theorem localized_area_coefficient_as_extension (psiAmplitude chiAmplitude : ℂ) :
    localizedAreaCoefficient psiAmplitude chiAmplitude =
      extendCarrierAmplitude (fun corner =>
        -(chiAmplitude * psiAmplitude * localizedTwistMinus corner)) := by
  funext site
  unfold localizedAreaCoefficient extendCarrierAmplitude
  by_cases hcarrier : Site27IsCarrier site <;> simp [hcarrier]

theorem localized_area_laplacian_at_corner
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) :
    scalarC19Laplacian site27C19Geometry
        (localizedAreaCoefficient psiAmplitude chiAmplitude)
        (cornerSite27 corner) =
      27 * (-(chiAmplitude * psiAmplitude * localizedTwistMinus corner)) := by
  rw [localized_area_coefficient_as_extension]
  unfold scalarC19Laplacian
  rw [show (∑ direction : C19Neighbor,
      (c19Weight direction : ℂ) *
        (extendCarrierAmplitude
            (fun c => -(chiAmplitude * psiAmplitude * localizedTwistMinus c))
            (cornerSite27 corner) -
          extendCarrierAmplitude
            (fun c => -(chiAmplitude * psiAmplitude * localizedTwistMinus c))
            (site27C19Geometry.neighbor (cornerSite27 corner) direction))) =
      -(chiAmplitude * psiAmplitude) *
        scalarC19Laplacian site27C19Geometry
          (extendCarrierAmplitude localizedTwistMinus) (cornerSite27 corner) by
    unfold scalarC19Laplacian
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro direction _
    unfold extendCarrierAmplitude
    by_cases hsource : Site27IsCarrier (cornerSite27 corner)
    · by_cases htarget : Site27IsCarrier
          (site27C19Geometry.neighbor (cornerSite27 corner) direction)
      · simp [hsource, htarget]
        ring
      · simp [hsource, htarget]
        ring
    · exact False.elim (hsource (corner_site27_is_carrier corner))]
  rw [extend_twist_laplacian_at_corner]
  ring

theorem localized_wedge_all_sites_early (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27) :
    wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude site) unitArea := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [localized_carrier_chi_at_carrier _ _ _ _ hcarrier,
      localized_carrier_psi_at_carrier _ _ _ _ hcarrier]
    rw [show localizedAreaCoefficient psiAmplitude chiAmplitude site =
        -(chiAmplitude * psiAmplitude *
          localizedTwistMinus (site27Corner site)) by
      simp [localizedAreaCoefficient, hcarrier]]
    rw [wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
    funext component
    simp [c3ScaleComplex, c3Neg]
    ring
  · rw [localized_carrier_chi_at_halo _ _ _ _ hcarrier,
      localized_carrier_psi_at_halo _ _ _ _ hcarrier]
    funext component
    fin_cases component <;>
      simp [wedge, c3ScaleComplex, c3Zero, localizedAreaCoefficient,
        hcarrier]

def singleCornerAmplitude (selected : CubeCorner) (value : ℂ) : Cell8 :=
  fun corner => if corner = selected then value else 0

@[simp] theorem c3_scale_complex_zero (u : C3) :
    c3ScaleComplex 0 u = c3Zero := by
  funext component
  simp [c3ScaleComplex, c3Zero]

@[simp] theorem c3_real_pair_zero_right (u : C3) :
    c3RealPair u c3Zero = 0 := by
  simp [c3RealPair, c3Zero]

@[simp] theorem c3_real_pair_zero_left (u : C3) :
    c3RealPair c3Zero u = 0 := by
  simp [c3RealPair, c3Zero]

theorem corner_site27_site27_corner (site : Site27)
    (hcarrier : Site27IsCarrier site) :
    cornerSite27 (site27Corner site) = site := by
  rcases site with ⟨x, y, z⟩
  fin_cases x <;> fin_cases y <;> fin_cases z <;>
    simp_all [Site27IsCarrier, site27Corner, cornerSite27]

def localizedPsiRayDirection (variation : Cell8) : State Site27 where
  psi site :=
    if Site27IsCarrier site then
      c3ScaleComplex (variation (site27Corner site))
        (tetraCorner (site27Corner site))
    else c3Zero
  chi _ := c3Zero

def localizedChiRayDirection (variation : Cell8) : State Site27 where
  psi _ := c3Zero
  chi site :=
    if Site27IsCarrier site then
      c3ScaleComplex (variation (site27Corner site))
        (companion (site27Corner site))
    else c3Zero

theorem localized_psi_ray_direction_single (selected : CubeCorner)
    (value : ℂ) (site : Site27) :
    (localizedPsiRayDirection
      (singleCornerAmplitude selected value)).psi site =
      if site = cornerSite27 selected then
        c3ScaleComplex value (tetraCorner selected)
      else c3Zero := by
  by_cases hsite : site = cornerSite27 selected
  · subst site
    simp [localizedPsiRayDirection, singleCornerAmplitude]
  · by_cases hcarrier : Site27IsCarrier site
    · have hcorner : site27Corner site ≠ selected := by
        intro heq
        apply hsite
        calc
          site = cornerSite27 (site27Corner site) :=
            (corner_site27_site27_corner site hcarrier).symm
          _ = cornerSite27 selected := congrArg cornerSite27 heq
      simp [localizedPsiRayDirection, singleCornerAmplitude, hsite,
        hcarrier, hcorner]
    · simp [localizedPsiRayDirection, hsite, hcarrier]

theorem localized_chi_ray_direction_single (selected : CubeCorner)
    (value : ℂ) (site : Site27) :
    (localizedChiRayDirection
      (singleCornerAmplitude selected value)).chi site =
      if site = cornerSite27 selected then
        c3ScaleComplex value (companion selected)
      else c3Zero := by
  by_cases hsite : site = cornerSite27 selected
  · subst site
    simp [localizedChiRayDirection, singleCornerAmplitude]
  · by_cases hcarrier : Site27IsCarrier site
    · have hcorner : site27Corner site ≠ selected := by
        intro heq
        apply hsite
        calc
          site = cornerSite27 (site27Corner site) :=
            (corner_site27_site27_corner site hcarrier).symm
          _ = cornerSite27 selected := congrArg cornerSite27 heq
      simp [localizedChiRayDirection, singleCornerAmplitude, hsite,
        hcarrier, hcorner]
    · simp [localizedChiRayDirection, hsite, hcarrier]

def localizedPsiAreaVariation (chiAmplitude : ℂ)
    (variation : Cell8) : Site27 → ℂ :=
  extendCarrierAmplitude (fun corner => -(chiAmplitude * variation corner))

def localizedChiAreaVariation (psiAmplitude : ℂ)
    (variation : Cell8) : Site27 → ℂ :=
  extendCarrierAmplitude (fun corner =>
    -(variation corner * psiAmplitude * localizedTwistMinus corner))

theorem localized_psi_area_variation_single (chiAmplitude value : ℂ)
    (selected : CubeCorner) (site : Site27) :
    localizedPsiAreaVariation chiAmplitude
        (singleCornerAmplitude selected value) site =
      if site = cornerSite27 selected then -(chiAmplitude * value) else 0 := by
  rcases site with ⟨x, y, z⟩
  fin_cases x <;> fin_cases y <;> fin_cases z <;> fin_cases selected <;>
    simp [localizedPsiAreaVariation, extendCarrierAmplitude,
      singleCornerAmplitude, Site27IsCarrier, site27Corner, cornerSite27]

theorem localized_chi_area_variation_single (psiAmplitude value : ℂ)
    (selected : CubeCorner) (site : Site27) :
    localizedChiAreaVariation psiAmplitude
        (singleCornerAmplitude selected value) site =
      if site = cornerSite27 selected then
        -(value * psiAmplitude * localizedTwistMinus selected)
      else 0 := by
  rcases site with ⟨x, y, z⟩
  fin_cases x <;> fin_cases y <;> fin_cases z <;> fin_cases selected <;>
    simp [localizedChiAreaVariation, extendCarrierAmplitude,
      singleCornerAmplitude, Site27IsCarrier, site27Corner, cornerSite27]

@[simp] theorem wedge_zero_left (u : C3) : wedge c3Zero u = c3Zero := by
  funext component
  fin_cases component <;> simp [wedge, c3Zero]

@[simp] theorem wedge_zero_right (u : C3) : wedge u c3Zero = c3Zero := by
  funext component
  fin_cases component <;> simp [wedge, c3Zero]

@[simp] theorem c3_add_zero_left (u : C3) : c3Add c3Zero u = u := by
  funext component
  simp [c3Add, c3Zero]

theorem localized_psi_wedge_first_variation (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8) (site : Site27) :
    wedgeFirstVariation
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection variation) site =
      c3ScaleComplex (localizedPsiAreaVariation chiAmplitude variation site)
        unitArea := by
  unfold wedgeFirstVariation
  by_cases hcarrier : Site27IsCarrier site
  · rw [show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site =
        c3ScaleComplex chiAmplitude (companion (site27Corner site)) by
      exact localized_carrier_chi_at_carrier chi0Value psiAmplitude
        chiAmplitude site hcarrier]
    rw [show (localizedPsiRayDirection variation).chi site = c3Zero by rfl,
      show (localizedPsiRayDirection variation).psi site =
        c3ScaleComplex (variation (site27Corner site))
          (tetraCorner (site27Corner site)) by
        simp [localizedPsiRayDirection, hcarrier],
      wedge_zero_left, c3_add_zero_left,
      wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
    funext component
    fin_cases component <;>
      simp [c3Neg, c3ScaleComplex,
        localizedPsiAreaVariation, extendCarrierAmplitude, hcarrier,
        unitArea]
  · rw [show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site =
        c3ScaleComplex (chi0Value : ℂ) haloDirection by
      exact localized_carrier_chi_at_halo chi0Value psiAmplitude
        chiAmplitude site hcarrier]
    funext component
    fin_cases component <;>
      simp [localizedPsiRayDirection, hcarrier,
        localizedPsiAreaVariation, extendCarrierAmplitude, c3Add,
        c3Zero, wedge, c3ScaleComplex]

theorem localized_chi_wedge_first_variation (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8) (site : Site27) :
    wedgeFirstVariation
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection variation) site =
      c3ScaleComplex
        (localizedChiAreaVariation psiAmplitude variation site) unitArea := by
  unfold wedgeFirstVariation
  by_cases hcarrier : Site27IsCarrier site
  · rw [show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site =
        c3ScaleComplex
          (psiAmplitude * localizedTwistMinus (site27Corner site))
          (tetraCorner (site27Corner site)) by
      exact localized_carrier_psi_at_carrier chi0Value psiAmplitude
        chiAmplitude site hcarrier]
    rw [show (localizedChiRayDirection variation).chi site =
        c3ScaleComplex (variation (site27Corner site))
          (companion (site27Corner site)) by
        simp [localizedChiRayDirection, hcarrier],
      show (localizedChiRayDirection variation).psi site = c3Zero by rfl,
      wedge_zero_right, c3_add_zero_right,
      wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
    funext component
    fin_cases component
    all_goals
      simp [c3Neg, c3ScaleComplex,
        localizedChiAreaVariation, extendCarrierAmplitude, hcarrier,
        unitArea] <;> ring
  · rw [show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site =
        c3Zero by
      exact localized_carrier_psi_at_halo chi0Value psiAmplitude
        chiAmplitude site hcarrier]
    funext component
    fin_cases component <;>
      simp [localizedChiRayDirection, hcarrier,
        localizedChiAreaVariation, extendCarrierAmplitude, c3Add,
        c3Zero, wedge, c3ScaleComplex]

theorem c3_real_pair_scaled_unit_area (left right : ℂ) :
    c3RealPair (c3ScaleComplex left unitArea)
      (c3ScaleComplex right unitArea) = (star left * right).re := by
  simp [c3RealPair, c3ScaleComplex, unitArea, Fin.sum_univ_succ]

theorem localized_psi_bond_variation_as_scalar (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8)
    (edge : Bond Site27) :
    exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection variation) edge =
      edge.weight / chi0Value^2 *
        (star
          (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
            localizedAreaCoefficient psiAmplitude chiAmplitude edge.source) *
          (localizedPsiAreaVariation chiAmplitude variation edge.target -
            localizedPsiAreaVariation chiAmplitude variation edge.source)).re := by
  unfold exteriorAreaBondFirstVariation bondAreaDifference
    bondAreaFirstVariation
  rw [localized_wedge_all_sites_early, localized_wedge_all_sites_early,
    localized_psi_wedge_first_variation,
    localized_psi_wedge_first_variation]
  rw [show difference
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
          localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring,
    show difference
      (c3ScaleComplex
        (localizedPsiAreaVariation chiAmplitude variation edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedPsiAreaVariation chiAmplitude variation edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedPsiAreaVariation chiAmplitude variation edge.target -
          localizedPsiAreaVariation chiAmplitude variation edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring,
    c3_real_pair_scaled_unit_area]

theorem localized_chi_bond_variation_as_scalar (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8)
    (edge : Bond Site27) :
    exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection variation) edge =
      edge.weight / chi0Value^2 *
        (star
          (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
            localizedAreaCoefficient psiAmplitude chiAmplitude edge.source) *
          (localizedChiAreaVariation psiAmplitude variation edge.target -
            localizedChiAreaVariation psiAmplitude variation edge.source)).re := by
  unfold exteriorAreaBondFirstVariation bondAreaDifference
    bondAreaFirstVariation
  rw [localized_wedge_all_sites_early, localized_wedge_all_sites_early,
    localized_chi_wedge_first_variation,
    localized_chi_wedge_first_variation]
  rw [show difference
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
          localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring,
    show difference
      (c3ScaleComplex
        (localizedChiAreaVariation psiAmplitude variation edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedChiAreaVariation psiAmplitude variation edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedChiAreaVariation psiAmplitude variation edge.target -
          localizedChiAreaVariation psiAmplitude variation edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring,
    c3_real_pair_scaled_unit_area]

theorem localized_psi_bond_variation_sum (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8) :
    (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection variation)
        (c19Bond site27C19Geometry edge)) =
      2 / chi0Value^2 *
        ∑ site : Site27,
          (star
              (scalarC19Laplacian site27C19Geometry
                (localizedAreaCoefficient psiAmplitude chiAmplitude) site) *
            localizedPsiAreaVariation chiAmplitude variation site).re := by
  rw [show (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection variation)
        (c19Bond site27C19Geometry edge)) =
      (1 / chi0Value^2) *
        ∑ edge : C19Edge Site27,
          c19Weight edge.2 *
            (star
              (scalarC19Difference site27C19Geometry
                (localizedAreaCoefficient psiAmplitude chiAmplitude) edge) *
              scalarC19Difference site27C19Geometry
                (localizedPsiAreaVariation chiAmplitude variation) edge).re by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro edge _
    rw [localized_psi_bond_variation_as_scalar]
    simp [c19Bond, scalarC19Difference]
    ring]
  rw [c19_scalar_green_real_identity]
  ring

theorem localized_chi_bond_variation_sum (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (variation : Cell8) :
    (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection variation)
        (c19Bond site27C19Geometry edge)) =
      2 / chi0Value^2 *
        ∑ site : Site27,
          (star
              (scalarC19Laplacian site27C19Geometry
                (localizedAreaCoefficient psiAmplitude chiAmplitude) site) *
            localizedChiAreaVariation psiAmplitude variation site).re := by
  rw [show (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection variation)
        (c19Bond site27C19Geometry edge)) =
      (1 / chi0Value^2) *
        ∑ edge : C19Edge Site27,
          c19Weight edge.2 *
            (star
              (scalarC19Difference site27C19Geometry
                (localizedAreaCoefficient psiAmplitude chiAmplitude) edge) *
              scalarC19Difference site27C19Geometry
                (localizedChiAreaVariation psiAmplitude variation) edge).re by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro edge _
    rw [localized_chi_bond_variation_as_scalar]
    simp [c19Bond, scalarC19Difference]
    ring]
  rw [c19_scalar_green_real_identity]
  ring

theorem localized_psi_green_sum_single (psiAmplitude chiAmplitude value : ℂ)
    (selected : CubeCorner) :
    (∑ site : Site27,
      (star
          (scalarC19Laplacian site27C19Geometry
            (localizedAreaCoefficient psiAmplitude chiAmplitude) site) *
        localizedPsiAreaVariation chiAmplitude
          (singleCornerAmplitude selected value) site).re) =
      (star
          (27 * (-(chiAmplitude * psiAmplitude *
            localizedTwistMinus selected))) *
        (-(chiAmplitude * value))).re := by
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · rw [localized_area_laplacian_at_corner,
      localized_psi_area_variation_single]
    simp
  · intro site _ hne
    rw [localized_psi_area_variation_single]
    simp [hne]
  · simp

theorem localized_chi_green_sum_single (psiAmplitude chiAmplitude value : ℂ)
    (selected : CubeCorner) :
    (∑ site : Site27,
      (star
          (scalarC19Laplacian site27C19Geometry
            (localizedAreaCoefficient psiAmplitude chiAmplitude) site) *
        localizedChiAreaVariation psiAmplitude
          (singleCornerAmplitude selected value) site).re) =
      (star
          (27 * (-(chiAmplitude * psiAmplitude *
            localizedTwistMinus selected))) *
        (-(value * psiAmplitude * localizedTwistMinus selected))).re := by
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · rw [localized_area_laplacian_at_corner,
      localized_chi_area_variation_single]
    simp
  · intro site _ hne
    rw [localized_chi_area_variation_single]
    simp [hne]
  · simp

theorem c3_real_pair_scaled_unit (left right : ℂ) (u : C3)
    (hunit : density u = 1) :
    c3RealPair (c3ScaleComplex left u) (c3ScaleComplex right u) =
      (star left * right).re := by
  rw [c3_real_pair_eq_inner_re, c3_inner_scale_left,
    c3_inner_scale_right, c3_inner_self, hunit]
  simp

theorem localized_psi_site_variation_single (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    (∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection
          (singleCornerAmplitude selected value)) site) =
      (Complex.normSq chiAmplitude + Complex.normSq psiAmplitude) *
        (star (psiAmplitude * localizedTwistMinus selected) * value).re := by
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · unfold sitePotentialFirstVariation
    dsimp
    rw [localized_carrier_chi_at_carrier chi0Value psiAmplitude
        chiAmplitude (cornerSite27 selected) (corner_site27_is_carrier selected),
      localized_carrier_psi_at_carrier chi0Value psiAmplitude
        chiAmplitude (cornerSite27 selected) (corner_site27_is_carrier selected),
      site27_corner_corner_site27,
      show
        (localizedPsiRayDirection
          (singleCornerAmplitude selected value)).psi
            (cornerSite27 selected) =
          c3ScaleComplex value (tetraCorner selected) by
        simp [localizedPsiRayDirection, singleCornerAmplitude]]
    rw [show
      (localizedPsiRayDirection
        (singleCornerAmplitude selected value)).chi (cornerSite27 selected) =
          c3Zero by rfl]
    rw [density_aligned_complex_scale _ _ (companion_density_one selected),
      density_aligned_complex_scale _ _ (tetra_corner_density_one selected),
      c3_real_pair_scaled_unit _ _ _ (tetra_corner_density_one selected)]
    simp [c3RealPair, c3Zero, Complex.normSq_mul]
  · intro site _ hne
    unfold sitePotentialFirstVariation
    rw [localized_psi_ray_direction_single]
    simp [hne, localizedPsiRayDirection, c3RealPair, c3Zero]
  · simp

theorem localized_chi_site_variation_single (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    (∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection
          (singleCornerAmplitude selected value)) site) =
      (Complex.normSq chiAmplitude - chi0Value^2 +
          Complex.normSq psiAmplitude) *
        (star chiAmplitude * value).re := by
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · unfold sitePotentialFirstVariation
    dsimp
    rw [localized_carrier_chi_at_carrier chi0Value psiAmplitude
        chiAmplitude (cornerSite27 selected) (corner_site27_is_carrier selected),
      localized_carrier_psi_at_carrier chi0Value psiAmplitude
        chiAmplitude (cornerSite27 selected) (corner_site27_is_carrier selected),
      site27_corner_corner_site27,
      show
        (localizedChiRayDirection
          (singleCornerAmplitude selected value)).chi
            (cornerSite27 selected) =
          c3ScaleComplex value (companion selected) by
        simp [localizedChiRayDirection, singleCornerAmplitude]]
    rw [show
      (localizedChiRayDirection
        (singleCornerAmplitude selected value)).psi (cornerSite27 selected) =
          c3Zero by rfl]
    rw [density_aligned_complex_scale _ _ (companion_density_one selected),
      density_aligned_complex_scale _ _ (tetra_corner_density_one selected),
      c3_real_pair_scaled_unit _ _ _ (companion_density_one selected)]
    simp [c3RealPair, c3Zero, Complex.normSq_mul]
  · intro site _ hne
    unfold sitePotentialFirstVariation
    rw [localized_chi_ray_direction_single]
    simp [hne, localizedChiRayDirection, c3RealPair, c3Zero]
  · simp

def localizedPsiStiffness (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) : ℝ :=
  Complex.normSq chiAmplitude + Complex.normSq psiAmplitude +
    54 / chi0Value^2 * Complex.normSq chiAmplitude

def localizedChiStiffness (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) : ℝ :=
  Complex.normSq chiAmplitude - chi0Value^2 + Complex.normSq psiAmplitude +
    54 / chi0Value^2 * Complex.normSq psiAmplitude

theorem localized_psi_bond_variation_single (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection (singleCornerAmplitude selected value))
        (c19Bond site27C19Geometry edge)) =
      (54 / chi0Value^2 * Complex.normSq chiAmplitude) *
        (star (psiAmplitude * localizedTwistMinus selected) * value).re := by
  rw [localized_psi_bond_variation_sum,
    localized_psi_green_sum_single]
  fin_cases selected <;>
    simp [localizedTwistMinus, Complex.normSq, Complex.mul_re,
      Complex.mul_im] <;>
    ring

theorem localized_chi_bond_variation_single (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection (singleCornerAmplitude selected value))
        (c19Bond site27C19Geometry edge)) =
      (54 / chi0Value^2 * Complex.normSq psiAmplitude) *
        (star chiAmplitude * value).re := by
  rw [localized_chi_bond_variation_sum,
    localized_chi_green_sum_single]
  fin_cases selected <;>
    simp [localizedTwistMinus, Complex.normSq, Complex.mul_re,
      Complex.mul_im] <;>
    ring

theorem localized_psi_force_single_ray (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    psiEulerLagrangeForce chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (c19Bond site27C19Geometry)
        (localizedPsiRayDirection
          (singleCornerAmplitude selected value)).psi =
      -localizedPsiStiffness chi0Value psiAmplitude chiAmplitude *
        (star (psiAmplitude * localizedTwistMinus selected) * value).re := by
  unfold psiEulerLagrangeForce lagrangianStateFirstVariation
    potentialFirstVariation purePsiDirection
  change -((∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedPsiRayDirection
          (singleCornerAmplitude selected value)) site) +
      ∑ edge : C19Edge Site27,
        exteriorAreaBondFirstVariation chi0Value
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (localizedPsiRayDirection
            (singleCornerAmplitude selected value))
          (c19Bond site27C19Geometry edge)) = _
  rw [localized_psi_site_variation_single,
    localized_psi_bond_variation_single]
  unfold localizedPsiStiffness
  ring

theorem localized_chi_force_single_ray (chi0Value : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (selected : CubeCorner) :
    chiEulerLagrangeForce chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (c19Bond site27C19Geometry)
        (localizedChiRayDirection
          (singleCornerAmplitude selected value)).chi =
      -localizedChiStiffness chi0Value psiAmplitude chiAmplitude *
        (star chiAmplitude * value).re := by
  unfold chiEulerLagrangeForce lagrangianStateFirstVariation
    potentialFirstVariation pureChiDirection
  change -((∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (localizedChiRayDirection
          (singleCornerAmplitude selected value)) site) +
      ∑ edge : C19Edge Site27,
        exteriorAreaBondFirstVariation chi0Value
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (localizedChiRayDirection
            (singleCornerAmplitude selected value))
          (c19Bond site27C19Geometry edge)) = _
  rw [localized_chi_site_variation_single,
    localized_chi_bond_variation_single]
  unfold localizedChiStiffness
  ring

@[simp] theorem localized_carrier_wedge (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) :
    wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi
          (cornerSite27 corner))
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi
          (cornerSite27 corner)) =
      c3ScaleComplex
        (-(chiAmplitude * psiAmplitude * localizedTwistMinus corner))
        unitArea := by
  simp [localizedCarrierState]
  rw [wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
  funext component
  simp [c3ScaleComplex, c3Neg]
  ring

@[simp] theorem localized_halo_wedge_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hexterior : ¬ Site27IsCarrier site) :
    wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site) =
      c3Zero := by
  simp only [localizedCarrierState, hexterior]
  funext component
  fin_cases component <;>
    simp [wedge, c3ScaleComplex, c3Zero, haloDirection, basis]

theorem localized_wedge_all_sites (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27) :
    wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude site) unitArea := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [localized_carrier_chi_at_carrier _ _ _ _ hcarrier,
      localized_carrier_psi_at_carrier _ _ _ _ hcarrier]
    rw [show localizedAreaCoefficient psiAmplitude chiAmplitude site =
        -(chiAmplitude * psiAmplitude *
          localizedTwistMinus (site27Corner site)) by
      simp [localizedAreaCoefficient, hcarrier]]
    rw [wedge_scale_left, wedge_scale_right, wedge_companion_tetra]
    funext component
    simp [c3ScaleComplex, c3Neg]
    ring
  · rw [localized_halo_wedge_zero chi0Value psiAmplitude chiAmplitude site
      hcarrier]
    funext component
    simp [localizedAreaCoefficient, hcarrier, c3ScaleComplex, c3Zero]

def LocalizedPsiNormal (eta : Site27 → C3) : Prop :=
  ∀ site, Site27IsCarrier site →
    c3Inner (tetraCorner (site27Corner site)) (eta site) = 0

def LocalizedChiNormal (eta : Site27 → C3) : Prop :=
  ∀ site, Site27IsCarrier site →
    c3Inner (companion (site27Corner site)) (eta site) = 0

theorem unit_area_wedge_halo_pair_zero (eta : C3) :
    c3Inner unitArea (wedge haloDirection eta) = 0 := by
  apply Complex.ext <;>
    simp [c3Inner, Fin.sum_univ_succ, unitArea, wedge,
      haloDirection, basis]

theorem unit_area_wedge_zero_right_pair_zero (eta : C3) :
    c3Inner unitArea (wedge eta c3Zero) = 0 := by
  apply Complex.ext <;>
    simp [c3Inner, Fin.sum_univ_succ, unitArea, wedge, c3Zero]

theorem localized_psi_normal_local_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedPsiNormal eta) (site : Site27) :
    c3RealPair
      ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site)
      (eta site) = 0 := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [c3_real_pair_eq_inner_re]
    rw [localized_carrier_psi_at_carrier _ _ _ _ hcarrier]
    rw [c3_inner_scale_left, hnormal site hcarrier]
    simp
  · simp [localizedCarrierState, hcarrier, c3RealPair, c3Zero]

theorem localized_chi_normal_local_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedChiNormal eta) (site : Site27)
    (hcarrier : Site27IsCarrier site) :
    c3RealPair
      ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site)
      (eta site) = 0 := by
  rw [c3_real_pair_eq_inner_re]
  rw [localized_carrier_chi_at_carrier _ _ _ _ hcarrier]
  rw [c3_inner_scale_left, hnormal site hcarrier]
  simp

theorem localized_psi_normal_wedge_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedPsiNormal eta) (site : Site27) :
    c3Inner unitArea
      (wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site)
        (eta site)) = 0 := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [localized_carrier_chi_at_carrier _ _ _ _ hcarrier]
    rw [wedge_scale_left]
    rw [show c3Inner unitArea
        (c3ScaleComplex chiAmplitude
          (wedge (companion (site27Corner site)) (eta site))) =
        chiAmplitude * c3Inner unitArea
          (wedge (companion (site27Corner site)) (eta site)) by
      unfold c3Inner c3ScaleComplex
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro component _
      ring,
      c3_inner_unitArea_wedge_companion, hnormal site hcarrier]
    simp
  · rw [localized_carrier_chi_at_halo _ _ _ _ hcarrier]
    rw [wedge_scale_left]
    rw [show c3Inner unitArea
        (c3ScaleComplex (chi0Value : ℂ) (wedge haloDirection (eta site))) =
        (chi0Value : ℂ) * c3Inner unitArea
          (wedge haloDirection (eta site)) by
      unfold c3Inner c3ScaleComplex
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro component _
      ring,
      unit_area_wedge_halo_pair_zero]
    simp

theorem localized_chi_normal_wedge_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedChiNormal eta) (site : Site27) :
    c3Inner unitArea
      (wedge (eta site)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site)) =
      0 := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [localized_carrier_psi_at_carrier _ _ _ _ hcarrier]
    rw [wedge_scale_right]
    rw [show c3Inner unitArea
        (c3ScaleComplex
          (psiAmplitude * localizedTwistMinus (site27Corner site))
          (wedge (eta site) (tetraCorner (site27Corner site)))) =
        (psiAmplitude * localizedTwistMinus (site27Corner site)) *
          c3Inner unitArea
            (wedge (eta site) (tetraCorner (site27Corner site))) by
      unfold c3Inner c3ScaleComplex
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro component _
      ring,
      c3_inner_unitArea_wedge_tetra, hnormal site hcarrier]
    simp
  · rw [localized_carrier_psi_at_halo _ _ _ _ hcarrier]
    exact unit_area_wedge_zero_right_pair_zero (eta site)

theorem localized_psi_normal_bond_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedPsiNormal eta) (edge : Bond Site27) :
    c3RealPair
      (bondAreaDifference
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude) edge)
      (bondAreaFirstVariation
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (purePsiDirection eta) edge) = 0 := by
  rw [c3_real_pair_eq_inner_re]
  unfold bondAreaDifference bondAreaFirstVariation
  rw [localized_wedge_all_sites, localized_wedge_all_sites]
  rw [show wedgeFirstVariation
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (purePsiDirection eta) edge.source =
      wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi
          edge.source) (eta edge.source) by
      funext component
      fin_cases component <;>
        simp [wedgeFirstVariation, purePsiDirection, c3Add, c3Zero, wedge],
    show wedgeFirstVariation
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (purePsiDirection eta) edge.target =
      wedge
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi
          edge.target) (eta edge.target) by
      funext component
      fin_cases component <;>
        simp [wedgeFirstVariation, purePsiDirection, c3Add, c3Zero, wedge]]
  rw [show difference
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
          localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring]
  rw [c3_inner_scale_left_general, c3_inner_sub_right,
    localized_psi_normal_wedge_pair_zero chi0Value psiAmplitude chiAmplitude
      eta hnormal edge.source,
    localized_psi_normal_wedge_pair_zero chi0Value psiAmplitude chiAmplitude
      eta hnormal edge.target]
  simp

theorem localized_chi_normal_bond_pair_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedChiNormal eta) (edge : Bond Site27) :
    c3RealPair
      (bondAreaDifference
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude) edge)
      (bondAreaFirstVariation
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (pureChiDirection eta) edge) = 0 := by
  rw [c3_real_pair_eq_inner_re]
  unfold bondAreaDifference bondAreaFirstVariation
  rw [localized_wedge_all_sites, localized_wedge_all_sites]
  rw [show wedgeFirstVariation
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (pureChiDirection eta) edge.source =
      wedge (eta edge.source)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi
          edge.source) by
      funext component
      fin_cases component <;>
        simp [wedgeFirstVariation, pureChiDirection, c3Add, c3Zero, wedge],
    show wedgeFirstVariation
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (pureChiDirection eta) edge.target =
      wedge (eta edge.target)
        ((localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi
          edge.target) by
      funext component
      fin_cases component <;>
        simp [wedgeFirstVariation, pureChiDirection, c3Add, c3Zero, wedge]]
  rw [show difference
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea)
      (c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target)
        unitArea) =
      c3ScaleComplex
        (localizedAreaCoefficient psiAmplitude chiAmplitude edge.target -
          localizedAreaCoefficient psiAmplitude chiAmplitude edge.source)
        unitArea by
      funext component
      simp [difference, c3ScaleComplex]
      ring]
  rw [c3_inner_scale_left_general, c3_inner_sub_right,
    localized_chi_normal_wedge_pair_zero chi0Value psiAmplitude chiAmplitude
      eta hnormal edge.source,
    localized_chi_normal_wedge_pair_zero chi0Value psiAmplitude chiAmplitude
      eta hnormal edge.target]
  simp

theorem localized_psi_normal_site_variation_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedPsiNormal eta) (site : Site27) :
    sitePotentialFirstVariation chi0Value
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (purePsiDirection eta) site = 0 := by
  simp only [sitePotentialFirstVariation, purePsiDirection]
  rw [localized_psi_normal_local_pair_zero chi0Value psiAmplitude chiAmplitude
    eta hnormal site]
  simp [c3RealPair, c3Zero]

theorem localized_chi_normal_site_variation_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedChiNormal eta) (site : Site27) :
    sitePotentialFirstVariation chi0Value
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (pureChiDirection eta) site = 0 := by
  by_cases hcarrier : Site27IsCarrier site
  · simp only [sitePotentialFirstVariation, pureChiDirection]
    rw [localized_chi_normal_local_pair_zero chi0Value psiAmplitude
      chiAmplitude eta hnormal site hcarrier]
    simp [c3RealPair, c3Zero]
  · unfold sitePotentialFirstVariation
    rw [show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).psi site =
        c3Zero by
      exact localized_carrier_psi_at_halo chi0Value psiAmplitude chiAmplitude
        site hcarrier,
      show
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude).chi site =
        c3ScaleComplex (chi0Value : ℂ) haloDirection by
      exact localized_carrier_chi_at_halo chi0Value psiAmplitude chiAmplitude
        site hcarrier]
    simp [pureChiDirection, density,
      Fin.sum_univ_succ, c3ScaleComplex, haloDirection, basis,
      Complex.normSq, c3RealPair, c3Zero]
    left
    ring

theorem localized_psi_normal_force_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedPsiNormal eta) :
    psiEulerLagrangeForce chi0Value
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry) eta = 0 := by
  unfold psiEulerLagrangeForce lagrangianStateFirstVariation
  simp only [potentialFirstVariation]
  rw [show (∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (purePsiDirection eta) site) = 0 by
      apply Finset.sum_eq_zero
      intro site _
      exact localized_psi_normal_site_variation_zero chi0Value psiAmplitude
        chiAmplitude eta hnormal site]
  rw [show (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (purePsiDirection eta) (c19Bond site27C19Geometry edge)) = 0 by
      apply Finset.sum_eq_zero
      intro edge _
      unfold exteriorAreaBondFirstVariation
      rw [localized_psi_normal_bond_pair_zero chi0Value psiAmplitude
        chiAmplitude eta hnormal (c19Bond site27C19Geometry edge)]
      ring]
  ring

theorem localized_chi_normal_force_zero (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (eta : Site27 → C3)
    (hnormal : LocalizedChiNormal eta) :
    chiEulerLagrangeForce chi0Value
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry) eta = 0 := by
  unfold chiEulerLagrangeForce lagrangianStateFirstVariation
  simp only [potentialFirstVariation]
  rw [show (∑ site : Site27,
      sitePotentialFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (pureChiDirection eta) site) = 0 by
      apply Finset.sum_eq_zero
      intro site _
      exact localized_chi_normal_site_variation_zero chi0Value psiAmplitude
        chiAmplitude eta hnormal site]
  rw [show (∑ edge : C19Edge Site27,
      exteriorAreaBondFirstVariation chi0Value
        (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
        (pureChiDirection eta) (c19Bond site27C19Geometry edge)) = 0 by
      apply Finset.sum_eq_zero
      intro edge _
      unfold exteriorAreaBondFirstVariation
      rw [localized_chi_normal_bond_pair_zero chi0Value psiAmplitude
        chiAmplitude eta hnormal (c19Bond site27C19Geometry edge)]
      ring]
  ring

def singleSiteField27 (selected : Site27) (z : C3) : Site27 → C3 :=
  fun site => if site = selected then z else c3Zero

theorem localized_single_site_psi_normal (selected : Site27) (z : C3)
    (_hcarrier : Site27IsCarrier selected)
    (hnormal : ComplexNormalTo (tetraCorner (site27Corner selected)) z) :
    LocalizedPsiNormal (singleSiteField27 selected z) := by
  intro site hsite
  by_cases hs : site = selected
  · subst site
    simpa [singleSiteField27, ComplexNormalTo] using hnormal
  · simp [singleSiteField27, hs, c3Inner, c3Zero]

theorem localized_single_site_chi_normal (selected : Site27) (z : C3)
    (_hcarrier : Site27IsCarrier selected)
    (hnormal : ComplexNormalTo (companion (site27Corner selected)) z) :
    LocalizedChiNormal (singleSiteField27 selected z) := by
  intro site hsite
  by_cases hs : site = selected
  · subst site
    simpa [singleSiteField27, ComplexNormalTo] using hnormal
  · simp [singleSiteField27, hs, c3Inner, c3Zero]

theorem exterior_real_direction_psi_normal (selected : Site27)
    (component : Fin 3) (hexterior : ¬ Site27IsCarrier selected) :
    LocalizedPsiNormal (realCoordinateDirection selected component) := by
  intro site hcarrier
  have hne : site ≠ selected := by
    intro heq
    subst site
    exact hexterior hcarrier
  simp [realCoordinateDirection, hne, c3Inner]

theorem exterior_imag_direction_psi_normal (selected : Site27)
    (component : Fin 3) (hexterior : ¬ Site27IsCarrier selected) :
    LocalizedPsiNormal (imagCoordinateDirection selected component) := by
  intro site hcarrier
  have hne : site ≠ selected := by
    intro heq
    subst site
    exact hexterior hcarrier
  simp [imagCoordinateDirection, hne, c3Inner]

theorem exterior_real_direction_chi_normal (selected : Site27)
    (component : Fin 3) (hexterior : ¬ Site27IsCarrier selected) :
    LocalizedChiNormal (realCoordinateDirection selected component) := by
  intro site hcarrier
  have hne : site ≠ selected := by
    intro heq
    subst site
    exact hexterior hcarrier
  simp [realCoordinateDirection, hne, c3Inner]

theorem exterior_imag_direction_chi_normal (selected : Site27)
    (component : Fin 3) (hexterior : ¬ Site27IsCarrier selected) :
    LocalizedChiNormal (imagCoordinateDirection selected component) := by
  intro site hcarrier
  have hne : site ≠ selected := by
    intro heq
    subst site
    exact hexterior hcarrier
  simp [imagCoordinateDirection, hne, c3Inner]

theorem psi_momentum_single_site27
    (velocity : Velocity Site27) (selected : Site27) (z : C3) :
    psiMomentumPair velocity (singleSiteField27 selected z) =
      c3RealPair (velocity.psiDot selected) z := by
  unfold psiMomentumPair
  rw [Finset.sum_eq_single selected]
  · simp [singleSiteField27]
  · intro other _ hne
    simp [singleSiteField27, hne, c3RealPair, c3Zero]
  · simp

theorem chi_momentum_single_site27
    (B : ℝ) (velocity : Velocity Site27) (selected : Site27) (z : C3) :
    chiMomentumPair B velocity (singleSiteField27 selected z) =
      B * c3RealPair (velocity.chiDot selected) z := by
  unfold chiMomentumPair
  rw [Finset.sum_eq_single selected]
  · simp [singleSiteField27]
  · intro other _ hne
    simp [singleSiteField27, hne, c3RealPair, c3Zero]
  · simp

theorem psi_momentum_localized_ray (velocity : Velocity Site27)
    (selected : CubeCorner) (value : ℂ) :
    psiMomentumPair velocity
        (localizedPsiRayDirection
          (singleCornerAmplitude selected value)).psi =
      c3RealPair (velocity.psiDot (cornerSite27 selected))
        (c3ScaleComplex value (tetraCorner selected)) := by
  unfold psiMomentumPair
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · rw [localized_psi_ray_direction_single]
    simp
  · intro site _ hne
    rw [localized_psi_ray_direction_single]
    simp [hne]
  · simp

theorem chi_momentum_localized_ray (B : ℝ) (velocity : Velocity Site27)
    (selected : CubeCorner) (value : ℂ) :
    chiMomentumPair B velocity
        (localizedChiRayDirection
          (singleCornerAmplitude selected value)).chi =
      B * c3RealPair (velocity.chiDot (cornerSite27 selected))
        (c3ScaleComplex value (companion selected)) := by
  unfold chiMomentumPair
  rw [Finset.sum_eq_single (cornerSite27 selected)]
  · rw [localized_chi_ray_direction_single]
    simp
  · intro site _ hne
    rw [localized_chi_ray_direction_single]
    simp [hne]
  · simp

def localizedPsiCommonAcceleration (chi0Value : ℝ)
    (psiAmplitude chiAmplitude : ℂ) : ℂ :=
  -(localizedPsiStiffness chi0Value psiAmplitude chiAmplitude : ℂ) *
    psiAmplitude

def localizedChiCommonAcceleration (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) : ℂ :=
  -((localizedChiStiffness chi0Value psiAmplitude chiAmplitude / B : ℝ) : ℂ) *
    chiAmplitude

def localizedPsiAccelerationRayCoefficient (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) : ℂ :=
  c3Inner (tetraCorner corner)
    ((accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).psiDot (cornerSite27 corner))

def localizedChiAccelerationAmplitude (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) : ℂ :=
  c3Inner (companion corner)
    ((accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).chiDot (cornerSite27 corner))

theorem localized_psi_acceleration_on_carrier_ray (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).psiDot (cornerSite27 corner) =
      c3ScaleComplex
        (localizedPsiAccelerationRayCoefficient chi0Value B
          psiAmplitude chiAmplitude corner)
        (tetraCorner corner) := by
  let acceleration := accelerationVelocity chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry)
  let residual := rayResidual (tetraCorner corner)
    (acceleration.psiDot (cornerSite27 corner))
  have hnormal : ComplexNormalTo (tetraCorner corner) residual := by
    exact ray_residual_normal _ _ (tetra_corner_density_one corner)
  have hall := psi_acceleration_all_directions chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry)
    (singleSiteField27 (cornerSite27 corner) residual)
  rw [localized_psi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (singleSiteField27 (cornerSite27 corner) residual)
      (localized_single_site_psi_normal (cornerSite27 corner) residual
        (corner_site27_is_carrier corner) (by simpa using hnormal))] at hall
  rw [psi_momentum_single_site27] at hall
  have hpair : c3RealPair
      (acceleration.psiDot (cornerSite27 corner)) residual = 0 := hall
  rw [show residual = rayResidual (tetraCorner corner)
      (acceleration.psiDot (cornerSite27 corner)) by rfl,
    c3_real_pair_with_residual _ _ (tetra_corner_density_one corner)] at hpair
  have hresidual := density_zero_implies_c3_zero residual hpair
  funext component
  have hc := congrFun hresidual component
  simp [residual, rayResidual, c3Sub, c3Zero, c3ScaleComplex,
    localizedPsiAccelerationRayCoefficient, acceleration] at hc ⊢
  exact sub_eq_zero.mp hc

theorem localized_chi_acceleration_on_carrier_ray (chi0Value B : ℝ)
    (hB : B ≠ 0) (psiAmplitude chiAmplitude : ℂ)
    (corner : CubeCorner) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).chiDot (cornerSite27 corner) =
      c3ScaleComplex
        (localizedChiAccelerationAmplitude chi0Value B
          psiAmplitude chiAmplitude corner)
        (companion corner) := by
  let acceleration := accelerationVelocity chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry)
  let residual := rayResidual (companion corner)
    (acceleration.chiDot (cornerSite27 corner))
  have hnormal : ComplexNormalTo (companion corner) residual := by
    exact ray_residual_normal _ _ (companion_density_one corner)
  have hall := chi_acceleration_all_directions chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry) hB
    (singleSiteField27 (cornerSite27 corner) residual)
  rw [localized_chi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (singleSiteField27 (cornerSite27 corner) residual)
      (localized_single_site_chi_normal (cornerSite27 corner) residual
        (corner_site27_is_carrier corner) (by simpa using hnormal))] at hall
  rw [chi_momentum_single_site27] at hall
  have hpair : c3RealPair
      (acceleration.chiDot (cornerSite27 corner)) residual = 0 := by
    exact (mul_eq_zero.mp hall).resolve_left hB
  rw [show residual = rayResidual (companion corner)
      (acceleration.chiDot (cornerSite27 corner)) by rfl,
    c3_real_pair_with_residual _ _ (companion_density_one corner)] at hpair
  have hresidual := density_zero_implies_c3_zero residual hpair
  funext component
  have hc := congrFun hresidual component
  simp [residual, rayResidual, c3Sub, c3Zero, c3ScaleComplex,
    localizedChiAccelerationAmplitude, acceleration] at hc ⊢
  exact sub_eq_zero.mp hc

theorem localized_psi_acceleration_pair (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude value : ℂ) (corner : CubeCorner) :
    c3RealPair
        ((accelerationVelocity chi0Value B
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (c19Bond site27C19Geometry)).psiDot (cornerSite27 corner))
        (c3ScaleComplex value (tetraCorner corner)) =
      -localizedPsiStiffness chi0Value psiAmplitude chiAmplitude *
        (star (psiAmplitude * localizedTwistMinus corner) * value).re := by
  have h := psi_acceleration_all_directions chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry)
    (localizedPsiRayDirection
      (singleCornerAmplitude corner value)).psi
  rw [psi_momentum_localized_ray,
    localized_psi_force_single_ray] at h
  exact h

theorem localized_chi_acceleration_pair (chi0Value B : ℝ)
    (hB : B ≠ 0) (psiAmplitude chiAmplitude value : ℂ)
    (corner : CubeCorner) :
    B * c3RealPair
        ((accelerationVelocity chi0Value B
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (c19Bond site27C19Geometry)).chiDot (cornerSite27 corner))
        (c3ScaleComplex value (companion corner)) =
      -localizedChiStiffness chi0Value psiAmplitude chiAmplitude *
        (star chiAmplitude * value).re := by
  have h := chi_acceleration_all_directions chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry) hB
    (localizedChiRayDirection
      (singleCornerAmplitude corner value)).chi
  rw [chi_momentum_localized_ray,
    localized_chi_force_single_ray] at h
  exact h

theorem localized_psi_acceleration_exact (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (corner : CubeCorner) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).psiDot (cornerSite27 corner) =
      c3ScaleComplex
        (localizedPsiCommonAcceleration chi0Value psiAmplitude chiAmplitude *
          localizedTwistMinus corner)
        (tetraCorner corner) := by
  rw [localized_psi_acceleration_on_carrier_ray]
  congr 1
  apply Complex.ext
  · have h := localized_psi_acceleration_pair chi0Value B
      psiAmplitude chiAmplitude 1 corner
    rw [localized_psi_acceleration_on_carrier_ray,
      c3_real_pair_scaled_unit _ _ _ (tetra_corner_density_one corner)] at h
    simp [localizedPsiCommonAcceleration, Complex.mul_re,
      Complex.mul_im] at h ⊢
    rw [h]
    ring
  · have h := localized_psi_acceleration_pair chi0Value B
      psiAmplitude chiAmplitude I corner
    rw [localized_psi_acceleration_on_carrier_ray,
      c3_real_pair_scaled_unit _ _ _ (tetra_corner_density_one corner)] at h
    simp [localizedPsiCommonAcceleration, Complex.mul_re,
      Complex.mul_im] at h ⊢
    rw [h]
    ring

theorem localized_chi_acceleration_exact (chi0Value B : ℝ)
    (hB : B ≠ 0) (psiAmplitude chiAmplitude : ℂ)
    (corner : CubeCorner) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).chiDot (cornerSite27 corner) =
      c3ScaleComplex
        (localizedChiCommonAcceleration chi0Value B
          psiAmplitude chiAmplitude)
        (companion corner) := by
  rw [localized_chi_acceleration_on_carrier_ray chi0Value B hB]
  congr 1
  apply Complex.ext
  · have h := localized_chi_acceleration_pair chi0Value B hB
      psiAmplitude chiAmplitude 1 corner
    rw [localized_chi_acceleration_on_carrier_ray chi0Value B hB,
      c3_real_pair_scaled_unit _ _ _ (companion_density_one corner)] at h
    simp [localizedChiCommonAcceleration, Complex.mul_re] at h ⊢
    field_simp [hB]
    nlinarith [h]
  · have h := localized_chi_acceleration_pair chi0Value B hB
      psiAmplitude chiAmplitude I corner
    rw [localized_chi_acceleration_on_carrier_ray chi0Value B hB,
      c3_real_pair_scaled_unit _ _ _ (companion_density_one corner)] at h
    simp [localizedChiCommonAcceleration, Complex.mul_re,
      Complex.mul_im] at h ⊢
    field_simp [hB]
    nlinarith [h]

theorem localized_psi_acceleration_zero_on_halo (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hexterior : ¬ Site27IsCarrier site) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).psiDot site = c3Zero := by
  funext component
  change psiAcceleration chi0Value
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry) site component = 0
  unfold psiAcceleration
  rw [localized_psi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (realCoordinateDirection site component)
      (exterior_real_direction_psi_normal site component hexterior),
    localized_psi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (imagCoordinateDirection site component)
      (exterior_imag_direction_psi_normal site component hexterior)]
  simp

theorem localized_chi_acceleration_zero_on_halo (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27)
    (hexterior : ¬ Site27IsCarrier site) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).chiDot site = c3Zero := by
  funext component
  change chiAcceleration chi0Value B
    (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
    (c19Bond site27C19Geometry) site component = 0
  unfold chiAcceleration
  rw [localized_chi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (realCoordinateDirection site component)
      (exterior_real_direction_chi_normal site component hexterior),
    localized_chi_normal_force_zero chi0Value psiAmplitude chiAmplitude
      (imagCoordinateDirection site component)
      (exterior_imag_direction_chi_normal site component hexterior)]
  simp

theorem localized_psi_acceleration_all_sites (chi0Value B : ℝ)
    (psiAmplitude chiAmplitude : ℂ) (site : Site27) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).psiDot site =
      if Site27IsCarrier site then
        c3ScaleComplex
          (localizedPsiCommonAcceleration chi0Value psiAmplitude chiAmplitude *
            localizedTwistMinus (site27Corner site))
          (tetraCorner (site27Corner site))
      else c3Zero := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [if_pos hcarrier]
    calc
      _ = (accelerationVelocity chi0Value B
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (c19Bond site27C19Geometry)).psiDot
            (cornerSite27 (site27Corner site)) :=
        congrArg _ (corner_site27_site27_corner site hcarrier).symm
      _ = _ := localized_psi_acceleration_exact chi0Value B psiAmplitude
        chiAmplitude (site27Corner site)
  · rw [if_neg hcarrier]
    exact localized_psi_acceleration_zero_on_halo chi0Value B
      psiAmplitude chiAmplitude site hcarrier

theorem localized_chi_acceleration_all_sites (chi0Value B : ℝ)
    (hB : B ≠ 0) (psiAmplitude chiAmplitude : ℂ) (site : Site27) :
    (accelerationVelocity chi0Value B
      (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
      (c19Bond site27C19Geometry)).chiDot site =
      if Site27IsCarrier site then
        c3ScaleComplex
          (localizedChiCommonAcceleration chi0Value B
            psiAmplitude chiAmplitude)
          (companion (site27Corner site))
      else c3Zero := by
  by_cases hcarrier : Site27IsCarrier site
  · rw [if_pos hcarrier]
    calc
      _ = (accelerationVelocity chi0Value B
          (localizedCarrierState chi0Value psiAmplitude chiAmplitude)
          (c19Bond site27C19Geometry)).chiDot
            (cornerSite27 (site27Corner site)) :=
        congrArg _ (corner_site27_site27_corner site hcarrier).symm
      _ = _ := localized_chi_acceleration_exact chi0Value B hB psiAmplitude
        chiAmplitude (site27Corner site)
  · rw [if_neg hcarrier]
    exact localized_chi_acceleration_zero_on_halo chi0Value B
      psiAmplitude chiAmplitude site hcarrier

theorem native_next_preserves_localized_carrier_sector
    (chi0Value B dt : ℝ) (hB : B ≠ 0)
    (previous current : State Site27)
    (hprevious : InLocalizedCarrierSector chi0Value previous)
    (hcurrent : InLocalizedCarrierSector chi0Value current) :
    InLocalizedCarrierSector chi0Value
      (nativeNextState chi0Value B dt (c19Bond site27C19Geometry)
        previous current) := by
  rcases hprevious with ⟨previousPsi, previousChi, rfl⟩
  rcases hcurrent with ⟨currentPsi, currentChi, rfl⟩
  let nextPsi : ℂ :=
    2 * currentPsi - previousPsi +
      (dt^2 : ℂ) *
        localizedPsiCommonAcceleration chi0Value currentPsi currentChi
  let nextChi : ℂ :=
    2 * currentChi - previousChi +
      (dt^2 : ℂ) *
        localizedChiCommonAcceleration chi0Value B currentPsi currentChi
  refine ⟨nextPsi, nextChi, ?_⟩
  apply congrArg₂ State.mk <;> funext site component
  · rw [show
      (accelerationVelocity chi0Value B
        (localizedCarrierState chi0Value currentPsi currentChi)
        (c19Bond site27C19Geometry)).psiDot site =
        if Site27IsCarrier site then
          c3ScaleComplex
            (localizedPsiCommonAcceleration chi0Value currentPsi currentChi *
              localizedTwistMinus (site27Corner site))
            (tetraCorner (site27Corner site))
        else c3Zero by
      exact localized_psi_acceleration_all_sites chi0Value B
        currentPsi currentChi site]
    by_cases hcarrier : Site27IsCarrier site
    · simp [localizedCarrierState, hcarrier, nextPsi,
        c3ScaleComplex]
      ring
    · simp [localizedCarrierState, hcarrier, c3Zero]
  · rw [show
      (accelerationVelocity chi0Value B
        (localizedCarrierState chi0Value currentPsi currentChi)
        (c19Bond site27C19Geometry)).chiDot site =
        if Site27IsCarrier site then
          c3ScaleComplex
            (localizedChiCommonAcceleration chi0Value B currentPsi currentChi)
            (companion (site27Corner site))
        else c3Zero by
      exact localized_chi_acceleration_all_sites chi0Value B hB
        currentPsi currentChi site]
    by_cases hcarrier : Site27IsCarrier site
    · simp [localizedCarrierState, hcarrier, nextChi,
        c3ScaleComplex]
      ring
    · simp [localizedCarrierState, hcarrier, c3Zero]
      ring

end
end LeanDirac

