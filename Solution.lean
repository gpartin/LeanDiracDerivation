module

public import LeanDirac.Dirac015

public section

namespace LFMDiracChallenge

noncomputable section

open BigOperators Complex

def chi0 : ℝ := 19

abbrev LatticeMomentum := Fin 3 → ℝ
abbrev CubeCorner := Fin 8
abbrev Cell8 := CubeCorner → ℂ
abbrev C3 := Fin 3 → ℂ

structure NativeBond where
  source : CubeCorner
  target : CubeCorner
  weight : ℝ

def c3Density (u : C3) : ℝ :=
  ∑ component, Complex.normSq (u component)

def c3Wedge (u v : C3) : C3
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | _ => u 0 * v 1 - u 1 * v 0

def c3Sub (u v : C3) : C3 := fun component => v component - u component

def c3Scale (a : ℂ) (u : C3) : C3 := fun component => a * u component

def alignedField (amplitude : Cell8) (ray : CubeCorner → C3) :
    CubeCorner → C3 :=
  fun corner => c3Scale (amplitude corner) (ray corner)

def nativeKinetic
    (B : ℝ) (psiVelocity chiVelocity : CubeCorner → C3) : ℝ :=
  (∑ corner, c3Density (psiVelocity corner)) / 2 +
    B * (∑ corner, c3Density (chiVelocity corner)) / 2

def nativeSiteEnergy
    (vacuum : ℝ) (psi chi : CubeCorner → C3) (corner : CubeCorner) : ℝ :=
  let rhoPsi := c3Density (psi corner)
  let rhoChi := c3Density (chi corner)
  (rhoChi - vacuum^2)^2 / 4 + rhoChi * rhoPsi / 2 + rhoPsi^2 / 4

def nativeBondEnergy
    (vacuum : ℝ) (bond : NativeBond)
    (psi chi : CubeCorner → C3) : ℝ :=
  bond.weight / (2 * vacuum^2) *
    c3Density (c3Sub
      (c3Wedge (chi bond.source) (psi bond.source))
      (c3Wedge (chi bond.target) (psi bond.target)))

def nativeLagrangian [Fintype Edge]
    (vacuum B : ℝ) (psi chi psiVelocity chiVelocity : CubeCorner → C3)
    (bond : Edge → NativeBond) : ℝ :=
  nativeKinetic B psiVelocity chiVelocity -
    ((∑ corner, nativeSiteEnergy vacuum psi chi corner) +
      ∑ edge, nativeBondEnergy vacuum (bond edge) psi chi)

def restrictedSiteEnergy
    (vacuum : ℝ) (psiAmplitude chiAmplitude : ℂ) : ℝ :=
  let rhoPsi := Complex.normSq psiAmplitude
  let rhoChi := Complex.normSq chiAmplitude
  (rhoChi - vacuum^2)^2 / 4 + rhoChi * rhoPsi / 2 + rhoPsi^2 / 4

def restrictedKinetic
    (B : ℝ) (psiVelocity chiVelocity : Cell8) : ℝ :=
  (∑ corner, Complex.normSq (psiVelocity corner)) / 2 +
    B * (∑ corner, Complex.normSq (chiVelocity corner)) / 2

def restrictedBondEnergy
    (vacuum weight : ℝ) (sourceProduct targetProduct : ℂ) : ℝ :=
  weight / (2 * vacuum^2) *
    Complex.normSq (targetProduct - sourceProduct)

def restrictedLagrangian [Fintype Edge]
    (vacuum B : ℝ)
    (psiAmplitude chiAmplitude psiVelocity chiVelocity : Cell8)
    (bond : Edge → NativeBond) : ℝ :=
  restrictedKinetic B psiVelocity chiVelocity -
    ((∑ corner, restrictedSiteEnergy vacuum
      (psiAmplitude corner) (chiAmplitude corner)) +
      ∑ edge, restrictedBondEnergy vacuum (bond edge).weight
        (chiAmplitude (bond edge).source * psiAmplitude (bond edge).source)
        (chiAmplitude (bond edge).target * psiAmplitude (bond edge).target))

theorem c3_density_scale (a : ℂ) (u : C3) :
    c3Density (c3Scale a u) = Complex.normSq a * c3Density u := by
  unfold c3Density c3Scale
  simp only [Complex.normSq_mul, Finset.mul_sum]

theorem c3_wedge_scale (a b : ℂ) (u v : C3) :
    c3Wedge (c3Scale a u) (c3Scale b v) =
      c3Scale (a * b) (c3Wedge u v) := by
  funext component
  fin_cases component <;> simp [c3Wedge, c3Scale] <;> ring

theorem aligned_density
    (amplitude : Cell8) (ray : CubeCorner → C3)
    (hray : ∀ corner, c3Density (ray corner) = 1) (corner : CubeCorner) :
    c3Density (alignedField amplitude ray corner) =
      Complex.normSq (amplitude corner) := by
  rw [show alignedField amplitude ray corner =
      c3Scale (amplitude corner) (ray corner) by rfl,
    c3_density_scale, hray, mul_one]

theorem aligned_wedge
    (psiAmplitude chiAmplitude : Cell8)
    (psiRay chiRay : CubeCorner → C3) (corner : CubeCorner) :
    c3Wedge (alignedField chiAmplitude chiRay corner)
        (alignedField psiAmplitude psiRay corner) =
      c3Scale (chiAmplitude corner * psiAmplitude corner)
        (c3Wedge (chiRay corner) (psiRay corner)) := by
  exact c3_wedge_scale _ _ _ _

theorem aligned_native_kinetic
    (B : ℝ) (psiRay chiRay : CubeCorner → C3)
    (hpsi : ∀ corner, c3Density (psiRay corner) = 1)
    (hchi : ∀ corner, c3Density (chiRay corner) = 1)
    (psiVelocity chiVelocity : Cell8) :
    nativeKinetic B (alignedField psiVelocity psiRay)
        (alignedField chiVelocity chiRay) =
      restrictedKinetic B psiVelocity chiVelocity := by
  unfold nativeKinetic restrictedKinetic
  simp_rw [aligned_density psiVelocity psiRay hpsi,
    aligned_density chiVelocity chiRay hchi]

theorem aligned_native_site_energy
    (vacuum : ℝ) (psiRay chiRay : CubeCorner → C3)
    (hpsi : ∀ corner, c3Density (psiRay corner) = 1)
    (hchi : ∀ corner, c3Density (chiRay corner) = 1)
    (psiAmplitude chiAmplitude : Cell8) (corner : CubeCorner) :
    nativeSiteEnergy vacuum (alignedField psiAmplitude psiRay)
        (alignedField chiAmplitude chiRay) corner =
      restrictedSiteEnergy vacuum
        (psiAmplitude corner) (chiAmplitude corner) := by
  unfold nativeSiteEnergy restrictedSiteEnergy
  dsimp only
  rw [aligned_density psiAmplitude psiRay hpsi,
    aligned_density chiAmplitude chiRay hchi]

theorem c3_sub_scaled_common (a b : ℂ) (u : C3) :
    c3Sub (c3Scale a u) (c3Scale b u) = c3Scale (b - a) u := by
  funext component
  simp [c3Sub, c3Scale]
  ring

theorem aligned_native_bond_energy
    (vacuum : ℝ) (psiRay chiRay : CubeCorner → C3) (areaRay : C3)
    (harea : ∀ corner, c3Wedge (chiRay corner) (psiRay corner) = areaRay)
    (hareaDensity : c3Density areaRay = 1)
    (psiAmplitude chiAmplitude : Cell8) (bond : NativeBond) :
    nativeBondEnergy vacuum bond
        (alignedField psiAmplitude psiRay)
        (alignedField chiAmplitude chiRay) =
      restrictedBondEnergy vacuum bond.weight
        (chiAmplitude bond.source * psiAmplitude bond.source)
        (chiAmplitude bond.target * psiAmplitude bond.target) := by
  unfold nativeBondEnergy restrictedBondEnergy
  rw [aligned_wedge, aligned_wedge, harea, harea,
    c3_sub_scaled_common, c3_density_scale, hareaDensity, mul_one]

theorem full_native_action_restricts_exactly [Fintype Edge]
    (vacuum B : ℝ) (psiRay chiRay : CubeCorner → C3) (areaRay : C3)
    (hpsi : ∀ corner, c3Density (psiRay corner) = 1)
    (hchi : ∀ corner, c3Density (chiRay corner) = 1)
    (harea : ∀ corner, c3Wedge (chiRay corner) (psiRay corner) = areaRay)
    (hareaDensity : c3Density areaRay = 1)
    (psiAmplitude chiAmplitude psiVelocity chiVelocity : Cell8)
    (bond : Edge → NativeBond) :
    nativeLagrangian vacuum B
        (alignedField psiAmplitude psiRay)
        (alignedField chiAmplitude chiRay)
        (alignedField psiVelocity psiRay)
        (alignedField chiVelocity chiRay) bond =
      restrictedLagrangian vacuum B psiAmplitude chiAmplitude
        psiVelocity chiVelocity bond := by
  unfold nativeLagrangian restrictedLagrangian
  rw [aligned_native_kinetic B psiRay chiRay hpsi hchi]
  congr 2
  · apply Finset.sum_congr rfl
    intro corner _
    exact aligned_native_site_energy vacuum psiRay chiRay hpsi hchi
      psiAmplitude chiAmplitude corner
  · apply Finset.sum_congr rfl
    intro edge _
    exact aligned_native_bond_energy vacuum psiRay chiRay areaRay
      harea hareaDensity psiAmplitude chiAmplitude (bond edge)

def cell8RealLine (base direction : Cell8) (eps : ℝ) : Cell8 :=
  fun corner => base corner + (eps : ℂ) * direction corner

theorem native_first_variation_commutes_with_restriction [Fintype Edge]
    (vacuum B : ℝ) (psiRay chiRay : CubeCorner → C3) (areaRay : C3)
    (hpsi : ∀ corner, c3Density (psiRay corner) = 1)
    (hchi : ∀ corner, c3Density (chiRay corner) = 1)
    (harea : ∀ corner, c3Wedge (chiRay corner) (psiRay corner) = areaRay)
    (hareaDensity : c3Density areaRay = 1)
    (psiAmplitude chiAmplitude psiVelocity chiVelocity : Cell8)
    (psiDirection chiDirection psiVelocityDirection chiVelocityDirection : Cell8)
    (bond : Edge → NativeBond) (derivative : ℝ) :
    HasDerivAt
      (fun eps : ℝ => nativeLagrangian vacuum B
        (alignedField (cell8RealLine psiAmplitude psiDirection eps) psiRay)
        (alignedField (cell8RealLine chiAmplitude chiDirection eps) chiRay)
        (alignedField (cell8RealLine psiVelocity psiVelocityDirection eps) psiRay)
        (alignedField (cell8RealLine chiVelocity chiVelocityDirection eps) chiRay)
        bond) derivative 0 ↔
    HasDerivAt
      (fun eps : ℝ => restrictedLagrangian vacuum B
        (cell8RealLine psiAmplitude psiDirection eps)
        (cell8RealLine chiAmplitude chiDirection eps)
        (cell8RealLine psiVelocity psiVelocityDirection eps)
        (cell8RealLine chiVelocity chiVelocityDirection eps)
        bond) derivative 0 := by
  have hfunctions :
      (fun eps : ℝ => nativeLagrangian vacuum B
        (alignedField (cell8RealLine psiAmplitude psiDirection eps) psiRay)
        (alignedField (cell8RealLine chiAmplitude chiDirection eps) chiRay)
        (alignedField (cell8RealLine psiVelocity psiVelocityDirection eps) psiRay)
        (alignedField (cell8RealLine chiVelocity chiVelocityDirection eps) chiRay)
        bond) =
      (fun eps : ℝ => restrictedLagrangian vacuum B
        (cell8RealLine psiAmplitude psiDirection eps)
        (cell8RealLine chiAmplitude chiDirection eps)
        (cell8RealLine psiVelocity psiVelocityDirection eps)
        (cell8RealLine chiVelocity chiVelocityDirection eps)
        bond) := by
    funext eps
    exact full_native_action_restricts_exactly vacuum B psiRay chiRay areaRay
      hpsi hchi harea hareaDensity _ _ _ _ bond
  rw [hfunctions]

def c19TransportSymbol (momentum : LatticeMomentum) : ℝ :=
  8 * ((1 - Real.cos (momentum 0)) +
    (1 - Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 1) * Real.cos (momentum 2)))

def c19QuadraticSymbol (momentum : LatticeMomentum) : ℝ :=
  12 * (momentum 0 ^ 2 + momentum 1 ^ 2 + momentum 2 ^ 2)

def c19SpatialRemainder (momentum : LatticeMomentum) : ℝ :=
  c19TransportSymbol momentum - c19QuadraticSymbol momentum

def nativeSitePotential (rhoPsi rhoChi : ℝ) : ℝ :=
  (rhoChi - chi0^2)^2 / 4 + rhoChi * rhoPsi / 2 + rhoPsi^2 / 4

def preparedLocalPotential (eps : ℝ) (amplitude : ℂ) : ℝ :=
  nativeSitePotential
    (Complex.normSq ((eps : ℂ) * amplitude)) (chi0^2)

def preparedBondPotential
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) : ℝ :=
  Complex.normSq ((eps : ℂ) * amplitude) / 2 *
    c19TransportSymbol momentum

def nativePreparedModalPotential
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) : ℝ :=
  preparedLocalPotential eps amplitude +
    preparedBondPotential eps amplitude momentum

def nativePsiTangentSymbol (momentum : LatticeMomentum) : ℝ :=
  chi0^2 + c19TransportSymbol momentum

theorem action_to_tangent_symbol
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) :
    nativePreparedModalPotential eps amplitude momentum =
      eps^2 * Complex.normSq amplitude / 2 *
          nativePsiTangentSymbol momentum +
        eps^4 * Complex.normSq amplitude ^ 2 / 4 := by
  simp [nativePreparedModalPotential, preparedLocalPotential,
    preparedBondPotential, nativeSitePotential, nativePsiTangentSymbol,
    chi0, Complex.normSq]
  ring

theorem exact_c19_symbol_split (momentum : LatticeMomentum) :
    nativePsiTangentSymbol momentum =
      chi0^2 + c19QuadraticSymbol momentum +
        c19SpatialRemainder momentum := by
  simp [nativePsiTangentSymbol, c19SpatialRemainder]

theorem c19_spatial_remainder_control
    (momentum : LatticeMomentum) (radius : ℝ)
    (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (hsmall : 2 * radius ≤ 1) :
    |c19SpatialRemainder momentum| ≤ 40 * radius^4 := by
  have hclosed :
      c19TransportSymbol momentum =
        LeanDirac.c19TransportSymbol momentum := by
    rw [LeanDirac.c19_transport_symbol_closed]
    rfl
  have hremainder :
      c19SpatialRemainder momentum =
        LeanDirac.c19SpatialRemainder momentum := by
    simp [c19SpatialRemainder, LeanDirac.c19SpatialRemainder,
      c19QuadraticSymbol, LeanDirac.c19QuadraticSymbol, hclosed]
  rw [hremainder]
  exact LeanDirac.c19_spatial_remainder_bound momentum radius hradius
    hmomentum hsmall

def cellFlip0 : CubeCorner → CubeCorner
  | 0 => 4 | 1 => 5 | 2 => 6 | 3 => 7
  | 4 => 0 | 5 => 1 | 6 => 2 | _ => 3

def cellFlip1 : CubeCorner → CubeCorner
  | 0 => 2 | 1 => 3 | 2 => 0 | 3 => 1
  | 4 => 6 | 5 => 7 | 6 => 4 | _ => 5

def cellFlip2 : CubeCorner → CubeCorner
  | 0 => 1 | 1 => 0 | 2 => 3 | 3 => 2
  | 4 => 5 | 5 => 4 | 6 => 7 | _ => 6

def cellBit0 : CubeCorner → Nat
  | 0 | 1 | 2 | 3 => 0
  | _ => 1

def cellBit1 : CubeCorner → Nat
  | 0 | 1 | 4 | 5 => 0
  | _ => 1

def cellBit2 : CubeCorner → Nat
  | 0 | 2 | 4 | 6 => 0
  | _ => 1

def cubeIncidenceFlip (axis : Fin 3) : CubeCorner → CubeCorner :=
  match axis with
  | 0 => cellFlip0
  | 1 => cellFlip1
  | _ => cellFlip2

def cubeIncidenceSign (axis : Fin 3) (corner : CubeCorner) : ℂ :=
  match axis with
  | 0 => 1
  | 1 => if cellBit0 corner = 0 then 1 else -1
  | _ => if (cellBit0 corner + cellBit1 corner) % 2 = 0 then 1 else -1

def cubeIncidenceGamma (axis : Fin 3) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    cubeIncidenceSign axis corner *
      amplitude (cubeIncidenceFlip axis corner)

def cubeParityBeta (amplitude : Cell8) : Cell8 :=
  fun corner =>
    if (cellBit0 corner + cellBit1 corner + cellBit2 corner) % 2 = 0 then
      amplitude corner
    else
      -amplitude corner

def cubeIncidenceLinear
    (k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    k0 * cubeIncidenceGamma 0 amplitude corner +
    k1 * cubeIncidenceGamma 1 amplitude corner +
    k2 * cubeIncidenceGamma 2 amplitude corner

def cubeDiracSymbol
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    mass * cubeParityBeta amplitude corner +
      cubeIncidenceLinear k0 k1 k2 amplitude corner

def c19ContinuumSpeed : ℝ := Real.sqrt 12

def c19ContinuumDiracSymbol
    (mass : ℂ) (momentum : LatticeMomentum) (amplitude : Cell8) : Cell8 :=
  cubeDiracSymbol mass
    (c19ContinuumSpeed * momentum 0)
    (c19ContinuumSpeed * momentum 1)
    (c19ContinuumSpeed * momentum 2) amplitude

def c19ContinuumScalarSymbol
    (mass : ℂ) (momentum : LatticeMomentum) (amplitude : Cell8) : Cell8 :=
  fun corner =>
    (mass ^ 2 + (12 : ℂ) *
      ((momentum 0 : ℂ) ^ 2 + (momentum 1 : ℂ) ^ 2 +
        (momentum 2 : ℂ) ^ 2)) * amplitude corner

theorem cubical_dirac_square
    (mass : ℂ) (momentum : LatticeMomentum) (amplitude : Cell8) :
    c19ContinuumDiracSymbol mass momentum
        (c19ContinuumDiracSymbol mass momentum amplitude) =
      c19ContinuumScalarSymbol mass momentum amplitude := by
  change LeanDirac.c19ContinuumDiracSymbol mass momentum
      (LeanDirac.c19ContinuumDiracSymbol mass momentum amplitude) =
    LeanDirac.c19ContinuumScalarSymbol mass momentum amplitude
  exact LeanDirac.c19_continuum_dirac_square_is_native_quadratic_symbol
    mass momentum amplitude

abbrev Cell4 := Fin 4 → ℂ

def standardDiracBeta (z : Cell4) : Cell4
  | 0 => z 0 | 1 => z 1 | 2 => -z 2 | _ => -z 3

def standardDiracAlphaX (z : Cell4) : Cell4
  | 0 => z 3 | 1 => z 2 | 2 => z 1 | _ => z 0

def standardDiracAlphaY (z : Cell4) : Cell4
  | 0 => -I * z 3 | 1 => I * z 2 | 2 => -I * z 1 | _ => I * z 0

def standardDiracAlphaZ (z : Cell4) : Cell4
  | 0 => z 2 | 1 => -z 3 | 2 => z 0 | _ => -z 1

def standardDiracHamiltonian
    (mass px py pz : ℂ) (z : Cell4) : Cell4 :=
  fun component =>
    mass * standardDiracBeta z component +
      (px * standardDiracAlphaX z component +
        py * standardDiracAlphaY z component +
        pz * standardDiracAlphaZ z component)

def standardC19Hamiltonian
    (mass : ℂ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  standardDiracHamiltonian mass
    (-((c19ContinuumSpeed * momentum 2 : ℝ) : ℂ))
    (-((c19ContinuumSpeed * momentum 1 : ℝ) : ℂ))
    (((c19ContinuumSpeed * momentum 0 : ℝ) : ℂ)) z

def continuumModeSquare (momentum : LatticeMomentum) : ℝ :=
  chi0^2 + c19QuadraticSymbol momentum

def continuumModeFrequency (momentum : LatticeMomentum) : ℝ :=
  Real.sqrt (continuumModeSquare momentum)

def recurrenceBranchCos (dt frequency : ℝ) : ℝ :=
  1 - (dt * frequency)^2 / 2

def recurrenceBranchRadical (dt frequency : ℝ) : ℝ :=
  Real.sqrt (1 - (dt * frequency)^2 / 4)

def recurrenceBranchSin (dt frequency : ℝ) : ℝ :=
  dt * frequency * recurrenceBranchRadical dt frequency

def cell4LinearCombination (a b : ℂ) (z w : Cell4) : Cell4 :=
  fun component => a * z component + b * w component

def recurrenceDiracBranchStep
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  let frequency := continuumModeFrequency momentum
  let cosine := recurrenceBranchCos dt frequency
  let sine := recurrenceBranchSin dt frequency
  cell4LinearCombination (cosine : ℂ)
    (-I * (sine / frequency : ℝ)) z
    (standardC19Hamiltonian (chi0 : ℂ) momentum z)

theorem recurrence_branch_solves_centered_update
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4)
    (hstable : (dt * continuumModeFrequency momentum)^2 ≤ 4) :
    recurrenceDiracBranchStep dt momentum
        (recurrenceDiracBranchStep dt momentum z) =
      fun component =>
        2 * recurrenceDiracBranchStep dt momentum z component - z component -
          ((dt^2 * continuumModeSquare momentum : ℝ) : ℂ) *
            recurrenceDiracBranchStep dt momentum z component := by
  change LeanDirac.recurrenceDiracBranchStep dt momentum
        (LeanDirac.recurrenceDiracBranchStep dt momentum z) =
      fun component =>
        2 * LeanDirac.recurrenceDiracBranchStep dt momentum z component -
          z component -
          ((dt^2 * LeanDirac.continuumModeSquare momentum : ℝ) : ℂ) *
            LeanDirac.recurrenceDiracBranchStep dt momentum z component
  exact
    LeanDirac.recurrence_dirac_branch_solves_continuum_centered_recurrence
      dt momentum z hstable

def nativeBranchRecurrenceDefect
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  fun component =>
    recurrenceDiracBranchStep dt momentum
        (recurrenceDiracBranchStep dt momentum z) component -
      2 * recurrenceDiracBranchStep dt momentum z component + z component +
      ((dt^2 * nativePsiTangentSymbol momentum : ℝ) : ℂ) *
        recurrenceDiracBranchStep dt momentum z component

theorem native_branch_defect_is_c19_remainder
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4)
    (hstable : (dt * continuumModeFrequency momentum)^2 ≤ 4) :
    nativeBranchRecurrenceDefect dt momentum z =
      fun component =>
        ((dt^2 * c19SpatialRemainder momentum : ℝ) : ℂ) *
          recurrenceDiracBranchStep dt momentum z component := by
  have hrecurrence :=
    recurrence_branch_solves_centered_update dt momentum z hstable
  funext component
  have hc := congrFun hrecurrence component
  unfold nativeBranchRecurrenceDefect
  rw [hc, exact_c19_symbol_split]
  unfold continuumModeSquare
  push_cast
  ring

def recurrenceBranchGeneratorDefect
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  fun component =>
    (recurrenceDiracBranchStep dt momentum z component - z component) /
        (dt : ℂ) +
      I * standardC19Hamiltonian (chi0 : ℂ) momentum z component

theorem recurrence_branch_generator_tends_to_dirac
    (momentum : LatticeMomentum) (z : Cell4) (component : Fin 4) :
    Filter.Tendsto
      (fun dt : ℝ => recurrenceBranchGeneratorDefect dt momentum z component)
      (nhdsWithin 0 ({0} : Set ℝ)ᶜ) (nhds 0) := by
  change Filter.Tendsto
      (fun dt : ℝ =>
        LeanDirac.recurrenceBranchGeneratorDefect dt momentum z component)
      (nhdsWithin 0 ({0} : Set ℝ)ᶜ) (nhds 0)
  exact LeanDirac.recurrence_branch_generator_defect_tends_to_zero
    momentum z component

abbrev C19Neighbor := Fin 18

def c19Weight (neighbor : C19Neighbor) : ℝ :=
  if neighbor.val < 6 then 2 else 1

def scalarPhasePair (source target : ℂ) : ℝ :=
  (star source * target).im

def nativeOutwardBondCurrent (weight : ℝ) (source target : ℂ) : ℝ :=
  -2 * weight * scalarPhasePair source target

def nativePhaseDensityDerivative
    (position velocity acceleration : ℂ) : ℝ :=
  scalarPhasePair velocity velocity + scalarPhasePair position acceleration

def scalarTangentAcceleration [Fintype Site]
    (massSq : ℝ) (neighbor : Site → C19Neighbor → Site)
    (position : Site → ℂ) (site : Site) : ℂ :=
  -(massSq : ℂ) * position site -
    2 * ∑ direction : C19Neighbor,
      (c19Weight direction : ℂ) *
        (position site - position (neighbor site direction))

def nativeOutwardCurrentDivergence [Fintype Site]
    (neighbor : Site → C19Neighbor → Site)
    (position : Site → ℂ) (site : Site) : ℝ :=
  ∑ direction : C19Neighbor,
    nativeOutwardBondCurrent (c19Weight direction)
      (position site) (position (neighbor site direction))

theorem native_tangent_local_continuity [Fintype Site]
    (massSq : ℝ) (neighbor : Site → C19Neighbor → Site)
    (position velocity : Site → ℂ) (site : Site) :
    nativePhaseDensityDerivative (position site) (velocity site)
        (scalarTangentAcceleration massSq neighbor position site) +
      nativeOutwardCurrentDivergence neighbor position site = 0 := by
  simp [nativePhaseDensityDerivative, scalarTangentAcceleration,
    nativeOutwardCurrentDivergence, nativeOutwardBondCurrent,
    scalarPhasePair, Complex.mul_im]
  ring_nf
  rw [Finset.mul_sum, Finset.mul_sum]
  rw [Finset.sum_mul, Finset.sum_mul]
  rw [← Finset.sum_neg_distrib]
  rw [← Finset.sum_add_distrib]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_eq_zero
  intro direction _
  ring

end

end LFMDiracChallenge
