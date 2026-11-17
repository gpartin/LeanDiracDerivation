module

public import Mathlib

public section

/-!
# Native-action restriction and recurrence-derived cubical Dirac limit

This is the mathematical statement surface submitted to Palomar. It states a
connected result for the prepared, uniform-chi tangent sector of a specified
positive two-field cubic-lattice action:

1. the full native action restricts exactly to the aligned eight-site sector,
   and first variation commutes with that restriction;
2. the restricted native action has a massive C19 Klein--Gordon tangent;
3. the exact C19 symbol is its isotropic quadratic part plus an explicit
   finite-spacing remainder;
4. signed cube-incidence operations give a first-order operator whose square
   is that quadratic continuum symbol;
5. a first-order branch is derived from the actual centered recurrence, its
   exact native defect is the C19 spatial remainder, and its generator tends
   to the derived Dirac Hamiltonian; and
6. the same tangent bond action supplies an exact local phase-continuity law.

The prepared background and low-momentum/small-timestep hypotheses are
visible in the statements.  These results do not assert a localized electron,
physical spin, nonlinear live-chi closure, uniqueness of the square root, or
an exact finite-spacing Dirac equation.
-/

namespace LFMDiracChallenge

noncomputable section

open BigOperators Complex

/-! ## Restricted native action and the exact C19 symbol -/

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

/-- The advertised two-field action, not a separately invented modal action,
restricts exactly to the aligned eight-site amplitudes. -/
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
  sorry

def cell8RealLine (base direction : Cell8) (eps : ℝ) : Cell8 :=
  fun corner => base corner + (eps : ℂ) * direction corner

/-- Since the equality holds on every real line through the retained fields,
the first variation of the full action and that of its restriction coincide. -/
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
  sorry

/-- Exact C19 Fourier symbol: six face neighbors have relative weight two
and twelve edge neighbors have relative weight one. -/
def c19TransportSymbol (momentum : LatticeMomentum) : ℝ :=
  8 * ((1 - Real.cos (momentum 0)) +
    (1 - Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 1)) +
    (1 - Real.cos (momentum 0) * Real.cos (momentum 2)) +
    (1 - Real.cos (momentum 1) * Real.cos (momentum 2)))

/-- Isotropic quadratic part of the C19 symbol. -/
def c19QuadraticSymbol (momentum : LatticeMomentum) : ℝ :=
  12 * (momentum 0 ^ 2 + momentum 1 ^ 2 + momentum 2 ^ 2)

/-- Exact finite-spacing spatial remainder. -/
def c19SpatialRemainder (momentum : LatticeMomentum) : ℝ :=
  c19TransportSymbol momentum - c19QuadraticSymbol momentum

/-- Local density part of the native positive potential. -/
def nativeSitePotential (rhoPsi rhoChi : ℝ) : ℝ :=
  (rhoChi - chi0^2)^2 / 4 + rhoChi * rhoPsi / 2 + rhoPsi^2 / 4

/-- Local action restricted to a uniform-chi Fourier perturbation. -/
def preparedLocalPotential (eps : ℝ) (amplitude : ℂ) : ℝ :=
  nativeSitePotential
    (Complex.normSq ((eps : ℂ) * amplitude)) (chi0^2)

/-- C19 bond action restricted to the same Fourier perturbation. -/
def preparedBondPotential
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) : ℝ :=
  Complex.normSq ((eps : ℂ) * amplitude) / 2 *
    c19TransportSymbol momentum

/-- Complete prepared-mode restriction of the local-plus-bond potential. -/
def nativePreparedModalPotential
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) : ℝ :=
  preparedLocalPotential eps amplitude +
    preparedBondPotential eps amplitude momentum

/-- Exact quadratic tangent coefficient supplied by the restricted action. -/
def nativePsiTangentSymbol (momentum : LatticeMomentum) : ℝ :=
  chi0^2 + c19TransportSymbol momentum

/-- The specified positive action supplies the massive C19 tangent, while its
quartic self-interaction remains explicit. -/
theorem action_to_tangent_symbol
    (eps : ℝ) (amplitude : ℂ) (momentum : LatticeMomentum) :
    nativePreparedModalPotential eps amplitude momentum =
      eps^2 * Complex.normSq amplitude / 2 *
          nativePsiTangentSymbol momentum +
        eps^4 * Complex.normSq amplitude ^ 2 / 4 := by
  sorry

/-- Exact decomposition, not an asymptotic replacement. -/
theorem exact_c19_symbol_split (momentum : LatticeMomentum) :
    nativePsiTangentSymbol momentum =
      chi0^2 + c19QuadraticSymbol momentum +
        c19SpatialRemainder momentum := by
  sorry

/-- The explicit spatial correction is fourth order at low momentum. -/
theorem c19_spatial_remainder_control
    (momentum : LatticeMomentum) (radius : ℝ)
    (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (hsmall : 2 * radius ≤ 1) :
    |c19SpatialRemainder momentum| ≤ 40 * radius^4 := by
  sorry

/-! ## Cubical first-order factor -/

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

/-- The signed cube-incidence operator squares to the same isotropic
quadratic symbol that occurs in the native tangent. -/
theorem cubical_dirac_square
    (mass : ℂ) (momentum : LatticeMomentum) (amplitude : Cell8) :
    c19ContinuumDiracSymbol mass momentum
        (c19ContinuumDiracSymbol mass momentum amplitude) =
      c19ContinuumScalarSymbol mass momentum amplitude := by
  sorry

/-! ## First-order branch derived from the centered recurrence -/

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

/-- This branch is not a trial exponential: its coefficients are the
characteristic roots of the centered recurrence. -/
theorem recurrence_branch_solves_centered_update
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4)
    (hstable : (dt * continuumModeFrequency momentum)^2 ≤ 4) :
    recurrenceDiracBranchStep dt momentum
        (recurrenceDiracBranchStep dt momentum z) =
      fun component =>
        2 * recurrenceDiracBranchStep dt momentum z component - z component -
          ((dt^2 * continuumModeSquare momentum : ℝ) : ℂ) *
            recurrenceDiracBranchStep dt momentum z component := by
  sorry

def nativeBranchRecurrenceDefect
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  fun component =>
    recurrenceDiracBranchStep dt momentum
        (recurrenceDiracBranchStep dt momentum z) component -
      2 * recurrenceDiracBranchStep dt momentum z component + z component +
      ((dt^2 * nativePsiTangentSymbol momentum : ℝ) : ℂ) *
        recurrenceDiracBranchStep dt momentum z component

/-- In the exact native C19 recurrence the complete remaining defect is
precisely the explicit finite-spacing spatial remainder. -/
theorem native_branch_defect_is_c19_remainder
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4)
    (hstable : (dt * continuumModeFrequency momentum)^2 ≤ 4) :
    nativeBranchRecurrenceDefect dt momentum z =
      fun component =>
        ((dt^2 * c19SpatialRemainder momentum : ℝ) : ℂ) *
          recurrenceDiracBranchStep dt momentum z component := by
  sorry

def recurrenceBranchGeneratorDefect
    (dt : ℝ) (momentum : LatticeMomentum) (z : Cell4) : Cell4 :=
  fun component =>
    (recurrenceDiracBranchStep dt momentum z component - z component) /
        (dt : ℂ) +
      I * standardC19Hamiltonian (chi0 : ℂ) momentum z component

/-- On the punctured timestep domain, the recurrence-derived branch has the
derived Dirac Hamiltonian as its infinitesimal generator. -/
theorem recurrence_branch_generator_tends_to_dirac
    (momentum : LatticeMomentum) (z : Cell4) (component : Fin 4) :
    Filter.Tendsto
      (fun dt : ℝ => recurrenceBranchGeneratorDefect dt momentum z component)
      (nhdsWithin 0 ({0} : Set ℝ)ᶜ) (nhds 0) := by
  sorry

/-! ## Action-derived local continuity -/

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

/-- Exact local continuity for the tangent equation generated by the same
quadratic C19 bond action. -/
theorem native_tangent_local_continuity [Fintype Site]
    (massSq : ℝ) (neighbor : Site → C19Neighbor → Site)
    (position velocity : Site → ℂ) (site : Site) :
    nativePhaseDensityDerivative (position site) (velocity site)
        (scalarTangentAcceleration massSq neighbor position site) +
      nativeOutwardCurrentDivergence neighbor position site = 0 := by
  sorry

end

end LFMDiracChallenge
