module

public import LeanDirac.Dirac015

public section

namespace LFMDiracChallenge

noncomputable section

open BigOperators Complex

def chi0 : ℝ := 19

abbrev LatticeMomentum := Fin 3 → ℝ

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

abbrev CubeCorner := Fin 8
abbrev Cell8 := CubeCorner → ℂ

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

def centeredTimeNumerator (dt frequency : ℝ) : ℝ :=
  2 * (1 - Real.cos (dt * frequency))

def centeredTimeRemainder (dt frequency : ℝ) : ℝ :=
  centeredTimeNumerator dt frequency - (dt * frequency)^2

def nativeDiracModeResidual
    (dt frequency : ℝ) (momentum : LatticeMomentum) : ℝ :=
  centeredTimeNumerator dt frequency -
    dt^2 * nativePsiTangentSymbol momentum

theorem controlled_centered_dirac_residual
    (dt frequency radius : ℝ) (momentum : LatticeMomentum)
    (hradius : 0 ≤ radius)
    (hmomentum : ∀ axis, |momentum axis| ≤ radius)
    (hspace : 2 * radius ≤ 1)
    (htime : |dt * frequency| ≤ 1)
    (hdispersion : frequency^2 =
      chi0^2 + c19QuadraticSymbol momentum) :
    |nativeDiracModeResidual dt frequency momentum| ≤
      (5/48 : ℝ) * |dt * frequency|^4 +
        dt^2 * (40 * radius^4) := by
  have htimeBound :
      |centeredTimeRemainder dt frequency| ≤
        (5/48 : ℝ) * |dt * frequency|^4 := by
    change |LeanDirac.centeredTimeRemainder dt frequency| ≤
      (5/48 : ℝ) * |dt * frequency|^4
    exact LeanDirac.centered_time_remainder_bound dt frequency htime
  have hspaceBound := c19_spatial_remainder_control momentum radius
    hradius hmomentum hspace
  have hsplit :
      nativeDiracModeResidual dt frequency momentum =
        centeredTimeRemainder dt frequency -
          dt^2 * c19SpatialRemainder momentum := by
    unfold nativeDiracModeResidual centeredTimeRemainder
      nativePsiTangentSymbol c19SpatialRemainder
    nlinarith
  rw [hsplit]
  calc
    |centeredTimeRemainder dt frequency -
        dt^2 * c19SpatialRemainder momentum| ≤
      |centeredTimeRemainder dt frequency| +
        |dt^2 * c19SpatialRemainder momentum| := abs_sub _ _
    _ ≤ (5/48 : ℝ) * |dt * frequency|^4 +
        dt^2 * (40 * radius^4) := by
      rw [abs_mul, abs_of_nonneg (sq_nonneg dt)]
      exact add_le_add htimeBound
        (mul_le_mul_of_nonneg_left hspaceBound (sq_nonneg dt))

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
