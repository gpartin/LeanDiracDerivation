import LeanDirac.V32IrreducibleParticle

/-!
# V3.2 Stage 7: finite continuum Lorentz transformations

Finite rotations were derived in V3.1.  Here the finite axial boost is the
closed exponential of the already-derived boost generator: because the
transport generator squares to one, its exponential has the forced
cosh/sinh form.  No independent spin transformation is postulated.
-/

namespace LeanDirac

noncomputable section

def standardCubicalFiniteBoost
    (axis : Fin 3) (rapidity : ℝ) (z : CubicalCell4) : CubicalCell4 :=
  fun component =>
    (Real.cosh (rapidity / 2) : ℂ) * z component +
      (Real.sinh (rapidity / 2) : ℂ) *
        standardCubicalGamma axis z component

theorem standard_cubical_finite_boost_zero
    (axis : Fin 3) (z : CubicalCell4) :
    standardCubicalFiniteBoost axis 0 z = z := by
  funext component
  simp [standardCubicalFiniteBoost]

theorem standard_cubical_finite_boost_add
    (axis : Fin 3) (first second : ℝ) (z : CubicalCell4) :
    standardCubicalFiniteBoost axis first
        (standardCubicalFiniteBoost axis second z) =
      standardCubicalFiniteBoost axis (first + second) z := by
  have hc : (Real.cosh ((first + second) / 2) : ℂ) =
      (Real.cosh (first / 2) : ℂ) * (Real.cosh (second / 2) : ℂ) +
        (Real.sinh (first / 2) : ℂ) * (Real.sinh (second / 2) : ℂ) := by
    rw [show (first + second) / 2 = first / 2 + second / 2 by ring,
      Real.cosh_add]
    norm_num
  have hs : (Real.sinh ((first + second) / 2) : ℂ) =
      (Real.sinh (first / 2) : ℂ) * (Real.cosh (second / 2) : ℂ) +
        (Real.cosh (first / 2) : ℂ) * (Real.sinh (second / 2) : ℂ) := by
    rw [show (first + second) / 2 = first / 2 + second / 2 by ring,
      Real.sinh_add]
    norm_num
  funext component
  simp only [standardCubicalFiniteBoost]
  rw [hc, hs]
  fin_cases axis <;> fin_cases component
  all_goals
    simp [standardCubicalFiniteBoost, standardCubicalGamma,
      cubicalStandardAlphaX, cubicalStandardAlphaY,
      cubicalStandardAlphaZ]
  all_goals
    ring_nf
    try norm_num [Complex.I_sq]

theorem standard_cubical_finite_boost_inverse
    (axis : Fin 3) (rapidity : ℝ) (z : CubicalCell4) :
    standardCubicalFiniteBoost axis (-rapidity)
        (standardCubicalFiniteBoost axis rapidity z) = z := by
  rw [standard_cubical_finite_boost_add]
  simp [standard_cubical_finite_boost_zero]

structure V32FiniteLorentzCertificate : Prop where
  infinitesimalAndFiniteRotations : LorentzSpinHalfCertificate
  finiteBoostZero : ∀ axis z,
    standardCubicalFiniteBoost axis 0 z = z
  finiteBoostComposition : ∀ axis first second z,
    standardCubicalFiniteBoost axis first
        (standardCubicalFiniteBoost axis second z) =
      standardCubicalFiniteBoost axis (first + second) z
  finiteBoostInverse : ∀ axis rapidity z,
    standardCubicalFiniteBoost axis (-rapidity)
        (standardCubicalFiniteBoost axis rapidity z) = z
  infinitesimalBoostCovariance : ∀ axis energy mass momentum z,
    (fun component =>
      standardCubicalBoost axis
          (orientedCubicalDiracSymbol energy mass momentum z) component -
        orientedCubicalDiracSymbol energy mass momentum
          (standardCubicalBoost axis z) component) =
      fun component => Complex.I *
        (orientedCubicalDiracSymbol
          (cubicalMomentumBoostEnergy axis momentum)
          0 (cubicalMomentumBoostSpace axis energy)
          z) component

theorem v32_finite_lorentz : V32FiniteLorentzCertificate where
  infinitesimalAndFiniteRotations := lorentz_spin_half_from_native_action
  finiteBoostZero := standard_cubical_finite_boost_zero
  finiteBoostComposition := standard_cubical_finite_boost_add
  finiteBoostInverse := standard_cubical_finite_boost_inverse
  infinitesimalBoostCovariance := oriented_dirac_symbol_boost_covariant

#print axioms v32_finite_lorentz

end
end LeanDirac
