import LeanDirac.LorentzSpin

open Complex
open LeanDirac

noncomputable section

/-!
Regression for the V3 Pauli--Lubanski sign repair.

The historical spatial definition used `E J + p x K`. Orthogonality and the
rest-frame square did not expose its failure. The concrete moving-frame
witness below proves that its infinitesimal boost defect is a nonzero boost
generator. The production definition uses `E J - p x K` and its all-axis
four-vector laws are checked at the end of this file.
-/

def historicalPlusPauliLubanskiSpace
    (energy : Complex) (momentum : Fin 3 -> Complex) (axis : Fin 3)
    (z : CubicalCell4) : CubicalCell4 :=
  fun component =>
    energy * standardCubicalSpin axis z component +
      cubicalMomentumCrossBoost momentum axis z component

def historicalPlusMomentum : Fin 3 -> Complex := ![1, 0, 0]

def historicalPlusBoostDefect
    (energy : Complex) (z : CubicalCell4) : CubicalCell4 :=
  fun component =>
    cubicalPauliLubanskiBoostVariationTime 2 energy z component +
      I * (standardCubicalBoost 2
          (cubicalPauliLubanskiTime historicalPlusMomentum z) component -
        cubicalPauliLubanskiTime historicalPlusMomentum
          (standardCubicalBoost 2 z) component) -
      historicalPlusPauliLubanskiSpace
        energy historicalPlusMomentum 2 z component

theorem historical_plus_sign_defect_is_nonzero_boost
    (energy : Complex) (z : CubicalCell4) :
    historicalPlusBoostDefect energy z =
      fun component => -2 * standardCubicalBoost 1 z component := by
  funext component
  fin_cases component <;>
    simp [historicalPlusBoostDefect, historicalPlusPauliLubanskiSpace,
      historicalPlusMomentum, cubicalPauliLubanskiBoostVariationTime,
      cubicalPauliLubanskiTime, cubicalMomentumBoostSpace,
      cubicalMomentumCrossBoost, standardCubicalBoost,
      standardCubicalSpin, standardCubicalPlaneBivector,
      standardCubicalGamma, cubicalStandardAlphaX,
      cubicalStandardAlphaY, cubicalStandardAlphaZ] <;>
    ring_nf <;> norm_num [pow_succ, Complex.I_sq] <;> ring

def historicalWitnessState : CubicalCell4 := ![1, 0, 0, 0]

theorem historical_plus_sign_fails_four_vector_law :
    historicalPlusBoostDefect 19 historicalWitnessState ≠ fun _ => 0 := by
  intro hzero
  rw [historical_plus_sign_defect_is_nonzero_boost] at hzero
  have hcomponent := congrFun hzero 3
  norm_num [historicalWitnessState, standardCubicalBoost,
    standardCubicalGamma, cubicalStandardAlphaY, Complex.I_sq] at hcomponent

#check cubical_pauli_lubanski_boost_covariant_time
#check cubical_pauli_lubanski_boost_covariant_space
#check cubical_pauli_lubanski_rotation_covariant_time
#check cubical_pauli_lubanski_rotation_covariant_space

#print axioms historical_plus_sign_fails_four_vector_law
#print axioms cubical_pauli_lubanski_boost_covariant_time
#print axioms cubical_pauli_lubanski_boost_covariant_space
#print axioms cubical_pauli_lubanski_rotation_covariant_time
#print axioms cubical_pauli_lubanski_rotation_covariant_space

#eval "LEAN_DIRAC_V3_PAULI_LUBANSKI_REGRESSION_PASS"
