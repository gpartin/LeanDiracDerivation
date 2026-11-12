import Lax423666.CubicalC19

namespace Lax423666Proofs

open Lax423666.CubicalC19

noncomputable section

private theorem firstOrderSymbol_square
    (mass k0 k1 k2 : ℂ) (amplitude : Cell8) :
    firstOrderSymbol mass k0 k1 k2
        (firstOrderSymbol mass k0 k1 k2 amplitude) =
      fun corner =>
        (mass ^ 2 + k0 ^ 2 + k1 ^ 2 + k2 ^ 2) * amplitude corner := by
  funext corner
  fin_cases corner <;>
    simp [firstOrderSymbol, parityBeta, incidenceLinear, incidenceGamma,
      incidenceSign, incidenceFlip, cellFlip0, cellFlip1, cellFlip2,
      cellBit0, cellBit1, cellBit2] <;>
    ring

private theorem c19ContinuumSpeed_square : c19ContinuumSpeed ^ 2 = 12 := by
  norm_num [c19ContinuumSpeed]

/--
---
conclusion: Lax423666.CubicalC19.cubicalC19Factorization
---
The proof expands the eight cube corners.  Cubical orientation makes each
signed flip square to the identity while every mixed pair cancels.  Cube
parity cancels the mass--transport cross terms.  Finally the C19 second-moment
normalization turns the three surviving directional squares into the isotropic
quadratic symbol.
-/
theorem cubicalC19Factorization_proof
    (mass : ℂ) (momentum : Fin 3 → ℝ) (amplitude : Cell8) :
    c19FirstOrderSymbol mass momentum
        (c19FirstOrderSymbol mass momentum amplitude) =
      c19QuadraticSymbol mass momentum amplitude := by
  simp only [c19FirstOrderSymbol]
  rw [firstOrderSymbol_square]
  funext corner
  simp only [c19QuadraticSymbol]
  have hc : ((c19ContinuumSpeed : ℝ) : ℂ) ^ 2 = 12 := by
    exact_mod_cast c19ContinuumSpeed_square
  rw [mul_pow, mul_pow, mul_pow, hc]
  ring

end

end Lax423666Proofs
