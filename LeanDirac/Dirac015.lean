import LeanDirac.Dirac013

namespace LeanDirac

/-!
# DIRAC-015: final free-Dirac continuum theorem

This file packages the already-proved chain into one explicit theorem.  Its
central statement is the action-to-Klein--Gordon-to-Dirac relationship:

1. the native two-field action supplies the quadratic massive C19 wave
   symbol on the declared prepared background;
2. that exact symbol splits into a continuum Klein--Gordon symbol and the
   retained finite-lattice remainder;
3. the cubical four-component Hamiltonian squares to that same continuum
   Klein--Gordon symbol;
4. the actual centered native recurrence approaches the first-order Dirac
   branch with the previously proved spatial and temporal error bound; and
5. the same action and embedding supply the admitted density/current limit.

This theorem is conditional on the declared prepared background and the
low-momentum, small-timestep hypotheses written in its residual field.  It
does not assert a localized electron, electromagnetic coupling, nonlinear
live-chi closure, uniqueness of the square root, or canonical LFM status.
-/

noncomputable section

structure LFMFreeDiracContinuumCertificate : Prop where
  completePriorChain : Dirac013Certificate
  kleinGordonTangentFromAction : ∀ eps amplitude momentum site,
    nativePreparedModalPotential eps amplitude momentum site =
      eps^2 * Complex.normSq amplitude / 2 *
          nativePsiTangentSymbol momentum +
        eps^4 * Complex.normSq amplitude^2 / 4
  exactNativeSymbolSplits : ∀ momentum,
    nativePsiTangentSymbol momentum =
      chi0^2 + c19QuadraticSymbol momentum +
        c19SpatialRemainder momentum
  diracSquareIsContinuumKleinGordon : ∀ momentum z,
    standardC19Hamiltonian (chi0 : ℂ) momentum
        (standardC19Hamiltonian (chi0 : ℂ) momentum z) =
      fun component =>
        ((chi0^2 + c19QuadraticSymbol momentum : ℝ) : ℂ) * z component
  controlledCenteredEquationLimit : ∀ dt frequency radius momentum,
    dt ≠ 0 →
    0 ≤ radius →
    (∀ axis, |momentum axis| ≤ radius) →
    2 * radius ≤ 1 →
    |dt * frequency| ≤ 1 →
    frequency^2 = chi0^2 + c19QuadraticSymbol momentum →
    |normalizedNativeDiracModeResidual dt frequency momentum| ≤
      (5/48 : ℝ) * |dt * frequency|^4 / dt^2 + 40 * radius^4
  nativeContinuityAndDiracCurrent : Dirac013Certificate

theorem lfm_free_dirac_continuum_limit :
    LFMFreeDiracContinuumCertificate where
  completePriorChain := dirac013_native_density_current_continuity
  kleinGordonTangentFromAction := native_prepared_modal_potential_expansion
  exactNativeSymbolSplits := native_psi_tangent_symbol_is_exact_scalar_coefficient
  diracSquareIsContinuumKleinGordon :=
    standard_c19_hamiltonian_square_is_continuum_dispersion
  controlledCenteredEquationLimit :=
    normalized_native_dirac_mode_residual_bound
  nativeContinuityAndDiracCurrent :=
    dirac013_native_density_current_continuity

end
end LeanDirac

#print axioms LeanDirac.lfm_free_dirac_continuum_limit
