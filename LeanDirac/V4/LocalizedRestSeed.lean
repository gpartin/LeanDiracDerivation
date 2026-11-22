/-
Copyright (c) 2026 G. D. Partin. All rights reserved.
Released under MIT license as described in the file LICENSE.
Authors: G. D. Partin
-/

module

public import LeanDirac.V2.LorentzSpin
public import LeanDirac.V2.RestOrbit

@[expose] public section

namespace LeanDirac.V4

/-!
# V4-001: exact localized rest seed

This module is the first gate of the localized-electron program.  It ports the
previously proved compact `Psi`--`chi` carrier to the public V3 action and keeps
its claim separate from the V3 free-field Lorentz/spin theorem.

The result below proves an exact, freely updated, fixed-center native orbit on
every finite periodic C19 box of side at least three.  It does not identify the
localized carrier with either V3 four-component spin sector, prove stability,
or prove translation.  Those are later V4 gates.
-/

open LeanDirac.V2

noncomputable section

structure ExactLocalizedRestSeedCertificate
    (L : ℕ) [NeZero L] (r B dt : ℝ) : Prop where
  nativeOrbit :
    let q : ℕ → State (PeriodicCubicSite L) :=
      fun n => boxLift (balancedRest27 r B dt n)
    (∀ n, q (n + 2) =
      nativeNextState r B dt (c19Bond (periodicC19Geometry L))
        (q n) (q (n + 1))) ∧
      (∀ n site, imbalance B (q n) (q (n + 1)) site = 0)
  phasesAdvance :
    0 < (harmonicPhaseStep dt (balancedPsiFrequencySq r B dt)).im ∧
      0 < (harmonicPhaseStep dt (balancedChiFrequencySq r B dt)).im

theorem exact_localized_rest_seed
    {L : ℕ} [NeZero L] (hL : 3 ≤ L)
    (r B dt : ℝ) (hr : 0 < r) (hB : 1 < B)
    (hdt : 0 < dt) (hd : dt^2 * r^2 / 4 < 1) :
    ExactLocalizedRestSeedCertificate L r B dt where
  nativeOrbit := balanced_rest_orbit hL r B dt hr hB hd
  phasesAdvance := balanced_phases_advance r B dt hr hB hdt hd

/-- V4 changes no V3 result: the free continuum Lorentz/spin certificate
remains independently available.  This theorem intentionally does not assert
that its spinor is already the localized rest seed above. -/
theorem prior_free_lorentz_spin_certificate :
    LeanDirac.V2.LorentzSpinHalfCertificate :=
  LeanDirac.V2.lorentz_spin_half_from_native_action

end
end LeanDirac.V4

#print axioms LeanDirac.V4.exact_localized_rest_seed
#print axioms LeanDirac.V4.prior_free_lorentz_spin_certificate

