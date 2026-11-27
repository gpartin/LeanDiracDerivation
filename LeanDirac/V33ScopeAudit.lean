import LeanDirac.V33FreePropagator

/-!
# V3.3 Stage 9: scope and provenance audit

This certificate keeps the new observable package tied to the unchanged V3.2
chain.  The exclusions documented in the release plan are claim restrictions,
not propositions manufactured inside Lean.
-/

namespace LeanDirac

structure V33ScopeAuditCertificate : Prop where
  unchangedFramework : V33FrameworkLockCertificate
  inheritedFreeTheory : V32CompleteFreeFermionCertificate
  basis : V33SpinorBasisCertificate
  branches : V33ProjectedSpinorCertificate
  nativeMassOnly : chi0 = 19

theorem v33_scope_audit : V33ScopeAuditCertificate where
  unchangedFramework := v33_framework_lock
  inheritedFreeTheory := v32_complete_free_fermion
  basis := v33_spinor_basis
  branches := v33_projected_spinors
  nativeMassOnly := v32_native_rest_mass_is_nineteen

#print axioms v33_scope_audit

end LeanDirac
