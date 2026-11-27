import LeanDirac.V33ScopeAudit

/-!
# V3.3 Stage 10: cumulative free-spinor observable certificate
-/

namespace LeanDirac

structure V33CompleteFreeSpinorCertificate : Prop where
  framework : V33FrameworkLockCertificate
  basis : V33SpinorBasisCertificate
  projectedSpinors : V33ProjectedSpinorCertificate
  spinSums : V33SpinSumCertificate
  observables : V33BilinearObservableCertificate
  helicity : V33HelicityCertificate
  forwardGordon : V33ForwardGordonCertificate
  freePropagator : V33FreePropagatorCertificate
  scope : V33ScopeAuditCertificate

theorem v33_complete_free_spinor : V33CompleteFreeSpinorCertificate where
  framework := v33_framework_lock
  basis := v33_spinor_basis
  projectedSpinors := v33_projected_spinors
  spinSums := v33_spin_sums
  observables := v33_bilinear_observables
  helicity := v33_helicity
  forwardGordon := v33_forward_gordon
  freePropagator := v33_free_propagator
  scope := v33_scope_audit

#print axioms v33_complete_free_spinor

end LeanDirac
