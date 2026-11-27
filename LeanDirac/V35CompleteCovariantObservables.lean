import LeanDirac.V35FrameworkAndGapLock
import LeanDirac.V35AntiparticleResidueClosure
import LeanDirac.V35GeneratedLorentzTransformations
import LeanDirac.V35CovariantSolutionTransport
import LeanDirac.V35CompleteBilinears
import LeanDirac.V35BilinearCliffordClosure
import LeanDirac.V35MovingSpinInvariant

/-!
# V3.5 Stage 8: complete covariant-observable closure

The certificate packages the theorem-only V3.5 extension.  The transport
field is algebraic covariance of free solutions under the derived finite
boost; it is not a claim about finite packets or nonlinear evolution.
-/

namespace LeanDirac

structure V35CompleteCovariantObservableCertificate : Prop where
  framework : V35FrameworkAndGapLockCertificate
  prior : V34CompleteCovariantFreeSpinorCertificate
  antiparticleResidue : V35AntiparticleResidueClosureCertificate
  generatedTransformations : V35GeneratedLorentzTransformationCertificate
  covariantSolutions : V35CovariantSolutionTransportCertificate
  completeBilinears : V35CompleteBilinearCertificate
  bilinearClosure : V35BilinearCliffordClosureCertificate
  movingSpin : V35MovingSpinInvariantCertificate

theorem v35_complete_covariant_observables :
    V35CompleteCovariantObservableCertificate where
  framework := v35_framework_and_gap_lock
  prior := v34_complete_covariant_free_spinor
  antiparticleResidue := v35_antiparticle_residue_closure
  generatedTransformations := v35_generated_lorentz_transformations
  covariantSolutions := v35_covariant_solution_transport
  completeBilinears := v35_complete_bilinears
  bilinearClosure := v35_bilinear_clifford_closure
  movingSpin := v35_moving_spin_invariant

#print axioms v35_complete_covariant_observables

end LeanDirac
