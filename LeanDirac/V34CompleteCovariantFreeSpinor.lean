import LeanDirac.V34SpinorSectorBasis
import LeanDirac.V34CovariantSpinSums
import LeanDirac.V34MomentumHelicity
import LeanDirac.V34NonforwardGordon
import LeanDirac.V34PropagatorResidues

/-!
# V3.4 Stage 7: complete covariant free-spinor closure

This certificate packages the cumulative V3.4 handoffs.  It extends, rather
than replaces, the V3.2 free-fermion and V3.3 free-spinor certificates.  The
scope remains the free linearized sector of the declared LFM action.
-/

namespace LeanDirac

structure V34CompleteCovariantFreeSpinorCertificate : Prop where
  priorFreeSpinor : V33CompleteFreeSpinorCertificate
  covariantSpinors : V34CovariantSpinorCertificate
  sectorBasis : V34SpinorSectorBasisCertificate
  covariantSpinSums : V34CovariantSpinSumCertificate
  momentumHelicity : V34MomentumHelicityCertificate
  nonforwardGordon : V34NonforwardGordonCertificate
  propagatorResidues : V34PropagatorResidueCertificate

theorem v34_complete_covariant_free_spinor :
    V34CompleteCovariantFreeSpinorCertificate where
  priorFreeSpinor := v33_complete_free_spinor
  covariantSpinors := v34_covariant_spinors
  sectorBasis := v34_spinor_sector_basis
  covariantSpinSums := v34_covariant_spin_sums
  momentumHelicity := v34_momentum_helicity
  nonforwardGordon := v34_nonforward_gordon
  propagatorResidues := v34_propagator_residues

#print axioms v34_complete_covariant_free_spinor

end LeanDirac
