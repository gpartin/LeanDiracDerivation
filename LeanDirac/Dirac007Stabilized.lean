import LeanDirac.Dirac006Stabilized
import LeanDirac.Dirac007

namespace LeanDirac

/-!
The DIRAC-007 paired-layer readout is an invertible coordinate change on the
complete native state.  It does not depend on the potential, so the proof is
unchanged by the covariance stability term.  This wrapper records the action
lineage explicitly instead of relying on an old transitive import.
-/

structure StabilizedPairedReadoutCertificate
    (dt : ℝ) (hdt : dt ≠ 0) : Prop where
  stabilizedTransport : StabilizedTransportCertificate
  losslessReadout : Dirac007Certificate dt hdt

theorem dirac007_stabilized_lossless_paired_readout
    (dt : ℝ) (hdt : dt ≠ 0) :
    StabilizedPairedReadoutCertificate dt hdt where
  stabilizedTransport := dirac006_stabilized_native_transport
  losslessReadout := dirac007_lossless_native_paired_readout dt hdt

end LeanDirac
