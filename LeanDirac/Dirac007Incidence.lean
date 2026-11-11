import LeanDirac.Dirac006Incidence
import LeanDirac.Dirac007

namespace LeanDirac

/-!
DIRAC-007 is the lossless conversion between two native time layers and the
current state plus interval velocity.  It is independent of the potential;
this wrapper records the incidence-action lineage explicitly.
-/

structure IncidencePairedReadoutCertificate
    (dt : ℝ) (hdt : dt ≠ 0) : Prop where
  incidenceTransport : IncidenceTransportCertificate
  losslessReadout : Dirac007Certificate dt hdt

theorem dirac007_incidence_lossless_paired_readout
    (dt : ℝ) (hdt : dt ≠ 0) :
    IncidencePairedReadoutCertificate dt hdt where
  incidenceTransport := dirac006_incidence_native_transport
  losslessReadout := dirac007_lossless_native_paired_readout dt hdt

end LeanDirac
