import LeanDirac.V2.V32ScaleIdentifiability
import LeanDirac.V2.V32LatticeArtifacts
import LeanDirac.V2.ReadoutCharacterization

/-! # V3.2 Stage 13: declared robustness and uniqueness scope -/

namespace LeanDirac.V2

noncomputable section

structure V32RobustnessCertificate : Prop where
  fullFreeChain : CompleteFreeDiracCertificate
  readoutScope : CubicalReadoutCharacterizationCertificate
  flavorScope : CubicalFlavorClassificationCertificate
  nativeMass : NativeRestMassCertificate
  artifacts : V32LatticeArtifactCertificate

theorem v32_robustness : V32RobustnessCertificate where
  fullFreeChain := complete_first_principles_free_dirac
  readoutScope := cubical_readout_characterization
  flavorScope := cubical_flavor_classification
  nativeMass := v32_native_rest_mass
  artifacts := v32_lattice_artifacts

#print axioms v32_robustness

end
end LeanDirac.V2
