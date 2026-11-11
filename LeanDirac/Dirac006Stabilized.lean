import LeanDirac.Dirac005Stabilized
import LeanDirac.Dirac006

namespace LeanDirac

/-!
DIRAC-006 for the stabilized action.  The new covariance square depends only
on `chi`; it therefore cannot insert or alter the direct `Psi` C19 transport
symbol proved by the original DIRAC-006 calculation.
-/

noncomputable section

def replacePsi (state : State Site) (psi : Site → C3) : State Site where
  psi := psi
  chi := state.chi

theorem shape_energy_independent_of_psi [Fintype Site]
    (chi0 : ℝ) (state : State Site) (psi : Site → C3)
    (geometry : C19Lattice Site) :
    chiCovarianceShapeEnergy chi0 (replacePsi state psi) geometry =
      chiCovarianceShapeEnergy chi0 state geometry := by
  rfl

structure StabilizedTransportCertificate : Prop where
  nativeBaseTransport : Dirac006Certificate
  covarianceTermDoesNotAlterDirectPsiTransport :
    ∀ (chi0 : ℝ) (state : State CubeCorner) (psi : CubeCorner → C3),
      chiCovarianceShapeEnergy chi0 (replacePsi state psi) cubeC19Geometry =
        chiCovarianceShapeEnergy chi0 state cubeC19Geometry

theorem dirac006_stabilized_native_transport :
    StabilizedTransportCertificate where
  nativeBaseTransport := dirac006_native_transport_symbol
  covarianceTermDoesNotAlterDirectPsiTransport :=
    fun chi0 state psi =>
      shape_energy_independent_of_psi chi0 state psi cubeC19Geometry

end
end LeanDirac
