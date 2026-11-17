module

public import LeanDirac.Dirac009Geometry

@[expose] public section

namespace LeanDirac

/-!
# DIRAC-009: descaffolded continuum Clifford gate

The native law remains the positive two-field C19 action admitted in
DIRAC-002 and evolved through DIRAC-008. The cubical Clifford operators are
introduced only as a square-root readout of its isotropic low-momentum symbol.
The exact finite-lattice mismatch is retained with an explicit fourth-order
bound. No signed-incidence, shape-holding, or Dirac transport energy is added
to the action.
-/

structure Dirac009ClosedCertificate : Prop where
  candidateAction : Dirac002Certificate
  throughFirstOrderMap : ∀ (dt : ℝ) (hdt : dt ≠ 0),
    Dirac008Certificate dt hdt
  geometry : Dirac009NecessaryGeometry

theorem dirac009_closed_native_clifford :
    Dirac009ClosedCertificate where
  candidateAction := dirac002_candidate_action_and_geometry
  throughFirstOrderMap := dirac008_complete_native_first_order_map
  geometry := dirac009_necessary_cube_geometry

end LeanDirac

#print axioms LeanDirac.dirac009_closed_native_clifford
