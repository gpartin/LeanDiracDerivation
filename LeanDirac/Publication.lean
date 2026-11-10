import LeanDirac.Dirac005

namespace LeanDirac

/-!
Sole public proof root. DIRAC-005 imports the explicit DIRAC-001 through 004
gate modules and is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac005
    (dt : ℝ) (previous current : State CubeCorner) :
    Dirac005Certificate chi0 chiKineticNormalization dt
      (c19Bond cubeC19Geometry) previous current :=
  dirac005_c19_native_sector dt previous current

end LeanDirac

