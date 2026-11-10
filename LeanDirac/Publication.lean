import LeanDirac.Dirac004

namespace LeanDirac

/-!
Sole public proof root. DIRAC-004 imports the explicit DIRAC-001/002/003 gate
modules and is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac004
    [Fintype Site] [DecidableEq Site]
    (dt : ℝ) (geometry : C19Lattice Site)
    (previous current : State Site) (hdt : dt ≠ 0) :
    NativeUpdateCertificate chi0 chiKineticNormalization dt
      (c19Bond geometry) previous current :=
  dirac004_c19_reversible_action_update dt geometry previous current hdt

end LeanDirac

