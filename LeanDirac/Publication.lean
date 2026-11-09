import LeanDirac.Dirac002

namespace LeanDirac

/-!
Sole public proof root. DIRAC-001 is enforced by this one-way import chain;
DIRAC-002 is the highest currently admitted mathematical gate.
-/

theorem current_highest_gate_is_dirac002 :
    (∀ corner : CubeCorner,
      wedge (tetraCorner corner) (companion corner) = unitArea) :=
  tetra_wedge_companion_constant

end LeanDirac

