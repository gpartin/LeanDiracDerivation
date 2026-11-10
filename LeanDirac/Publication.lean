import LeanDirac.Dirac005

namespace LeanDirac

/-!
Sole public proof root. DIRAC-005 imports the explicit DIRAC-001 through 004
gate modules and is the highest currently admitted mathematical gate.  The
public certificate is cumulative: it retains DIRAC-004's nonzero-step action
update theorem as well as DIRAC-005's invariant-sector theorem.
-/

structure CumulativeDirac005Certificate
    (dt : ℝ) (previous current : State CubeCorner) : Prop where
  discreteActionUpdate :
    DiscreteActionUpdateCertificate chi0 chiKineticNormalization dt
      (c19Bond cubeC19Geometry) previous current
  invariantUnitCellSector :
    Dirac005Certificate chi0 chiKineticNormalization dt
      (c19Bond cubeC19Geometry) previous current
  coordinateCount : Fintype.card NativeUnitCellCoordinate = 16
  cornerCoordinates : Function.Injective cubeCornerCoordinate
  c19UnitCellModel : ∀ corner direction,
    cubeCornerCoordinate (cubeC19Neighbor corner direction) =
      periodicC19Neighbor 2 (cubeCornerCoordinate corner) direction
  c19DirectionClasses : ∀ direction,
    (direction.val < 6 ∧ c19NonzeroAxisCount direction = 1 ∧
      c19Weight direction = 2) ∨
    (6 ≤ direction.val ∧ c19NonzeroAxisCount direction = 2 ∧
      c19Weight direction = 1)

theorem current_highest_gate_is_dirac005
    (dt : ℝ) (previous current : State CubeCorner) (hdt : dt ≠ 0) :
    CumulativeDirac005Certificate dt previous current where
  discreteActionUpdate :=
    dirac004_c19_update_iff_discrete_action_stationary
      dt cubeC19Geometry previous current hdt
  invariantUnitCellSector :=
    dirac005_c19_native_sector dt previous current
  coordinateCount := native_unit_cell_coordinate_count
  cornerCoordinates := cube_corner_coordinate_injective
  c19UnitCellModel := cube_c19_neighbor_is_periodic_c19
  c19DirectionClasses := c19_direction_has_declared_class

end LeanDirac

