import LeanDirac.Dirac007

namespace LeanDirac

/-!
Sole public proof root through DIRAC-007.  The public certificate is explicitly
cumulative: it retains the native framework, positive candidate action and
prepared geometry, arbitrary local action variations, nonzero-step discrete
action update, invariant unit-cell sector and native repeated-state transport
symbol, then proves an exact lossless change from the two committed layers to
the current state and its native discrete velocity.
-/

structure CumulativeDirac005Certificate
    (dt : ℝ) (previous current : State CubeCorner) : Prop where
  nativeFramework :
    chi0 = 19 ∧
    kappa = 1 / 63 ∧
    chiKineticNormalization = 1197 ∧
    current = ⟨current.psi, current.chi⟩ ∧
    ∀ neighbor : C19Neighbor, 0 < c19Weight neighbor
  candidateActionAndGeometry : Dirac002Certificate
  localActionVariations : ∀ velocity : Velocity CubeCorner,
    LocalTwoFieldVariationCertificate chi0 chiKineticNormalization
      current velocity (c19Bond cubeC19Geometry)
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
  nativeFramework := dirac001_native_framework current
  candidateActionAndGeometry := dirac002_candidate_action_and_geometry
  localActionVariations := fun velocity =>
    dirac003_c19_local_variations_from_actual_action
      chi0 chiKineticNormalization current velocity cubeC19Geometry
  discreteActionUpdate :=
    dirac004_c19_update_iff_discrete_action_stationary
      dt cubeC19Geometry previous current hdt
  invariantUnitCellSector :=
    dirac005_c19_native_sector dt previous current
  coordinateCount := native_unit_cell_coordinate_count
  cornerCoordinates := cube_corner_coordinate_injective
  c19UnitCellModel := cube_c19_neighbor_is_periodic_c19
  c19DirectionClasses := c19_direction_has_declared_class

structure CumulativeDirac006Certificate
    (dt : ℝ) (previous current : State CubeCorner) : Prop where
  throughDirac005 : CumulativeDirac005Certificate dt previous current
  nativeTransportSymbol : Dirac006Certificate

theorem current_highest_gate_is_dirac006
    (dt : ℝ) (previous current : State CubeCorner) (hdt : dt ≠ 0) :
    CumulativeDirac006Certificate dt previous current where
  throughDirac005 := current_highest_gate_is_dirac005
    dt previous current hdt
  nativeTransportSymbol := dirac006_native_transport_symbol

structure CumulativeDirac007Certificate
    (dt : ℝ) (hdt : dt ≠ 0)
    (previous current : State CubeCorner) : Prop where
  throughDirac006 : CumulativeDirac006Certificate dt previous current
  losslessNativeReadout : Dirac007Certificate dt hdt

theorem current_highest_gate_is_dirac007
    (dt : ℝ) (previous current : State CubeCorner) (hdt : dt ≠ 0) :
    CumulativeDirac007Certificate dt hdt previous current where
  throughDirac006 := current_highest_gate_is_dirac006
    dt previous current hdt
  losslessNativeReadout := dirac007_lossless_native_paired_readout dt hdt

end LeanDirac

