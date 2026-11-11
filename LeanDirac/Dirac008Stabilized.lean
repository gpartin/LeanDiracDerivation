import LeanDirac.Dirac004Stabilized
import LeanDirac.Dirac007Stabilized

namespace LeanDirac

/-!
DIRAC-008 rebuilt with the stabilized action.  The first-order state is the
lossless pair `(current native fields, backward interval velocity)`.  No
spinor or sector projection is inserted here.
-/

noncomputable section

def stabilizedLayerForward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) : NativeTwoLayerState Site where
  previous := layers.current
  current := stabilizedNativeNextState chi0 B dt geometry
    layers.previous layers.current

def stabilizedLayerBackward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) : NativeTwoLayerState Site where
  previous := stabilizedNativeNextState chi0 B dt geometry
    layers.current layers.previous
  current := layers.previous

theorem stabilized_layer_backward_after_forward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) :
    stabilizedLayerBackward chi0 B dt geometry
      (stabilizedLayerForward chi0 B dt geometry layers) = layers := by
  cases layers with
  | mk previous current =>
      apply congrArg₂ NativeTwoLayerState.mk
      · exact stabilized_native_next_is_reversible
          chi0 B dt geometry previous current
      · rfl

theorem stabilized_layer_forward_after_backward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) :
    stabilizedLayerForward chi0 B dt geometry
      (stabilizedLayerBackward chi0 B dt geometry layers) = layers := by
  cases layers with
  | mk previous current =>
      apply congrArg₂ NativeTwoLayerState.mk
      · rfl
      · exact stabilized_native_next_is_reversible
          chi0 B dt geometry current previous

def stabilizedRunningStep [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (_hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) : NativeRunningState Site :=
  nativePairedReadout dt
    (stabilizedLayerForward chi0 B dt geometry
      (rebuildNativeLayers dt running))

def stabilizedRunningStepBackward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (_hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) : NativeRunningState Site :=
  nativePairedReadout dt
    (stabilizedLayerBackward chi0 B dt geometry
      (rebuildNativeLayers dt running))

theorem stabilized_running_backward_after_forward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) :
    stabilizedRunningStepBackward chi0 B dt hdt geometry
      (stabilizedRunningStep chi0 B dt hdt geometry running) = running := by
  unfold stabilizedRunningStepBackward stabilizedRunningStep
  rw [rebuild_after_native_readout dt hdt,
    stabilized_layer_backward_after_forward,
    native_readout_after_rebuild dt hdt]

theorem stabilized_running_forward_after_backward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) :
    stabilizedRunningStep chi0 B dt hdt geometry
      (stabilizedRunningStepBackward chi0 B dt hdt geometry running) = running := by
  unfold stabilizedRunningStepBackward stabilizedRunningStep
  rw [rebuild_after_native_readout dt hdt,
    stabilized_layer_forward_after_backward,
    native_readout_after_rebuild dt hdt]

structure StabilizedFirstOrderMapCertificate
    (dt : ℝ) (hdt : dt ≠ 0) : Prop where
  actualActionUpdate : ∀ previous current : State CubeCorner,
    StabilizedNativeUpdateCertificate chi0 chiKineticNormalization dt
      cubeC19Geometry previous current
  losslessReadout : StabilizedPairedReadoutCertificate dt hdt
  runningRoundTrips :
    (∀ running : NativeRunningState CubeCorner,
      stabilizedRunningStepBackward chi0 chiKineticNormalization dt hdt
        cubeC19Geometry
        (stabilizedRunningStep chi0 chiKineticNormalization dt hdt
          cubeC19Geometry running) = running) ∧
    (∀ running : NativeRunningState CubeCorner,
      stabilizedRunningStep chi0 chiKineticNormalization dt hdt
        cubeC19Geometry
        (stabilizedRunningStepBackward chi0 chiKineticNormalization dt hdt
          cubeC19Geometry running) = running)

theorem dirac008_stabilized_complete_first_order_map
    (dt : ℝ) (hdt : dt ≠ 0) :
    StabilizedFirstOrderMapCertificate dt hdt where
  actualActionUpdate := fun previous current =>
    dirac004_stabilized_c19_reversible_action_update
      dt cubeC19Geometry previous current hdt
  losslessReadout := dirac007_stabilized_lossless_paired_readout dt hdt
  runningRoundTrips :=
    ⟨stabilized_running_backward_after_forward
      chi0 chiKineticNormalization dt hdt cubeC19Geometry,
     stabilized_running_forward_after_backward
      chi0 chiKineticNormalization dt hdt cubeC19Geometry⟩

end
end LeanDirac
