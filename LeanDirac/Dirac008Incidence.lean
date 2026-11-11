import LeanDirac.Dirac004Incidence
import LeanDirac.Dirac007Incidence

namespace LeanDirac

/-!
DIRAC-008 packages the incidence action's reversible second-order update as a
lossless first-order state.  The readout retains both complete native layers;
it inserts no four-component projection.
-/

noncomputable section

def incidenceLayerForward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) : NativeTwoLayerState Site where
  previous := layers.current
  current := incidenceNativeNextState chi0 B dt geometry
    layers.previous layers.current

def incidenceLayerBackward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) : NativeTwoLayerState Site where
  previous := incidenceNativeNextState chi0 B dt geometry
    layers.current layers.previous
  current := layers.previous

theorem incidence_layer_backward_after_forward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) :
    incidenceLayerBackward chi0 B dt geometry
      (incidenceLayerForward chi0 B dt geometry layers) = layers := by
  cases layers with
  | mk previous current =>
      apply congrArg₂ NativeTwoLayerState.mk
      · exact incidence_native_next_is_reversible
          chi0 B dt geometry previous current
      · rfl

theorem incidence_layer_forward_after_backward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (geometry : C19Lattice Site)
    (layers : NativeTwoLayerState Site) :
    incidenceLayerForward chi0 B dt geometry
      (incidenceLayerBackward chi0 B dt geometry layers) = layers := by
  cases layers with
  | mk previous current =>
      apply congrArg₂ NativeTwoLayerState.mk
      · rfl
      · exact incidence_native_next_is_reversible
          chi0 B dt geometry current previous

def incidenceRunningStep [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (_hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) : NativeRunningState Site :=
  nativePairedReadout dt
    (incidenceLayerForward chi0 B dt geometry
      (rebuildNativeLayers dt running))

def incidenceRunningStepBackward [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (_hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) : NativeRunningState Site :=
  nativePairedReadout dt
    (incidenceLayerBackward chi0 B dt geometry
      (rebuildNativeLayers dt running))

theorem incidence_running_backward_after_forward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) :
    incidenceRunningStepBackward chi0 B dt hdt geometry
      (incidenceRunningStep chi0 B dt hdt geometry running) = running := by
  unfold incidenceRunningStepBackward incidenceRunningStep
  rw [rebuild_after_native_readout dt hdt,
    incidence_layer_backward_after_forward,
    native_readout_after_rebuild dt hdt]

theorem incidence_running_forward_after_backward
    [Fintype Site] [DecidableEq Site]
    (chi0 B dt : ℝ) (hdt : dt ≠ 0) (geometry : C19Lattice Site)
    (running : NativeRunningState Site) :
    incidenceRunningStep chi0 B dt hdt geometry
      (incidenceRunningStepBackward chi0 B dt hdt geometry running) = running := by
  unfold incidenceRunningStepBackward incidenceRunningStep
  rw [rebuild_after_native_readout dt hdt,
    incidence_layer_forward_after_backward,
    native_readout_after_rebuild dt hdt]

structure IncidenceFirstOrderMapCertificate
    (dt : ℝ) (hdt : dt ≠ 0) : Prop where
  actualActionUpdate : ∀ previous current : State CubeCorner,
    IncidenceNativeUpdateCertificate chi0 chiKineticNormalization dt
      cubeC19Geometry previous current
  losslessReadout : IncidencePairedReadoutCertificate dt hdt
  runningRoundTrips :
    (∀ running : NativeRunningState CubeCorner,
      incidenceRunningStepBackward chi0 chiKineticNormalization dt hdt
        cubeC19Geometry
        (incidenceRunningStep chi0 chiKineticNormalization dt hdt
          cubeC19Geometry running) = running) ∧
    (∀ running : NativeRunningState CubeCorner,
      incidenceRunningStep chi0 chiKineticNormalization dt hdt
        cubeC19Geometry
        (incidenceRunningStepBackward chi0 chiKineticNormalization dt hdt
          cubeC19Geometry running) = running)

theorem dirac008_incidence_complete_first_order_map
    (dt : ℝ) (hdt : dt ≠ 0) :
    IncidenceFirstOrderMapCertificate dt hdt where
  actualActionUpdate := fun previous current =>
    dirac004_incidence_c19_reversible_action_update
      dt cubeC19Geometry previous current hdt
  losslessReadout := dirac007_incidence_lossless_paired_readout dt hdt
  runningRoundTrips :=
    ⟨incidence_running_backward_after_forward
      chi0 chiKineticNormalization dt hdt cubeC19Geometry,
     incidence_running_forward_after_backward
      chi0 chiKineticNormalization dt hdt cubeC19Geometry⟩

end
end LeanDirac
