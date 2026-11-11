import LeanDirac.Dirac009ActionBridge

namespace LeanDirac

/-!
# DIRAC-006 completed transport certificate

The earlier certificate established the C19 and reverse-bond lineage of the
full nonlinear action.  The theorem added here closes the part that was
previously missing: the surrounding-cell linear response of that same action
is the signed cubical incidence transport.  The period-two cell is used only
for the prepared internal geometry; the Bloch phases retain the distinct
neighbouring cells.
-/

structure Dirac006ActionTransportCertificate : Prop where
  nonlinearC19Lineage : IncidenceTransportCertificate
  exactActionBlochBridge : Dirac009ActionBridgeCertificate
  incidenceIsNonzero : ∃ (momentum : Fin 3 → ℝ) (amplitude : Cell8),
    cubeIncidenceLinear
      (momentum 0) (momentum 1) (momentum 2) amplitude ≠ 0

theorem cube_incidence_transport_nonzero :
    ∃ (momentum : Fin 3 → ℝ) (amplitude : Cell8),
      cubeIncidenceLinear
        (momentum 0) (momentum 1) (momentum 2) amplitude ≠ 0 := by
  let momentum : Fin 3 → ℝ := fun axis => if axis = 0 then 1 else 0
  let amplitude : Cell8 := fun _ => 1
  refine ⟨momentum, amplitude, ?_⟩
  intro hzero
  have hcorner := congrFun hzero 0
  norm_num [momentum, amplitude, cubeIncidenceLinear, cubeIncidenceGamma,
    cubeIncidenceSign, cubeIncidenceFlip, cellFlip0, cellFlip1,
    cellFlip2, cellBit0, cellBit1] at hcorner
  simp at hcorner

theorem dirac006_complete_action_transport :
    Dirac006ActionTransportCertificate where
  nonlinearC19Lineage := dirac006_incidence_native_transport
  exactActionBlochBridge := dirac009_action_bridge_certificate
  incidenceIsNonzero := cube_incidence_transport_nonzero

end LeanDirac

#print axioms LeanDirac.dirac006_complete_action_transport
