import LeanDirac.Dirac005Incidence
import LeanDirac.Dirac006Stabilized

namespace LeanDirac

/-! DIRAC-006 records the complete native transport lineage of the incidence action. -/

structure IncidenceTransportCertificate : Prop where
  stabilizedTransport : StabilizedTransportCertificate
  faceIncidenceIsC19 : ∀ neighbor,
    neighbor.val ≥ 6 → c19FaceIncidenceSign neighbor = 0
  reverseFaceOrientation : ∀ neighbor,
    c19FaceIncidenceSign (c19Reverse neighbor) =
      -c19FaceIncidenceSign neighbor
  onlyTwoNativeFields : ∀ state : State CubeCorner,
    state = ⟨state.psi, state.chi⟩

theorem dirac006_incidence_native_transport :
    IncidenceTransportCertificate where
  stabilizedTransport := dirac006_stabilized_native_transport
  faceIncidenceIsC19 := by
    intro neighbor h
    fin_cases neighbor <;> norm_num at h
    all_goals rfl
  reverseFaceOrientation := c19_face_incidence_reverse
  onlyTwoNativeFields := state_has_exactly_two_field_families

end LeanDirac
