from pathlib import Path
from hashlib import sha256
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    data = path.read_bytes()
    if path.suffix in {".json", ".py", ".md", ".lean"}:
        data = data.replace(b"\r\n", b"\n")
    return sha256(data).hexdigest()


def require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite metric")
    if isinstance(value, list):
        for item in value:
            require_finite(item)
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)


def main():
    pairs = {
        "face_phase_momentum": ["face_phase_momentum_seed", "face_phase_momentum_matched",
                                "face_phase_momentum_antiphase_bvp", "face_phase_momentum_antiphase_balanced"],
        "local_phase_transfer_rank": ["local_phase_transfer_rank", "local_phase_transfer_rank_antiphase"],
        "relative_receiver_phase_transfer": ["relative_receiver_phase_transfer", "antiphase_face_routing"],
        "phase_corrected_handoff": ["antiphase_handoff", "antiphase_phase_balanced_handoff"],
        "face_frame_transport_probe": ["face_frame_transport_probe"],
    }
    action = digest(HERE / "live_clifford_bond_action.py")
    candidate = HERE / "v4_006_face_frame_transport_action.py"
    sources = [HERE / ("v4_006_" + name + ".py") for name in pairs] + [candidate]
    states = {digest(HERE / ("v4_006_" + name + ".npz")): "v4_006_" + name + ".npz"
              for name in ("face_charge_transport_seed", "face_capacity_matched_speed",
                           "antiphase_handoff", "antiphase_phase_balanced_handoff", "antiphase_transfer_seed")}
    artifacts = {}
    for source, names in pairs.items():
        for name in names:
            path = HERE / ("v4_006_" + name + ".json")
            data = json.loads(path.read_text(encoding="utf-8"))
            require_finite(data)
            if data["action_sha256"] != action:
                raise ValueError(f"action mismatch: {path.name}")
            if "state_sha256" in data and data["state_sha256"] not in states:
                raise ValueError(f"missing originating state: {path.name}")
            if "candidate_sha256" in data and data["candidate_sha256"] != digest(candidate):
                raise ValueError(f"candidate mismatch: {path.name}")
            artifacts[path.name] = {"sha256": digest(path), "source": "v4_006_" + source + ".py",
                                    "status": data["status"]}
    tracker = json.loads((HERE / "v4_coherent_stage_tracker.json").read_text(encoding="utf-8"))
    gates = {row["id"]: row["status"] for row in tracker["stages"]}
    if any(gates[f"V4-{i:03d}"] != "PASS" for i in range(1, 6)) or gates["V4-006"] != "OPEN":
        raise ValueError("incorrect status boundary")
    proof = ROOT / "LeanDirac/V4/FacePhaseMomentum.lean"
    output = {
        "status": "RESEARCH_EVIDENCE_INTEGRITY_PASS__NOT_MOVEMENT_PASS",
        "action_sha256": action,
        "source_hashes": {path.name: digest(path) for path in sources},
        "research_lean_source": {"path": str(proof.relative_to(ROOT).as_posix()), "sha256": digest(proof)},
        "state_artifacts": states, "artifacts": artifacts, "stage_status": gates,
        "claim_boundary": "Evidence integrity only. Compile the separate Lean module and full chain independently."
    }
    (HERE / "v4_006_phase_momentum_artifact_audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Checked {len(artifacts)} reports and {len(states)} state artifacts; V4-006 remains OPEN")


if __name__ == "__main__":
    main()
