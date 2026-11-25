"""Hash the six-face V4-006 research evidence without promoting it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FILES = (
    "live_clifford_bond_action.py",
    "live_face_transport_geometry_gate.py",
    "live_face_transport_geometry_gate.json",
    "live_clifford_bond_rest_gate_joint_face_wedge_laplacian_smooth.json",
    "live_link_reflection_covariance_gate_joint_face_wedge_laplacian_smooth.json",
    "live_link_receiver_pattern_gate_joint_face_wedge_laplacian_smooth.json",
    "live_clifford_nonlinear_gate.py",
    "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_all8_priorchi.json",
    "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_decomposition_9updates.json",
    "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_k0_9updates.json",
    "live_clifford_nonlinear_gate_joint_face_laplacian_smooth_uniform_k0_9updates.json",
    "live_clifford_joint_face_laplacian_scale_scan_limit_0p05cycle.json",
    "live_tangent_cache_audit.json",
    "live_face_transport_verdict.py",
    "live_face_transport_verdict.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    missing = [name for name in FILES if not (HERE/name).is_file()]
    result = {
        "status": "V4_006_FACE_TRANSPORT_ARTIFACT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "v4_006_status": "OPEN",
        "hash_algorithm": "sha256",
        "missing": missing,
        "files": {
            name: digest(HERE/name) for name in FILES if (HERE/name).is_file()
        },
        "verdict": (
            "FACE_TRANSPORT_ARTIFACTS_COMPLETE_RESEARCH_NOT_ADMITTED"
            if not missing else "FACE_TRANSPORT_ARTIFACTS_INCOMPLETE"),
    }
    (HERE/"live_face_artifact_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
