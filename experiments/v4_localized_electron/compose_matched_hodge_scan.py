"""Compose the geometry-normalized Hodge branch response with the best timing action."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
BASE = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p016620498615_symplectic_0_overlap_0_"
    "overlapface_0_overlapmatched_1_overlappotential_1_triangle_0_"
    "dt_0p00390625.npz")
HODGE = HERE / "cube_incidence_hodge_branch_response_matrices.npz"
SCALES = (-1.0, -1/19, 0.0, 1/19, 1.0)


def tag(value: float) -> str:
    return format(value, ".8g").replace(".", "p").replace("-", "m")


def main() -> None:
    with np.load(BASE) as archive:
        base = {name: archive[name].copy()
                for name in ("previous", "current", "following")}
    with np.load(HODGE) as archive:
        mass = archive["mass"]
        gyro = archive["gyro"]
        stiffness = archive["stiffness"]
    dt = gate.DT
    minus = -mass/dt+gyro/2
    plus = -mass/dt-gyro/2
    zero = 2*mass/dt-dt*stiffness
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    raw_direction = {
        "previous": gate.rotate_native_phase(
            minus.reshape(-1), -theta).reshape(minus.shape),
        "current": zero,
        "following": gate.rotate_native_phase(
            plus.reshape(-1), theta).reshape(plus.shape),
    }
    outputs = []
    for scale in SCALES:
        output = HERE/f"matched_hodge_scan_{tag(scale)}.npz"
        np.savez_compressed(output, **{
            name: base[name]+scale*raw_direction[name] for name in base})
        outputs.append({"hodge_scale": scale, "output": output.name})
    report = {"status": "MATCHED_HODGE_SCAN_COMPOSED", "outputs": outputs}
    (HERE/"compose_matched_hodge_scan.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
