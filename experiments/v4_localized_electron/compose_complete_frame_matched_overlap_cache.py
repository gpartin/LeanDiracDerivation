"""Compose complete-frame blocks with the matched material-overlap mode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_nonlinear_action as prior_action
import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent
BASE = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0_"
    "dt_0p00390625.npz")
OVERLAP = HERE / "overlap_material_tangent_blocks.npz"
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"
OUTPUT = HERE / (
    "live_clifford_tangent_blocks_"
    "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
    "plaquette_0_spacetime_0p00554016620499_symplectic_0_"
    "overlap_0_overlapface_0_overlapmatched_1_dt_0p00390625.npz")


def main() -> None:
    base = np.load(BASE)
    overlap = np.load(OVERLAP)
    incumbent = np.load(INCUMBENT)
    rest, _ = prior_action.expected_blocks()
    current = torch.tensor(rest["rest"], dtype=torch.float64)
    potential_hessian = torch.func.jacrev(
        torch.func.grad(action.parallel_overlap_potential))(current)
    potential_hessian = potential_hessian.detach().numpy()
    blocks = {
        "previous": base["previous"] + overlap["previous"]
                    - incumbent["previous"],
        "current": base["current"] + overlap["current"]
                    - incumbent["current"]
                    - native_time.DT*potential_hessian,
        "following": base["following"] + overlap["following"]
                     - incumbent["following"],
    }
    np.savez_compressed(OUTPUT, **blocks)
    report = {
        "status": "V4_MATCHED_OVERLAP_TANGENT_CACHE_COMPOSED",
        "identity": (
            "combined = base + overlap_material - shared_incumbent "
            "- dt Hessian(overlap potential)"),
        "potential_hessian_symmetry_error": float(
            np.max(abs(potential_hessian-potential_hessian.T))),
        "output": OUTPUT.name,
    }
    (HERE / "compose_complete_frame_matched_overlap_cache.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
