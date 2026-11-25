"""Rank-test the tested timing-action vocabulary against translation pinning."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(HERE/name) as archive:
        return {key: archive[key].copy()
                for key in ("previous", "current", "following")}


def delta(left, right, denominator: float):
    return {key: (left[key]-right[key])/denominator for key in left}


def main() -> None:
    rank.lock_inertia()
    best = load("completed_spacetime_scan_0p016620498615.npz")
    st_zero = load("completed_spacetime_scan_0.npz")
    st_direction = delta(best, st_zero, 0.01662049861495845)
    base_two_name = (
        "live_clifford_tangent_blocks_"
        "joint_face_laplacian_plus_frame_oscillator_0p315789473684_"
        "plaquette_0_spacetime_0p00554016620499_symplectic_0_overlap_0_"
        "overlapface_0_overlapmatched_1_overlappotential_1_triangle_0_"
        "dt_0p00390625.npz")
    base_two = load(base_two_name)
    sym_plus = load(
        "live_clifford_tangent_blocks_joint_face_laplacian_plus_frame_"
        "oscillator_0p315789473684_plaquette_0_spacetime_0p00554016620499_"
        "symplectic_0p00277008310249_overlap_0_overlapface_0_"
        "overlapmatched_1_overlappotential_1_dt_0p00390625.npz")
    triangle = load(
        "live_clifford_tangent_blocks_joint_face_laplacian_plus_frame_"
        "oscillator_0p315789473684_plaquette_0_spacetime_0p00554016620499_"
        "symplectic_0_overlap_0_overlapface_0_overlapmatched_1_"
        "overlappotential_1_triangle_1_dt_0p00390625.npz")
    matched_half = load(
        "live_clifford_tangent_blocks_joint_face_laplacian_plus_frame_"
        "oscillator_0p315789473684_plaquette_0_spacetime_0p00554016620499_"
        "symplectic_0_overlap_0_overlapface_0_overlapmatched_0p5_"
        "dt_0p00390625.npz")
    potential_half = load(
        "live_clifford_tangent_blocks_joint_face_laplacian_plus_frame_"
        "oscillator_0p315789473684_plaquette_0_spacetime_0p00554016620499_"
        "symplectic_0_overlap_0_overlapface_0_overlapmatched_1_"
        "overlappotential_0p5_dt_0p00390625.npz")
    material_one = load(
        "live_clifford_tangent_blocks_joint_face_laplacian_plus_frame_"
        "oscillator_0p315789473684_plaquette_0_spacetime_0p016620498615_"
        "symplectic_0_overlap_0_overlapface_0_overlapmatched_1_"
        "overlappotential_1_triangle_0_materialarea_1_dt_0p00390625.npz")
    hodge_one = load("matched_hodge_scan_1.npz")
    directions = {
        "spacetime_face": st_direction,
        "parallel_material": delta(base_two, matched_half, 0.5),
        "parallel_clock_potential": delta(base_two, potential_half, 0.5),
        "face_symplectic": delta(sym_plus, base_two, 1/361),
        "triangle": delta(triangle, base_two, 1.0),
        "transverse_material_area": delta(material_one, best, 1.0),
        "hodge_branch": delta(hodge_one, best, 1.0),
    }
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    dt = gate.DT
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    psi, chi = rank.native.unpack(rest, 5)
    translations = rank.native.translation_vectors(psi, chi)

    def images(blocks):
        result = []
        for axis in range(3):
            vector = translations[:, axis]
            vector /= max(np.linalg.norm(vector), 1e-30)
            result.append(
                blocks["previous"]@(rotate_minus@vector)
                + blocks["current"]@vector
                + blocks["following"]@(rotate_plus@vector))
        return np.concatenate(result)

    target = images(best)
    matrix = np.column_stack([images(value) for value in directions.values()])
    coefficients, *_ = np.linalg.lstsq(matrix, -target, rcond=1e-10)
    residual = target+matrix@coefficients
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape)*max(singular[0], 1.0)*1e-10
    result = {
        "status": "V4_TIMING_ACTION_TRANSLATION_RESPONSE_RANK",
        "candidate_directions": list(directions),
        "response_rank": int(np.sum(singular > tolerance)),
        "singular_values": [float(value) for value in singular],
        "baseline_translation_residual_norm": float(np.linalg.norm(target)),
        "best_combined_translation_residual_norm": float(np.linalg.norm(residual)),
        "best_combined_relative_residual": float(
            np.linalg.norm(residual)/max(np.linalg.norm(target), 1e-30)),
        "diagnostic_coefficients": {
            name: float(value) for name, value in zip(
                directions, coefficients, strict=True)},
        "coefficients_admitted": False,
        "verdict": "TESTED_ACTION_VOCABULARY_" + (
            "SPANS_TRANSLATION_REPAIR" if np.linalg.norm(residual)
            < 1e-8*np.linalg.norm(target) else "DOES_NOT_SPAN_TRANSLATION_REPAIR"),
        "claim_boundary": (
            "Linear co-rotating translation response only. Diagnostic least-"
            "squares coefficients are not admitted action constants."),
    }
    (HERE/"timing_action_translation_rank.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
