"""Test the repaired action for a neutral localized translation tangent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import cube_incidence_nonlinear_action as prior_action
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", default="repaired_translation_mode_probe.json")
    args = parser.parse_args()
    rank.lock_inertia()
    with np.load(args.cache) as archive:
        previous = archive["previous"]
        current = archive["current"]
        following = archive["following"]
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    dt = gate.DT
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    polynomial_at_one = previous@rotate_minus+current+following@rotate_plus
    psi, chi = rank.native.unpack(rest, 5)
    translations = rank.native.translation_vectors(psi, chi)
    rows = []
    images = []
    vectors = []
    for axis in range(3):
        vector = translations[:, axis]
        vector /= max(np.linalg.norm(vector), 1e-30)
        image = polynomial_at_one@vector
        rows.append({
            "axis": axis,
            "unit_translation_del_residual": float(np.linalg.norm(image)),
            "dt_sq_normalized_residual": float(np.linalg.norm(image)/dt**2),
        })
        vectors.append(vector)
        images.append(image)
    q, _ = np.linalg.qr(np.column_stack(vectors))
    response = np.column_stack(images)
    outside = response-q@(q.T@response)
    result = {
        "status": "V4_REPAIRED_TRANSLATION_MODE_PROBE",
        "axis_rows": rows,
        "translation_response_cross_fraction": float(
            np.linalg.norm(outside)/max(np.linalg.norm(response), 1e-30)),
        "zero_mode_pass": max(
            row["unit_translation_del_residual"] for row in rows) < 1e-8,
        "verdict": "TRANSLATION_ZERO_MODE_" + (
            "PASS" if max(row["unit_translation_del_residual"]
                          for row in rows) < 1e-8 else "ABSENT"),
        "claim_boundary": (
            "Linear co-rotating translation tangent only. Absence establishes "
            "pinning of this exact rest orbit, not a no-go for every moving "
            "finite-amplitude branch."),
    }
    (HERE/args.output).write_text(json.dumps(result, indent=2)+"\n",
                                 encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
