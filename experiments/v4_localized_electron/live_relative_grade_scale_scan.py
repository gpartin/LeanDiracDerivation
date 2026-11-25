"""Signed normalization diagnostic for the live relative-edge grade."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_scale_scan as scan
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
BASE = HERE / "live_clifford_tangent_blocks_discrete_unit_link_psi_frame_phase.npz"
UNIT = HERE / "live_relative_grade_tangent_blocks.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=0.1)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    rank.lock_inertia()
    base = scan.blocks(BASE)
    unit = scan.blocks(UNIT)
    addition = tuple(unit[index]-base[index] for index in range(3))
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    rest_layers = (rotate_minus@rest, rest, rotate_plus@rest)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    rows = []
    for scale in (-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4):
        print(f"grade scale {scale}", flush=True)
        rows.append(scan.propagate(
            scale, args.cycles, specifications, rest, rest_layers,
            rotate_minus, rotate_plus, psi_frame, chi_frame, sites,
            base, addition))
    zero = next(row for row in rows if row["scale"] == 0)
    best = min(rows, key=lambda row: row["maximum_shape_error"])
    unit_choices = [row for row in rows if abs(row["scale"]) == 1]
    best_unit = min(unit_choices, key=lambda row: row["maximum_shape_error"])
    result = {
        "status": "V4_LIVE_RELATIVE_GRADE_SCALE_SCAN",
        "cycles": args.cycles,
        "case_count": len(specifications),
        "zero_grade_shape_error": zero["maximum_shape_error"],
        "best_unit_orientation": {
            "scale": best_unit["scale"],
            "maximum_shape_error": best_unit["maximum_shape_error"],
        },
        "diagnostic_best": {
            "scale": best["scale"],
            "maximum_shape_error": best["maximum_shape_error"],
        },
        "rows": rows,
        "verdict": (
            "UNIT_RELATIVE_GRADE_ORIENTATION_IMPROVES_TRANSPORT"
            if best_unit["maximum_shape_error"] < 0.95*zero["maximum_shape_error"]
            else "UNIT_RELATIVE_GRADE_NOT_SUFFICIENT"),
        "claim_boundary": (
            "Signed tangent control. Non-unit fitted scales are diagnostic "
            "only and are not action coefficients."),
    }
    (HERE/"live_relative_grade_scale_scan.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
