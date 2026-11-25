"""Screen the native Psi-led and chi-led live connections together.

Equal coefficients represent a common material connection; opposite
coefficients represent the relative Psi--chi connection.  Only simple
geometry-selected pairs are candidates for admission.  The wider grid is an
RCA diagnostic and cannot select a coefficient by fitting Dirac.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_scale_scan as scan
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"
PSI = HERE / "live_clifford_tangent_blocks_psi_frame_phase.npz"
CHI = HERE / "live_clifford_tangent_blocks_chi_frame_phase.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=0.1)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--output", default="live_clifford_relative_connection_scan.json")
    args = parser.parse_args()
    rank.lock_inertia()
    incumbent = scan.blocks(INCUMBENT)
    psi_unit = scan.blocks(PSI)
    chi_unit = scan.blocks(CHI)
    psi_addition = tuple(psi_unit[i]-incumbent[i] for i in range(3))
    chi_addition = tuple(chi_unit[i]-incumbent[i] for i in range(3))
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    import canonical_noncoplanar_bifurcation_audit as native_time
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    rest_layers = (rotate_minus@rest, rest, rotate_plus@rest)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    values = (-4, -2, -1, 0, 1, 2, 4)
    rows = []
    for psi_coefficient in values:
        for chi_coefficient in values:
            print(f"Psi {psi_coefficient}, chi {chi_coefficient}", flush=True)
            combined = tuple(
                psi_coefficient*psi_addition[i]
                + chi_coefficient*chi_addition[i] for i in range(3))
            row = scan.propagate(
                1, args.cycles, specifications, rest, rest_layers,
                rotate_minus, rotate_plus, psi_frame, chi_frame, sites,
                incumbent, combined)
            row["psi_connection_coefficient"] = psi_coefficient
            row["chi_connection_coefficient"] = chi_coefficient
            rows.append(row)
    baseline = next(row for row in rows
                    if row["psi_connection_coefficient"] == 0
                    and row["chi_connection_coefficient"] == 0)
    geometric = [row for row in rows
                 if abs(row["psi_connection_coefficient"]) <= 1
                 and abs(row["chi_connection_coefficient"]) <= 1]
    best_geometric = min(geometric, key=lambda row: row["maximum_shape_error"])
    best_grid = min(rows, key=lambda row: row["maximum_shape_error"])
    result = {
        "status": "V4_LIVE_CLIFFORD_RELATIVE_CONNECTION_SCAN",
        "cycles": args.cycles,
        "case_count": len(specifications),
        "baseline_maximum_shape_error": baseline["maximum_shape_error"],
        "best_geometry_selected_pair": {
            "psi": best_geometric["psi_connection_coefficient"],
            "chi": best_geometric["chi_connection_coefficient"],
            "maximum_shape_error": best_geometric["maximum_shape_error"],
        },
        "best_diagnostic_grid_pair": {
            "psi": best_grid["psi_connection_coefficient"],
            "chi": best_grid["chi_connection_coefficient"],
            "maximum_shape_error": best_grid["maximum_shape_error"],
        },
        "rows": rows,
        "verdict": (
            "SIMPLE_RELATIVE_CONNECTION_IMPROVES_TRANSPORT"
            if best_geometric["maximum_shape_error"]
            < 0.95*baseline["maximum_shape_error"] else
            "SIMPLE_RELATIVE_CONNECTION_NOT_SUFFICIENT"),
        "claim_boundary": (
            "Linear tangent diagnostic. Grid optimization is not coefficient "
            "provenance and no combination is admitted by this scan."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
