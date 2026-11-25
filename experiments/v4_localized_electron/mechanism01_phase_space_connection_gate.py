"""Gate mechanism 1+5: native symplectic connection on an extended packet.

The native cube-incidence one-form and its exact rest-force completion were
derived before this gate.  This script does not invent another force.  It asks
whether geometry-provenance scales of that reciprocal phase-space connection
make the best cubical spin-diagonal extended packet follow the frozen C19
Dirac comparator for one internal cycle while preserving the same rest orbit.

Every scale is diagnostic until it has independent provenance.  Dirac is used
only as a comparator and never supplies native forces or runtime support.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import cube_incidence_spin_diagonal_packet_gate as spin
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent
SCALES = (
    0.0,
    1.0 / native_time.CHI0**2,
    1.0 / native_time.CHI0,
    1.0,
    4.0,
    6.0,
    8.0,
    12.0,
    native_time.CHI0,
)


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def main() -> None:
    bare, _ = action.expected_blocks()
    block_family = ablation.blocks()
    base = block_family["base_plus_density_connection"]
    cube_only = tuple(
        block_family["base_plus_cube_incidence_connection"][index]
        - block_family["base_without_density_or_cube_connection"][index]
        for index in range(3))
    dt = native_time.DT / 16
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))

    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    chi_scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        chi_scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for scale in SCALES:
        mass0, gyro0, stiffness0 = (
            base[index] + scale * cube_only[index] for index in range(3))
        mass, gyro, stiffness = map(
            sparse, (mass0, gyro0, stiffness0))
        lhs = (mass / dt + gyro / 2).tocsc()
        current_block = (dt * stiffness - 2 * mass / dt).tocsc()
        previous_block = (mass / dt - gyro / 2).tocsc()
        try:
            factor = splu(lhs)
        except RuntimeError:
            rows.append({"scale": scale, "factorization_failed": True})
            continue

        cases = []
        for flavor, basis in gate.flavor_bases().items():
            alpha, beta = gate.compressed_clifford(basis)
            raw = envelope.raw_columns(
                basis, psi_frame, chi_frame, sites, gate.OFFSETS,
                chi_scales[flavor])
            for order in itertools.permutations(range(3)):
                frames = spin.local_frames(alpha, order)
                for axis, sign in ((0, -1), (0, 1), (3, 1)):
                    case = spin.run_case(
                        raw, alpha, beta, frames, axis, sign, dt, steps,
                        factor, current_block, previous_block)
                    case["flavor"] = flavor
                    case["axis_order"] = list(order)
                    cases.append(case)
        best = min(cases, key=lambda row: row[
            "one_cycle_phase_aligned_shape_error"])
        held = [case for case in cases
                if case["flavor"] == best["flavor"]
                and case["axis_order"] == best["axis_order"]]
        rows.append({
            "scale": scale,
            "minimum_mass_eigenvalue_lower_bound": float(
                eigvalsh(mass0, subset_by_index=[0, 0],
                          driver="evr")[0]),
            "best_case": best,
            "selected_frame_worst_case_error": max(
                case["one_cycle_phase_aligned_shape_error"]
                for case in held),
            "selected_frame_cases": held,
        })

    valid = [row for row in rows if "best_case" in row]
    best_row = min(valid, key=lambda row: row["best_case"][
        "one_cycle_phase_aligned_shape_error"])
    winner = (
        best_row["best_case"]["one_cycle_phase_aligned_shape_error"] < 0.05
        and best_row["selected_frame_worst_case_error"] < 0.10
        and best_row["minimum_mass_eigenvalue_lower_bound"] > 1e-9
    )
    result = {
        "status": "V4_MECHANISM_01_PHASE_SPACE_CONNECTION_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mechanisms": [
            "native phase-space parallel transport",
            "self-consistent extended leading tail",
        ],
        "scale_provenance_candidates": {
            "1/chi0^2": 1.0 / native_time.CHI0**2,
            "1/chi0": 1.0 / native_time.CHI0,
            "unit": 1.0,
            "four_components": 4.0,
            "six_faces": 6.0,
            "restricted_C19_character": 8.0,
            "C19_first_moment_normalization": 12.0,
            "chi0": native_time.CHI0,
        },
        "dt": dt,
        "one_cycle_steps": steps,
        "rows": rows,
        "best_row": best_row,
        "winner_thresholds": {
            "best_shape_error": 0.05,
            "selected_frame_worst_shape_error": 0.10,
            "positive_mass": True,
        },
        "verdict": (
            "MECHANISM_01_WINNER_PENDING_NONLINEAR_GATE" if winner else
            "MECHANISM_01_LINEAR_PACKET_NOT_CLOSED"),
        "claim_boundary": (
            "One-cycle linear tangent extended-packet screen of an already "
            "derived reciprocal action direction. A favorable row is not an "
            "admitted coefficient or nonlinear moving electron pass."),
    }
    (HERE / "mechanism01_phase_space_connection_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "best_row": best_row,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
