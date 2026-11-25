"""Zero-momentum mass/time audit across the live-grade coefficient."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_scale_scan as block_io
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
BASE = HERE / os.environ.get(
    "LFM_ZERO_SCAN_BASE",
    "live_clifford_tangent_blocks_discrete_unit_link_psi_frame_phase.npz")
UNIT = HERE / os.environ.get(
    "LFM_ZERO_SCAN_UNIT", "live_relative_grade_tangent_blocks.npz")
OUTPUT = os.environ.get(
    "LFM_ZERO_SCAN_OUTPUT", "live_relative_grade_zero_scan.json")


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def evaluate(blocks, rest, rotate_minus, rotate_plus, psi_frame, chi_frame, sites):
    previous_block, current_block, following_block = blocks
    factor = splu(sparse(following_block@rotate_plus))
    previous_operator = sparse(previous_block@rotate_minus)
    current_operator = sparse(current_block)
    psi_mask = np.zeros(rest.size, dtype=bool)
    for site_index in range(rest.size//12):
        psi_mask[site_index*12:site_index*12+6] = True
    rows = []
    for flavor, basis in dirac.flavor_bases().items():
        _alpha, beta = dirac.compressed_clifford(basis)
        raw = gate.psi_columns(basis, psi_frame, chi_frame, sites)
        hamiltonian = rank.R*(beta-np.eye(4))
        forward = expm(-1j*native_time.DT*hamiltonian)
        backward = expm(1j*native_time.DT*hamiltonian)
        for component in range(4):
            field = np.zeros((5, 5, 5, 4), dtype=complex)
            field[..., component] = 1
            q_previous = raw@dirac.real_coordinates(
                np.einsum("ab,...b->...a", backward, field))
            q_current = raw@dirac.real_coordinates(field)
            q_target = raw@dirac.real_coordinates(
                np.einsum("ab,...b->...a", forward, field))
            q_native = factor.solve(-(
                current_operator@q_current+previous_operator@q_previous))
            rows.append({
                "flavor": flavor,
                "component": component,
                "psi_error": float(
                    np.linalg.norm((q_native-q_target)[psi_mask])
                    / np.linalg.norm(q_target[psi_mask])),
            })
    return {
        "maximum_error": max(row["psi_error"] for row in rows),
        "maximum_negative_frequency_error": max(
            row["psi_error"] for row in rows if row["component"] < 2),
        "maximum_positive_frequency_error": max(
            row["psi_error"] for row in rows if row["component"] >= 2),
        "rows": rows,
    }


def main() -> None:
    rank.lock_inertia()
    base = block_io.blocks(BASE)
    unit = block_io.blocks(UNIT)
    addition = tuple(unit[i]-base[i] for i in range(3))
    rest = prior_action.expected_blocks()[0]["rest"]
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    rows = []
    for scale in (-32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32):
        print(f"zero-momentum grade scale {scale}", flush=True)
        blocks = tuple(base[i]+scale*addition[i] for i in range(3))
        row = evaluate(
            blocks, rest, rotate_minus, rotate_plus,
            psi_frame, chi_frame, sites)
        row["scale"] = scale
        rows.append(row)
    best = min(rows, key=lambda row: row["maximum_error"])
    result = {
        "status": "V4_LIVE_RELATIVE_GRADE_ZERO_SCAN",
        "rows": rows,
        "diagnostic_best_scale": best["scale"],
        "diagnostic_best_error": best["maximum_error"],
        "verdict": (
            "LIVE_GRADE_CAN_CLOSE_ZERO_MOMENTUM_GRADING"
            if best["maximum_error"] < 1e-3 else
            "LIVE_GRADE_ZERO_MOMENTUM_GRADING_OPEN"),
        "claim_boundary": (
            "Diagnostic scale scan. No fitted coefficient is admitted."),
    }
    (HERE/OUTPUT).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
