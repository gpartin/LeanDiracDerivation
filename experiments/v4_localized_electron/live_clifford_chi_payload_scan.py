"""Solve the diagnostic moving chi payload for the live-link recurrence.

Psi is always compared with the unchanged discrete-Dirac control.  A scalar
multiple of the native chi companion frame is supplied only in the two input
layers; the unrestricted two-field recurrence determines the next layer.
The scan diagnoses whether the missing response is a simple native inertia
normalization.  A fitted payload is never admitted as a rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
CACHE = HERE / "live_clifford_tangent_blocks_psi_frame_phase.npz"


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def propagate(chi_scale, cycles, specifications, factor, current_operator,
              previous_operator, rest, psi_frame, chi_frame, sites):
    steps = int(np.ceil(cycles*2*np.pi/rank.R/native_time.DT))
    psi_mask = np.zeros(rest.size, dtype=bool)
    for site_index in range(rest.size//12):
        psi_mask[site_index*12:site_index*12+6] = True
    rows = []
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = dirac.compressed_clifford(basis)
        raw_joint = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, dirac.OFFSETS,
            complex(chi_scale))
        raw_psi = gate.psi_columns(basis, psi_frame, chi_frame, sites)
        packet_previous, packet_initial, _, _momentum = dirac.packet(
            axis, sign, alpha, beta, native_time.DT)
        q_previous = raw_joint@dirac.real_coordinates(packet_previous)
        q_current = raw_joint@dirac.real_coordinates(packet_initial)
        for _step in range(steps):
            q_following = factor.solve(
                -(current_operator@q_current+previous_operator@q_previous))
            q_previous, q_current = q_current, q_following
        target = dirac.dirac_step(
            packet_initial, alpha, beta, 1, steps*native_time.DT)
        target_psi = raw_psi@dirac.real_coordinates(target)
        target_i = raw_psi@dirac.real_coordinates(1j*target)
        phase_basis = np.column_stack((target_psi[psi_mask], target_i[psi_mask]))
        coefficients, *_ = np.linalg.lstsq(
            phase_basis, q_current[psi_mask], rcond=1e-12)
        length = np.linalg.norm(coefficients)
        aligned_psi = phase_basis@(coefficients/max(length, 1e-30))
        rows.append({
            "flavor": flavor,
            "axis": axis,
            "sign": sign,
            "psi_phase_aligned_shape_error": float(
                np.linalg.norm(q_current[psi_mask]-aligned_psi)
                / np.linalg.norm(target_psi[psi_mask])),
            "psi_norm_drift": float(abs(
                np.linalg.norm(q_current[psi_mask])
                / np.linalg.norm(target_psi[psi_mask])-1)),
            "chi_to_psi_norm": float(
                np.linalg.norm(q_current[~psi_mask])
                / max(np.linalg.norm(q_current[psi_mask]), 1e-30)),
        })
    return {
        "chi_input_scale": float(chi_scale),
        "maximum_psi_shape_error": max(
            row["psi_phase_aligned_shape_error"] for row in rows),
        "maximum_psi_norm_drift": max(row["psi_norm_drift"] for row in rows),
        "maximum_chi_to_psi_norm": max(row["chi_to_psi_norm"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=0.1)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--output", default="live_clifford_chi_payload_scan.json")
    args = parser.parse_args()
    rank.lock_inertia()
    data = np.load(CACHE)
    previous_block, current_block, following_block = tuple(
        data[name] for name in ("previous", "current", "following"))
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    factor = splu(sparse(following_block@rotate_plus))
    current_operator = sparse(current_block)
    previous_operator = sparse(previous_block@rotate_minus)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    # Include the natural inertia normalizations explicitly among the controls.
    B = rank.B
    scales = sorted(set((
        -1/np.sqrt(B), -1/B, -0.0004, -0.0002, -0.0001, 0,
        0.0001, 0.0002, 0.0004, 1/B, 1/np.sqrt(B))))
    rows = []
    for scale in scales:
        print(f"chi input scale {scale:.9g}", flush=True)
        rows.append(propagate(
            scale, args.cycles, specifications, factor, current_operator,
            previous_operator, rest, psi_frame, chi_frame, sites))
    zero = next(row for row in rows if row["chi_input_scale"] == 0)
    best = min(rows, key=lambda row: row["maximum_psi_shape_error"])
    result = {
        "status": "V4_LIVE_CLIFFORD_CHI_PAYLOAD_SCAN",
        "cycles": args.cycles,
        "case_count": len(specifications),
        "rows": rows,
        "zero_payload_shape_error": zero["maximum_psi_shape_error"],
        "diagnostic_best_scale": best["chi_input_scale"],
        "diagnostic_best_shape_error": best["maximum_psi_shape_error"],
        "verdict": (
            "CHI_PAYLOAD_CHANNEL_IMPROVES_PSI_TRANSPORT"
            if best["maximum_psi_shape_error"] < 0.95*zero["maximum_psi_shape_error"]
            else "SCALAR_CHI_PAYLOAD_NOT_THE_MISSING_TRANSPORT_RESPONSE"),
        "claim_boundary": (
            "Diagnostic input preparation scan. No fitted payload is part of "
            "the action or accepted movement mechanism."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "zero_payload_shape_error": result["zero_payload_shape_error"],
        "diagnostic_best_scale": result["diagnostic_best_scale"],
        "diagnostic_best_shape_error": result["diagnostic_best_shape_error"],
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
