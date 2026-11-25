"""Screen already-derived reciprocal action invariants against the long defect.

This is an inverse response-rank diagnostic.  It never installs the fitted
coefficients.  Each column is the centered endpoint response to a one-percent
matrix-norm perturbation of an action term already derived in this research
tree.  A low residual would justify a fresh coefficient derivation; a high
residual means the existing repository vocabulary does not span the defect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def sparse(value):
    result = value.copy()
    result[abs(result) < 1e-13] = 0
    return csc_matrix(result)


def propagate(blocks, previous, current, dt, steps):
    mass, gyro, stiffness = map(sparse, blocks)
    lhs = (mass/dt+gyro/2).tocsc()
    cb = (dt*stiffness-2*mass/dt).tocsc()
    pb = (mass/dt-gyro/2).tocsc()
    factor = splu(lhs)
    for _ in range(steps):
        following = factor.solve(-(cb@current+pb@previous))
        previous, current = current, following
    return current


def matrix_scale(base, candidate):
    denominator = np.linalg.norm(candidate)
    if denominator < 1e-30:
        return 0.0
    return 0.01*np.linalg.norm(base)/denominator


def main():
    bare, _ = action.expected_blocks()
    base = ablation.blocks()["complete_research_action"]
    bare_blocks = ablation.blocks()["base_without_density_or_cube_connection"]
    density_blocks = ablation.blocks()["base_plus_density_connection"]
    cube_blocks = ablation.blocks()["base_plus_cube_incidence_connection"]
    zeros = np.zeros_like(base[0])
    candidates = {
        "existing_density_connection": tuple(
            density_blocks[index]-bare_blocks[index] for index in range(3)),
        "existing_cube_incidence_connection": tuple(
            cube_blocks[index]-bare_blocks[index] for index in range(3)),
    }
    for filename, key, name in (
        ("joint_phase_lock_action_hessian.npz", "hessian", "joint_phase_lock"),
        ("relative_bilinear_phase_action_hessian.npz", "normalized_hessian",
         "relative_bilinear_phase"),
        ("area_bilinear_phase_action_hessian.npz", "normalized_hessian",
         "area_bilinear_phase"),
    ):
        matrix = np.load(HERE/filename)[key]
        candidates[name] = (zeros, zeros, matrix)
    pair = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    for kind in ("common", "relative"):
        candidates[f"density_oriented_{kind}_pair"] = tuple(
            pair[f"{block}_{kind}_pair_connection"]
            for block in ("mass", "gyro", "stiffness"))

    dt = native_time.DT/16
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row = next(row for row in prior["rows"] if row["flavor"] == "plus")
    case = next(item for item in row["cases"]
                if item["envelope"] == "all_125_centers")
    scale_chi = complex(case["chi_scale_real"], case["chi_scale_imag"])
    basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, scale_chi)
    previous, initial, _following, _ = gate.packet(0, 1, alpha, beta, dt)
    q_previous = raw@gate.real_coordinates(previous)
    q_current = raw@gate.real_coordinates(initial)
    target = gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    q_target = raw@gate.real_coordinates(target)
    q_target_i = raw@gate.real_coordinates(1j*target)
    endpoint = propagate(base, q_previous.copy(), q_current.copy(), dt, steps)

    columns = []
    rows = []
    for name, candidate in candidates.items():
        scales = [matrix_scale(base[index], candidate[index])
                  for index in range(3)]
        active = [value for value in scales if value > 0]
        coefficient_scale = min(active) if active else 0.0
        plus = tuple(base[index]+coefficient_scale*candidate[index]
                     for index in range(3))
        minus = tuple(base[index]-coefficient_scale*candidate[index]
                      for index in range(3))
        endpoint_plus = propagate(
            plus, q_previous.copy(), q_current.copy(), dt, steps)
        endpoint_minus = propagate(
            minus, q_previous.copy(), q_current.copy(), dt, steps)
        response = (endpoint_plus-endpoint_minus)/2
        columns.append(response)
        rows.append({
            "action_invariant": name,
            "coefficient_scale_for_one_percent_matrix_perturbation":
                coefficient_scale,
            "endpoint_response_norm": float(np.linalg.norm(response)),
        })

    # Common phase and amplitude are readout freedoms, not action columns.
    response = np.column_stack(columns+[q_target_i, q_target])
    desired = q_target-endpoint
    coefficients, *_ = np.linalg.lstsq(response, desired, rcond=1e-10)
    repaired = endpoint+response@coefficients
    residual = np.linalg.norm(repaired-q_target)/np.linalg.norm(q_target)
    singular = np.linalg.svd(response[:, :-2], compute_uv=False)
    result = {
        "status": "V4_CUBE_INCIDENCE_REPOSITORY_ACTION_BASIS_SCREEN",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "rows": rows,
        "action_response_rank": int(np.sum(
            singular > max(singular[0], 1.0)*1e-10)),
        "action_response_singular_values": [float(value) for value in singular],
        "unrepaired_relative_endpoint_error": float(
            np.linalg.norm(endpoint-q_target)/np.linalg.norm(q_target)),
        "best_linearized_repository_basis_relative_error": float(residual),
        "diagnostic_coefficients": {
            row["action_invariant"]: float(coefficients[index])
            for index, row in enumerate(rows)
        },
        "verdict": (
            "EXISTING_REPOSITORY_ACTION_BASIS_HAS_CLOSING_RESPONSE"
            if residual < 0.05 else
            "EXISTING_REPOSITORY_ACTION_BASIS_DEFICIENT"),
        "claim_boundary": (
            "Endpoint linear-response span only. Coefficients are fitted to "
            "the Dirac comparator and are not admissible action constants."),
    }
    (HERE/"cube_incidence_repository_action_basis_screen.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
