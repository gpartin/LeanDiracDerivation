"""Test the complete-frame symplectic nullspace on the V4-006 defect.

The native pair ``(chi, Psi)`` determines both a parallel complex overlap

    s = chi^dagger Psi

and an oriented area

    W = chi wedge Psi.

The previously installed cube-incidence one-form uses one exact integer
combination of tensor/area Berry forms.  The exact rest-force calculation has
two additional independent directions.  This audit derives their full DEL
blocks and asks whether they span the measured one-cycle Dirac defect.  It is
an inverse response-rank calculation: fitted coefficients are diagnostics and
are never installed in the action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import cube_incidence_repository_action_basis_screen as repository
import cube_incidence_symplectic_basis as basis
import cube_incidence_symplectic_completion_gate as completion
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cube_incidence_complete_frame_matrices.npz"

# Exact null vectors of the rest DEL force.  On the retained rest orbit the
# four nonzero force columns obey
#
#   F_tensor_common = -F_tensor_relative = F_area_common
#                   = F_area_common_C19 / 12.
#
# K_OVERLAP is the scalar-overlap complement implied by the complex Lagrange
# identity; K_COMMON_RELATIVE is the independent common/relative time channel.
K_COMMON_RELATIVE = np.asarray((1., 0., 1., 0., 0., 0., 0., 0.))
K_OVERLAP = np.asarray((-1., 0., 0., 0., 1., 0., 0., 0.))


def derive_blocks(rest: np.ndarray, coefficients: np.ndarray) -> dict:
    """Differentiate one real reciprocal interval action exactly."""
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = [
        torch.tensor(rotation.T@rest, dtype=torch.float64,
                     requires_grad=True),
        torch.tensor(rest, dtype=torch.float64, requires_grad=True),
        torch.tensor(rotation@rest, dtype=torch.float64,
                     requires_grad=True),
    ]
    coefficient_tensor = torch.tensor(coefficients, dtype=torch.float64)

    def residual(left, middle, right):
        return basis.del_residual(
            left, middle, right, coefficient_tensor)

    jacobians = []
    for selected in range(3):
        def function(value, argument=selected):
            trial = layers.copy()
            trial[argument] = value
            return residual(*trial)
        jacobians.append(torch.autograd.functional.jacobian(
            function, layers[selected], vectorize=True).detach().numpy())
    minus = jacobians[0]@rotation.T
    zero = jacobians[1]
    plus = jacobians[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
        "rest_DEL": residual(*layers).detach().numpy(),
    }


def load_blocks(rest: np.ndarray) -> dict:
    if CACHE.exists():
        archive = np.load(CACHE)
        return {
            name: {kind: archive[f"{name}_{kind}"]
                   for kind in ("mass", "gyro", "stiffness", "rest_DEL")}
            for name in ("common_relative", "overlap")
        }
    result = {
        "common_relative": derive_blocks(rest, K_COMMON_RELATIVE),
        "overlap": derive_blocks(rest, K_OVERLAP),
    }
    np.savez_compressed(CACHE, **{
        f"{name}_{kind}": value
        for name, row in result.items() for kind, value in row.items()
    })
    return result


def matrix_scale(base, candidate):
    active = []
    for reference, delta in zip(base, candidate, strict=True):
        if np.linalg.norm(delta) > 1e-30:
            active.append(0.01*np.linalg.norm(reference)/np.linalg.norm(delta))
    return min(active) if active else 0.0


def main() -> None:
    bare, _ = action.expected_blocks()
    base = ablation.blocks()["complete_research_action"]
    complete = load_blocks(bare["rest"])
    candidates = {
        "complete_frame_common_relative": tuple(
            complete["common_relative"][kind]
            for kind in ("mass", "gyro", "stiffness")),
        "complete_frame_overlap": tuple(
            complete["overlap"][kind]
            for kind in ("mass", "gyro", "stiffness")),
        "installed_cube_incidence": tuple(
            np.load(completion.CACHE)[kind]
            for kind in ("mass", "gyro", "stiffness")),
    }
    # Add every previously derived reciprocal direction so the verdict is on
    # the union of the old vocabulary and the newly exposed complete-frame
    # nullspace, not on either family in isolation.
    bare_blocks = ablation.blocks()[
        "base_without_density_or_cube_connection"]
    density_blocks = ablation.blocks()["base_plus_density_connection"]
    cube_blocks = ablation.blocks()[
        "base_plus_cube_incidence_connection"]
    candidates.update({
        "existing_density_connection": tuple(
            density_blocks[index]-bare_blocks[index] for index in range(3)),
        "existing_cube_incidence_connection": tuple(
            cube_blocks[index]-bare_blocks[index] for index in range(3)),
    })
    zeros = np.zeros_like(base[0])
    for filename, key, name in (
        ("joint_phase_lock_action_hessian.npz", "hessian",
         "joint_phase_lock"),
        ("relative_bilinear_phase_action_hessian.npz",
         "normalized_hessian", "relative_bilinear_phase"),
        ("area_bilinear_phase_action_hessian.npz",
         "normalized_hessian", "area_bilinear_phase"),
    ):
        matrix = np.load(HERE/filename)[key]
        candidates[name] = (zeros, zeros, matrix)
    pair = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    for kind in ("common", "relative"):
        candidates[f"density_oriented_{kind}_pair"] = tuple(
            pair[f"{block}_{kind}_pair_connection"]
            for block in ("mass", "gyro", "stiffness"))

    # Reproduce exactly the one-cycle long-packet comparator.
    dt = native_time.DT/16
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row = next(item for item in prior["rows"] if item["flavor"] == "plus")
    case = next(item for item in row["cases"]
                if item["envelope"] == "all_125_centers")
    scale_chi = complex(case["chi_scale_real"], case["chi_scale_imag"])
    spin_basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(spin_basis)
    raw = envelope.raw_columns(
        spin_basis, psi_frame, chi_frame, sites, gate.OFFSETS, scale_chi)
    previous, initial, _following, _ = gate.packet(
        0, 1, alpha, beta, dt)
    q_previous = raw@gate.real_coordinates(previous)
    q_current = raw@gate.real_coordinates(initial)
    target = gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    q_target = raw@gate.real_coordinates(target)
    q_target_i = raw@gate.real_coordinates(1j*target)
    endpoint = repository.propagate(
        base, q_previous.copy(), q_current.copy(), dt, steps)

    columns = []
    rows = []
    for name, candidate in candidates.items():
        scale = matrix_scale(base, candidate)
        plus = tuple(base[index]+scale*candidate[index]
                     for index in range(3))
        minus = tuple(base[index]-scale*candidate[index]
                      for index in range(3))
        endpoint_plus = repository.propagate(
            plus, q_previous.copy(), q_current.copy(), dt, steps)
        endpoint_minus = repository.propagate(
            minus, q_previous.copy(), q_current.copy(), dt, steps)
        response = (endpoint_plus-endpoint_minus)/2
        columns.append(response)
        rows.append({
            "action_direction": name,
            "coefficient_scale_for_one_percent_matrix_perturbation": scale,
            "endpoint_response_norm": float(np.linalg.norm(response)),
        })

    action_response = np.column_stack(columns)
    desired = q_target-endpoint
    # Permit common phase and amplitude only as readout freedoms, exactly as
    # in the preceding repository-basis diagnostic.
    response = np.column_stack((action_response, q_target_i, q_target))
    coefficients, *_ = np.linalg.lstsq(response, desired, rcond=1e-10)
    repaired = endpoint+response@coefficients
    singular = np.linalg.svd(action_response, compute_uv=False)
    baseline_error = np.linalg.norm(desired)/np.linalg.norm(q_target)
    best_error = np.linalg.norm(repaired-q_target)/np.linalg.norm(q_target)

    # Exact rest identities and variational matrix symmetries.
    rest_rows = []
    for name, row_blocks in complete.items():
        rest_rows.append({
            "action_direction": name,
            "rest_DEL_norm": float(np.linalg.norm(row_blocks["rest_DEL"])),
            "mass_symmetry_defect": float(np.linalg.norm(
                row_blocks["mass"]-row_blocks["mass"].T)),
            "gyro_antisymmetry_defect": float(np.linalg.norm(
                row_blocks["gyro"]+row_blocks["gyro"].T)),
            "stiffness_symmetry_defect": float(np.linalg.norm(
                row_blocks["stiffness"]-row_blocks["stiffness"].T)),
        })

    result = {
        "status": "V4_CUBE_INCIDENCE_COMPLETE_FRAME_INVERSE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "native_complete_frame": {
            "parallel_coordinate": "s_i = chi_i^dagger Psi_i",
            "oriented_area_coordinate": "W_i = chi_i wedge Psi_i",
            "identity": "rho_chi rho_Psi = |s|^2 + ||W||^2",
        },
        "exact_rest_null_vectors": {
            "common_relative": dict(zip(
                basis.BASIS, K_COMMON_RELATIVE.tolist(), strict=True)),
            "overlap": dict(zip(
                basis.BASIS, K_OVERLAP.tolist(), strict=True)),
        },
        "rest_and_reciprocity_checks": rest_rows,
        "response_rows": rows,
        "action_response_rank": int(np.sum(
            singular > max(singular[0], 1.0)*1e-10)),
        "action_response_singular_values": [float(value) for value in singular],
        "baseline_relative_endpoint_error": float(baseline_error),
        "best_complete_frame_plus_repository_linearized_relative_error":
            float(best_error),
        "relative_improvement": float(
            (baseline_error-best_error)/max(baseline_error, 1e-30)),
        "diagnostic_coefficients": {
            rows[index]["action_direction"]: float(coefficients[index])
            for index in range(len(rows))
        },
        "verdict": (
            "COMPLETE_FRAME_SPAN_CLOSES_LINEARIZED_ENDPOINT"
            if best_error < 0.05 else
            "COMPLETE_FRAME_SPAN_DOES_NOT_CLOSE_LINEARIZED_ENDPOINT"),
        "claim_boundary": (
            "Exact action-derived tangent response on the one-cycle packet. "
            "Fitted inverse coefficients are diagnostic only and are not an "
            "admitted action or a V4-006 movement pass."),
    }
    (HERE/"cube_incidence_complete_frame_inverse.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
