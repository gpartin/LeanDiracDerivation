"""Inverse gate for the lowest C19 odd-momentum Psi--chi connection.

All previously screened position and symmetric-gradient responses are even in
lattice momentum.  The C19 centered first derivative supplies the missing
odd symbol.  The smallest rest-preserving field made from the native pair is

    X_i = (rho_Psi,i / chi0^2) chi_i,

and the real interval one-form

    A_curl = Im sum_i X_i^dagger curl_19 dX_i.

The carrier-density activation makes the term vanish in ordinary vacuum and,
on the exact rest orbit, its DEL force vanishes to roundoff.  This file tests
its exact Hessian response; it does not admit the term or its coefficient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_complete_frame_inverse as complete
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import cube_incidence_repository_action_basis_screen as repository
import cube_incidence_symplectic_completion_gate as completion
import current_action_localized_envelope_closure as envelope
import localized_stability as native


HERE = Path(__file__).resolve().parent
CACHE = HERE/"cube_incidence_c19_curl_matrices.npz"
SIDE = 5


def derivative19(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Centered C19 first derivative with the established /12 symbol."""
    result = torch.zeros_like(field)
    for offset, weight in native.DIRECTIONS:
        if offset[axis] != 0:
            result = result + weight*offset[axis]*torch.roll(
                field, shifts=offset, dims=(0, 1, 2))
    return result/12


def curl19(field: torch.Tensor) -> torch.Tensor:
    derivatives = [[derivative19(field[..., component], axis)
                    for component in range(3)] for axis in range(3)]
    return torch.stack((
        derivatives[1][2]-derivatives[2][1],
        derivatives[2][0]-derivatives[0][2],
        derivatives[0][1]-derivatives[1][0],
    ), dim=-1)


def activated_chi(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = native_box.unpack(raw, SIDE)
    density = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1,
                                   keepdim=True))
    return density*chi/native_time.CHI0**2


def one_form(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    midpoint = (left+right)/2
    field = activated_chi(midpoint)
    increment = activated_chi(right)-activated_chi(left)
    return torch.sum(torch.imag(torch.conj(field)*curl19(increment)))


def del_residual(previous: torch.Tensor, current: torch.Tensor,
                 following: torch.Tensor) -> torch.Tensor:
    value = one_form(previous, current)+one_form(current, following)
    return torch.autograd.grad(value, current, create_graph=True)[0]


def derive_blocks(rest: np.ndarray) -> dict:
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
    jacobians = []
    for selected in range(3):
        def function(value, argument=selected):
            trial = layers.copy()
            trial[argument] = value
            return del_residual(*trial)
        jacobians.append(torch.autograd.functional.jacobian(
            function, layers[selected], vectorize=True).detach().numpy())
    minus = jacobians[0]@rotation.T
    zero = jacobians[1]
    plus = jacobians[2]@rotation
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
        "rest_DEL": del_residual(*layers).detach().numpy(),
    }


def load_blocks(rest: np.ndarray) -> dict:
    if CACHE.exists():
        archive = np.load(CACHE)
        return {kind: archive[kind]
                for kind in ("mass", "gyro", "stiffness", "rest_DEL")}
    result = derive_blocks(rest)
    np.savez_compressed(CACHE, **result)
    return result


def main() -> None:
    bare, _ = action.expected_blocks()
    base = ablation.blocks()["complete_research_action"]
    curl = load_blocks(bare["rest"])
    candidate = tuple(curl[kind]
                      for kind in ("mass", "gyro", "stiffness"))

    dt = native_time.DT/16
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row = next(item for item in prior["rows"] if item["flavor"] == "plus")
    case = next(item for item in row["cases"]
                if item["envelope"] == "all_125_centers")
    chi_scale = complex(case["chi_scale_real"], case["chi_scale_imag"])
    spin_basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(spin_basis)
    raw = envelope.raw_columns(
        spin_basis, psi_frame, chi_frame, sites, gate.OFFSETS, chi_scale)
    previous, initial, _following, _momentum = gate.packet(
        0, 1, alpha, beta, dt)
    q_previous = raw@gate.real_coordinates(previous)
    q_current = raw@gate.real_coordinates(initial)
    target = gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    q_target = raw@gate.real_coordinates(target)
    q_target_i = raw@gate.real_coordinates(1j*target)
    endpoint = repository.propagate(
        base, q_previous.copy(), q_current.copy(), dt, steps)

    scale = complete.matrix_scale(base, candidate)
    plus = tuple(base[index]+scale*candidate[index] for index in range(3))
    minus = tuple(base[index]-scale*candidate[index] for index in range(3))
    endpoint_plus = repository.propagate(
        plus, q_previous.copy(), q_current.copy(), dt, steps)
    endpoint_minus = repository.propagate(
        minus, q_previous.copy(), q_current.copy(), dt, steps)
    response = (endpoint_plus-endpoint_minus)/2
    desired = q_target-endpoint
    fit_matrix = np.column_stack((response, q_target_i, q_target))
    coefficients, *_ = np.linalg.lstsq(
        fit_matrix, desired, rcond=1e-10)
    repaired = endpoint+fit_matrix@coefficients

    # Unit geometric coefficient is the only non-fitted coefficient tested.
    unit_endpoint = repository.propagate(
        tuple(base[index]+candidate[index] for index in range(3)),
        q_previous.copy(), q_current.copy(), dt, steps)
    target_phase = np.column_stack((q_target, q_target_i))
    unit_phase, *_ = np.linalg.lstsq(
        target_phase, unit_endpoint, rcond=1e-12)
    unit_phase /= max(np.linalg.norm(unit_phase), 1e-30)
    unit_aligned = target_phase@unit_phase

    # Union response: curl + the exact complete-frame nullspace + all prior
    # reciprocal directions. This determines whether curl adds the missing
    # independent response rather than merely correlating with it.
    complete_blocks = complete.load_blocks(bare["rest"])
    union = {
        "c19_activated_chi_curl": candidate,
        "complete_frame_common_relative": tuple(
            complete_blocks["common_relative"][kind]
            for kind in ("mass", "gyro", "stiffness")),
        "complete_frame_overlap": tuple(
            complete_blocks["overlap"][kind]
            for kind in ("mass", "gyro", "stiffness")),
        "installed_cube_incidence": tuple(
            np.load(completion.CACHE)[kind]
            for kind in ("mass", "gyro", "stiffness")),
    }
    bare_blocks = ablation.blocks()["base_without_density_or_cube_connection"]
    density_blocks = ablation.blocks()["base_plus_density_connection"]
    cube_blocks = ablation.blocks()["base_plus_cube_incidence_connection"]
    union.update({
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
        union[name] = (zeros, zeros, matrix)
    pair = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    for kind in ("common", "relative"):
        union[f"density_oriented_{kind}_pair"] = tuple(
            pair[f"{block}_{kind}_pair_connection"]
            for block in ("mass", "gyro", "stiffness"))
    union_columns = []
    union_rows = []
    for name, delta in union.items():
        delta_scale = complete.matrix_scale(base, delta)
        delta_plus = tuple(base[index]+delta_scale*delta[index]
                           for index in range(3))
        delta_minus = tuple(base[index]-delta_scale*delta[index]
                            for index in range(3))
        result_plus = repository.propagate(
            delta_plus, q_previous.copy(), q_current.copy(), dt, steps)
        result_minus = repository.propagate(
            delta_minus, q_previous.copy(), q_current.copy(), dt, steps)
        column = (result_plus-result_minus)/2
        union_columns.append(column)
        union_rows.append({
            "action_direction": name,
            "coefficient_scale": float(delta_scale),
            "response_norm": float(np.linalg.norm(column)),
        })
    union_action = np.column_stack(union_columns)
    union_matrix = np.column_stack((union_action, q_target_i, q_target))
    union_coefficients, *_ = np.linalg.lstsq(
        union_matrix, desired, rcond=1e-10)
    union_repaired = endpoint+union_matrix@union_coefficients
    union_singular = np.linalg.svd(union_action, compute_uv=False)
    union_error = np.linalg.norm(
        union_repaired-q_target)/np.linalg.norm(q_target)

    result = {
        "status": "V4_CUBE_INCIDENCE_C19_CURL_INVERSE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "candidate": (
            "Im sum X_i^dagger curl19 dX_i, "
            "X_i=(rho_Psi,i/chi0^2) chi_i"),
        "coefficient_provenance": (
            "unit C19 first-derivative normalization only; no fitted "
            "coefficient admitted"),
        "rest_DEL_norm": float(np.linalg.norm(curl["rest_DEL"])),
        "block_checks": {
            "mass_symmetry_defect": float(np.linalg.norm(
                curl["mass"]-curl["mass"].T)),
            "gyro_antisymmetry_defect": float(np.linalg.norm(
                curl["gyro"]+curl["gyro"].T)),
            "stiffness_symmetry_defect": float(np.linalg.norm(
                curl["stiffness"]-curl["stiffness"].T)),
        },
        "one_percent_matrix_scale": float(scale),
        "response_norm": float(np.linalg.norm(response)),
        "response_correlation_with_defect": float(
            np.vdot(response, desired).real/max(
                np.linalg.norm(response)*np.linalg.norm(desired), 1e-30)),
        "baseline_relative_endpoint_error": float(
            np.linalg.norm(desired)/np.linalg.norm(q_target)),
        "single_direction_best_fitted_relative_error": float(
            np.linalg.norm(repaired-q_target)/np.linalg.norm(q_target)),
        "diagnostic_fitted_coefficient": float(coefficients[0]),
        "unit_coefficient_phase_aligned_shape_error": float(
            np.linalg.norm(unit_endpoint-unit_aligned)/np.linalg.norm(q_target)),
        "union_with_complete_frame_and_repository": {
            "rows": union_rows,
            "response_rank": int(np.sum(
                union_singular > max(union_singular[0], 1.0)*1e-10)),
            "singular_values": [float(value) for value in union_singular],
            "best_linearized_relative_error": float(union_error),
            "diagnostic_coefficients": {
                row["action_direction"]: float(union_coefficients[index])
                for index, row in enumerate(union_rows)
            },
        },
        "parity_status": (
            "curl helicity is pseudoscalar under a polar-vector assignment; "
            "parity completion remains mandatory before admission"),
        "verdict": (
            "C19_CURL_RESPONSE_MATERIAL"
            if abs(np.vdot(response, desired).real)/max(
                np.linalg.norm(response)*np.linalg.norm(desired), 1e-30) > 0.1
            else "C19_CURL_RESPONSE_ORTHOGONAL_TO_DEFECT"),
        "claim_boundary": (
            "Exact rest and one-cycle inverse tangent screen only. The term "
            "is not admitted; parity, both copies, all axes, coefficient "
            "provenance, current, nonlinear motion and Lean remain open."),
    }
    (HERE/"cube_incidence_c19_curl_inverse.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
