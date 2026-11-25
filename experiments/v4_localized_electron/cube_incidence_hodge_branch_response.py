"""Motion-response gate for the admissible activated-relative Hodge branch.

This differentiates the *complete tangent operator* along the first-order
deformed rest branch.  Consequently the response includes both the new Hodge
one-form and the change of the native Psi--chi orbit; it is not the response
of a new term evaluated on a carrier that is no longer stationary.

The resulting action direction is screened against the fixed one-cycle Dirac
packet endpoint used by the preceding inverse tests.  No coefficient is fitted
into an action and no action is admitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_c19_curl_inverse as curl
import cube_incidence_complete_frame_inverse as complete
import cube_incidence_dirac_envelope_gate as gate
import cube_incidence_extended_packet_gate as packet_gate
import cube_incidence_hodge_branch_gate as branch
import cube_incidence_hodge_rest_gate as hodge
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import cube_incidence_repository_action_basis_screen as repository
import cube_incidence_symplectic_completion_gate as completion
import current_action_localized_envelope_closure as envelope
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
CACHE = HERE/"cube_incidence_hodge_branch_response_matrices.npz"
OUTPUT = HERE/"cube_incidence_hodge_branch_response.json"
HODGE_INDEX = 3  # activated relative phase
BRANCH_EPSILON = 1e-4


def orbit_tangent(rest: np.ndarray, rotation: np.ndarray):
    layers = branch.orbit_layers(rest, rotation)
    force = hodge.hodge_force_column(layers, HODGE_INDEX)
    stiffness = ablation.blocks()["complete_research_action"][2]
    stiffness = (stiffness+stiffness.T)/2
    values, vectors = np.linalg.eigh(stiffness)
    threshold = 1e-10*max(float(np.max(abs(values))), 1e-30)
    resolved = abs(values) > threshold
    tangent = (vectors[:, resolved]/values[resolved])@(
        vectors[:, resolved].T@(force/native_time.DT))
    return tangent


def total_del(layers, coefficient: float):
    hodge_coefficients = torch.zeros(
        len(hodge.HODGE_BASIS), dtype=torch.float64)
    hodge_coefficients[HODGE_INDEX] = coefficient
    return (action.del_residual(*layers)
            + hodge.hodge_del(layers, hodge_coefficients))


def blocks_at(raw: np.ndarray, coefficient: float,
              rotation: np.ndarray):
    arrays = (rotation.T@raw, raw, rotation@raw)
    layers = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
              for value in arrays]
    jacobians = []
    for selected in range(3):
        def function(value, argument=selected):
            trial = layers.copy()
            trial[argument] = value
            return total_del(trial, coefficient)
        jacobians.append(torch.autograd.functional.jacobian(
            function, layers[selected], vectorize=True).detach().numpy())
    minus = jacobians[0]@rotation.T
    zero = jacobians[1]
    plus = jacobians[2]@rotation
    dt = native_time.DT
    return {
        "mass": -0.5*dt*(plus+minus),
        "gyro": -(plus-minus),
        "stiffness": -(plus+zero+minus)/dt,
        "rest_DEL": total_del(layers, coefficient).detach().numpy(),
    }


def derive_direction(rest: np.ndarray, rotation: np.ndarray):
    tangent = orbit_tangent(rest, rotation)
    positive = blocks_at(
        rest+BRANCH_EPSILON*tangent, BRANCH_EPSILON, rotation)
    negative = blocks_at(
        rest-BRANCH_EPSILON*tangent, -BRANCH_EPSILON, rotation)
    direction = {
        kind: (positive[kind]-negative[kind])/(2*BRANCH_EPSILON)
        for kind in ("mass", "gyro", "stiffness")}
    np.savez_compressed(CACHE, tangent=tangent, **direction)
    return tangent, direction, positive, negative


def load_direction(rest: np.ndarray, rotation: np.ndarray):
    if CACHE.exists():
        archive = np.load(CACHE)
        return (archive["tangent"],
                {kind: archive[kind]
                 for kind in ("mass", "gyro", "stiffness")}, None, None)
    return derive_direction(rest, rotation)


def endpoint_problem(rest: np.ndarray, base):
    dt = native_time.DT/16
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    plus = next(item for item in prior["rows"] if item["flavor"] == "plus")
    case = next(item for item in plus["cases"]
                if item["envelope"] == "all_125_centers")
    chi_scale = complex(case["chi_scale_real"], case["chi_scale_imag"])
    spin_basis = packet_gate.flavor_bases()["plus"]
    alpha, beta = packet_gate.compressed_clifford(spin_basis)
    raw = envelope.raw_columns(
        spin_basis, psi_frame, chi_frame, sites,
        packet_gate.OFFSETS, chi_scale)
    previous, initial, _following, _momentum = packet_gate.packet(
        0, 1, alpha, beta, dt)
    q_previous = raw@packet_gate.real_coordinates(previous)
    q_current = raw@packet_gate.real_coordinates(initial)
    target = packet_gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    q_target = raw@packet_gate.real_coordinates(target)
    q_target_i = raw@packet_gate.real_coordinates(1j*target)
    endpoint = repository.propagate(
        base, q_previous.copy(), q_current.copy(), dt, steps)
    return (dt, steps, q_previous, q_current, q_target, q_target_i,
            endpoint)


def response_column(base, direction, q_previous, q_current, dt, steps):
    scale = complete.matrix_scale(base, direction)
    positive = tuple(base[index]+scale*direction[index]
                     for index in range(3))
    negative = tuple(base[index]-scale*direction[index]
                     for index in range(3))
    endpoint_positive = repository.propagate(
        positive, q_previous.copy(), q_current.copy(), dt, steps)
    endpoint_negative = repository.propagate(
        negative, q_previous.copy(), q_current.copy(), dt, steps)
    return scale, (endpoint_positive-endpoint_negative)/2


def baseline_trajectory(base, previous, current, dt, steps):
    mass, gyro, stiffness = map(csc_matrix, base)
    factor = splu((mass/dt+gyro/2).tocsc())
    center = (dt*stiffness-2*mass/dt).tocsc()
    memory = (mass/dt-gyro/2).tocsc()
    states = [previous.copy(), current.copy()]
    for _ in range(steps):
        following = factor.solve(-(center@current+memory@previous))
        previous, current = current, following
        states.append(current.copy())
    return states


def branch_direction_on_basis(rest, rotation, tangent, basis):
    """Apply the exact branch-operator derivative to a retained Krylov basis."""
    epsilon = BRANCH_EPSILON
    branch_data = []
    for sign in (1.0, -1.0):
        raw = rest+sign*epsilon*tangent
        layers = list(branch.orbit_layers(raw, rotation))
        coefficient = sign*epsilon
        images = {kind: [] for kind in ("mass", "gyro", "stiffness")}
        for column in range(basis.shape[1]):
            vector = basis[:, column]
            directions = (rotation.T@vector, vector, rotation@vector)
            layer_images = []
            for selected in range(3):
                def function(value, argument=selected):
                    trial = layers.copy()
                    trial[argument] = value
                    return total_del(trial, coefficient)
                _value, image = torch.autograd.functional.jvp(
                    function, layers[selected],
                    torch.tensor(directions[selected], dtype=torch.float64),
                    create_graph=False, strict=False)
                layer_images.append(image.detach().numpy())
            minus, zero, plus = layer_images
            images["mass"].append(-0.5*native_time.DT*(plus+minus))
            images["gyro"].append(-(plus-minus))
            images["stiffness"].append(
                -(plus+zero+minus)/native_time.DT)
        branch_data.append({kind: np.column_stack(values)
                            for kind, values in images.items()})
    positive, negative = branch_data
    derivative_images = {
        kind: (positive[kind]-negative[kind])/(2*epsilon)
        for kind in positive}
    matrices = {
        kind: derivative_images[kind]@basis.T for kind in derivative_images}
    matrices["mass"] = (matrices["mass"]+matrices["mass"].T)/2
    matrices["gyro"] = (matrices["gyro"]-matrices["gyro"].T)/2
    matrices["stiffness"] = (
        matrices["stiffness"]+matrices["stiffness"].T)/2
    return matrices


def efficient_main() -> None:
    branch_name = hodge.HODGE_BASIS[HODGE_INDEX]
    bare = np.load(canonical.CACHE)
    rest = bare["rest"]
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    tangent = orbit_tangent(rest, rotation)
    base = ablation.blocks()["complete_research_action"]
    dt, steps, q_previous, q_current, target, target_i, endpoint = (
        endpoint_problem(rest, base))
    states = baseline_trajectory(
        base, q_previous.copy(), q_current.copy(), dt, steps)
    snapshot = np.column_stack(states+[target, target_i])
    u, singular, _vh = np.linalg.svd(snapshot, full_matrices=False)
    threshold = 1e-10*max(float(singular[0]), 1e-30)
    rank = int(np.sum(singular > threshold))
    basis = u[:, :rank]
    projection_error = float(np.linalg.norm(
        snapshot-basis@(basis.T@snapshot))/np.linalg.norm(snapshot))
    direction_map = branch_direction_on_basis(
        rest, rotation, tangent, basis)
    np.savez_compressed(
        CACHE, tangent=tangent, basis=basis,
        **{kind: value for kind, value in direction_map.items()})
    direction = tuple(direction_map[kind]
                      for kind in ("mass", "gyro", "stiffness"))
    scale, response = response_column(
        base, direction, q_previous, q_current, dt, steps)
    desired = target-endpoint
    fit = np.column_stack((response, target_i, target))
    coefficients, *_ = np.linalg.lstsq(fit, desired, rcond=1e-10)
    repaired = endpoint+fit@coefficients

    # Combine the new branch direction with the already derived complete-frame
    # and odd-curl directions.  This is a response-rank diagnostic only.
    complete_archive = np.load(complete.CACHE)
    curl_archive = np.load(curl.CACHE)
    directions = {
        f"{branch_name}_branch": direction,
        "activated_chi_curl": tuple(
            curl_archive[kind] for kind in ("mass", "gyro", "stiffness")),
        "complete_frame_common_relative": tuple(
            complete_archive[f"common_relative_{kind}"]
            if f"common_relative_{kind}" in complete_archive.files
            else complete_archive[kind+"_common_relative"]
            for kind in ("mass", "gyro", "stiffness")),
        "complete_frame_overlap": tuple(
            complete_archive[f"overlap_{kind}"]
            if f"overlap_{kind}" in complete_archive.files
            else complete_archive[kind+"_overlap"]
            for kind in ("mass", "gyro", "stiffness")),
        "installed_cube_incidence": tuple(
            np.load(completion.CACHE)[kind]
            for kind in ("mass", "gyro", "stiffness")),
    }
    bare_blocks = ablation.blocks()["base_without_density_or_cube_connection"]
    density_blocks = ablation.blocks()["base_plus_density_connection"]
    cube_blocks = ablation.blocks()["base_plus_cube_incidence_connection"]
    directions.update({
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
        directions[name] = (zeros, zeros, matrix)
    pair = np.load(HERE/"density_oriented_pair_connection_matrices.npz")
    for kind in ("common", "relative"):
        directions[f"density_oriented_{kind}_pair"] = tuple(
            pair[f"{block}_{kind}_pair_connection"]
            for block in ("mass", "gyro", "stiffness"))
    columns = []
    rows = []
    for name, delta in directions.items():
        delta_scale, column = response_column(
            base, delta, q_previous, q_current, dt, steps)
        columns.append(column)
        rows.append({
            "direction": name,
            "matrix_scale": float(delta_scale),
            "response_norm": float(np.linalg.norm(column)),
            "desired_correlation": float(np.vdot(column, desired).real
                                          / max(np.linalg.norm(column)
                                                * np.linalg.norm(desired),
                                                1e-30)),
        })
    response_matrix = np.column_stack(columns)
    union = np.column_stack((response_matrix, target_i, target))
    union_coefficients, *_ = np.linalg.lstsq(union, desired, rcond=1e-10)
    union_repaired = endpoint+union@union_coefficients
    singular = np.linalg.svd(response_matrix, compute_uv=False)

    result = {
        "status": "V4_C19_HODGE_BRANCH_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "branch": branch_name,
        "branch_epsilon_for_operator_derivative": BRANCH_EPSILON,
        "branch_tangent_over_rest_norm": float(
            np.linalg.norm(tangent)/np.linalg.norm(rest)),
        "operator_evaluation": {
            "method": "exact Hessian-vector products on baseline Krylov span",
            "snapshot_count": len(states)+2,
            "retained_rank": rank,
            "relative_projection_error": projection_error,
            "singular_threshold": threshold,
        },
        "single_direction": {
            "matrix_scale": float(scale),
            "response_norm": float(np.linalg.norm(response)),
            "desired_correlation": float(np.vdot(response, desired).real
                                          / max(np.linalg.norm(response)
                                                * np.linalg.norm(desired),
                                                1e-30)),
            "baseline_relative_error": float(
                np.linalg.norm(endpoint-target)/np.linalg.norm(target)),
            "diagnostic_fitted_relative_error": float(
                np.linalg.norm(repaired-target)/np.linalg.norm(target)),
            "diagnostic_fitted_action_coefficient": float(coefficients[0]),
        },
        "small_union": {
            "directions": rows,
            "response_rank": int(np.linalg.matrix_rank(
                response_matrix, tol=1e-10)),
            "singular_values": [float(value) for value in singular],
            "diagnostic_relative_error": float(
                np.linalg.norm(union_repaired-target)/np.linalg.norm(target)),
            "diagnostic_coefficients": {
                **{row["direction"]: float(value)
                   for row, value in zip(rows, union_coefficients, strict=False)},
                "phase_alignment": float(union_coefficients[len(rows)]),
                "target_rescaling": float(union_coefficients[len(rows)+1]),
            },
        },
        "verdict": "HODGE_BRANCH_MOTION_RESPONSE_MEASURED_ON_KRYLOV_SPAN",
        "claim_boundary": (
            "The fitted coefficients are inverse diagnostics, not admitted "
            "action constants. A V4-006 pass requires a geometry-selected "
            "coefficient, finite re-solved orbit, free motion, and all "
            "regression gates."),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    efficient_main()
