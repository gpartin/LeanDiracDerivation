"""Screen a native joint-field frame connection for the missing direction.

The eight prepared corner rays have a basis-free tetrahedral projector
geometry, but the current area variable ``chi wedge Psi`` collapses that
geometry to one common internal normal.  This probe tests the smallest
positive kinetic completion suggested by that diagnosis.

At each site define the normalized joint Hermitian tensor

    Q_i = (Psi_i Psi_i^dagger + chi_i chi_i^dagger) / chi0^2

and the anti-Hermitian C19-average connection

    A_i = (1/24) sum_n w_n [Q_i, Q_(i+n)].

The interval kinetic energy uses

    D_t Psi = Psi_dot + g A Psi_mid,
    D_t chi = chi_dot + g A chi_mid.

The total kinetic square is positive, real, reciprocal after variation, and
contains no corner labels, Pauli/Dirac matrices, selected copy or preferred
axis.  Q is averaged from the two endpoint density matrices rather than
formed from the averaged fields, so an exact common phase rotation does not
spuriously change the local frame.

This is only a response-rank screen.  A nonzero result would still require a
coefficient derivation, free rest solve, full spectrum, nonlinear handoff,
and the same-state Dirac current audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad, jvp, vmap

import localized_carrier_contrast_component_phase_ramp as component
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent
STRENGTH = 1.0
AREA_WEIGHT = 0.0


def outer(value: torch.Tensor) -> torch.Tensor:
    return value[..., :, None] * torch.conj(value[..., None, :])


def joint_frame(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = inverse.current_audit.mold.unpack_tensor(
        left, inverse.SIDE)
    right_psi, right_chi = inverse.current_audit.mold.unpack_tensor(
        right, inverse.SIDE)
    left_area = torch.linalg.cross(left_chi, left_psi, dim=-1)
    right_area = torch.linalg.cross(right_chi, right_psi, dim=-1)
    vector_part = (
        outer(left_psi) + outer(left_chi)
        + outer(right_psi) + outer(right_chi)
    ) / (2 * inverse.rank.R**2)
    area_part = (
        outer(left_area) + outer(right_area)
    ) / (2 * inverse.rank.R**4)
    return vector_part + AREA_WEIGHT * area_part


def c19_connection(frame: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(frame)
    for offset, weight in inverse.rank.native.DIRECTIONS:
        neighbor = torch.roll(frame, shifts=offset, dims=(0, 1, 2))
        result = result + weight * (
            frame @ neighbor - neighbor @ frame)
    return result / inverse.rank.C19_WEIGHT_SUM


def connection_kinetic_delta(
        left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = inverse.current_audit.mold.unpack_tensor(
        left, inverse.SIDE)
    right_psi, right_chi = inverse.current_audit.mold.unpack_tensor(
        right, inverse.SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_velocity = (right_psi - left_psi) / inverse.DT
    chi_velocity = (right_chi - left_chi) / inverse.DT
    connection = c19_connection(joint_frame(left, right))
    connection_psi = torch.einsum("...ab,...b->...a", connection, psi)
    connection_chi = torch.einsum("...ab,...b->...a", connection, chi)
    covariant_psi = psi_velocity + STRENGTH * connection_psi
    covariant_chi = chi_velocity + STRENGTH * connection_chi

    def norm_sq(value: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.real(torch.conj(value) * value))

    candidate = (
        norm_sq(covariant_psi)
        + inverse.rank.B * norm_sq(covariant_chi)
    )
    canonical = (
        norm_sq(psi_velocity)
        + inverse.rank.B * norm_sq(chi_velocity)
    )
    return inverse.DT * (candidate - canonical) / 2


def interval_action(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return inverse.interval_action(left, right) + connection_kinetic_delta(
        left, right)


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current) + interval_action(
        current, following)


del_residual = grad(three_layer_action, argnums=1)
right_momentum = grad(interval_action, argnums=1)


def batched_jvp(argument: str, base: tuple[torch.Tensor, ...],
                vectors: np.ndarray) -> np.ndarray:
    previous, current, following = base
    vector_tensor = torch.tensor(vectors.T, dtype=torch.float64)
    if argument == "previous":
        function = lambda value: del_residual(value, current, following)
        point = previous
    else:
        function = lambda value: del_residual(previous, current, value)
        point = following
    return vmap(
        lambda vector: jvp(function, (point,), (vector,))[1],
        chunk_size=8)(vector_tensor).detach().numpy().T


def density_tensor(left: torch.Tensor, right: torch.Tensor,
                   mode: str) -> torch.Tensor:
    momentum = right_momentum(left, right)
    psi, chi = inverse.current_audit.mold.unpack_tensor(right, inverse.SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator = inverse.current_audit.pack_tensor(
        1j * psi if mode == "Psi" else zero_psi,
        1j * chi if mode == "chi" else zero_chi)
    return (momentum * generator).reshape(
        inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3).sum(dim=(-1, -2))


def density_and_derivative(left: np.ndarray, right: np.ndarray,
                           dleft: np.ndarray | None,
                           dright: np.ndarray | None,
                           mode: str) -> tuple[np.ndarray, np.ndarray]:
    l = torch.tensor(left, dtype=torch.float64)
    r = torch.tensor(right, dtype=torch.float64)
    dl = torch.zeros_like(l) if dleft is None else torch.tensor(
        dleft, dtype=torch.float64)
    dr = torch.zeros_like(r) if dright is None else torch.tensor(
        dright, dtype=torch.float64)
    function = lambda a, b: density_tensor(a, b, mode)
    value, derivative = jvp(function, (l, r), (dl, dr))
    return value.detach().numpy(), derivative.detach().numpy()


def center_derivative(density: np.ndarray,
                      derivative: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    dtotal = float(np.sum(derivative))
    coordinates = np.indices(density.shape)
    result = []
    for axis in range(3):
        moment = float(np.sum(coordinates[axis] * density))
        dmoment = float(np.sum(coordinates[axis] * derivative))
        result.append((dmoment * total - moment * dtotal) / total**2)
    return np.asarray(result)


def response(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray, launch: np.ndarray,
             target: np.ndarray) -> np.ndarray:
    current_psi, dcurrent_psi = density_and_derivative(
        previous, current, launch, None, "Psi")
    following_psi, dfollowing_psi = density_and_derivative(
        current, following, None, target, "Psi")
    _, dcurrent_chi = density_and_derivative(
        previous, current, launch, None, "chi")
    _, dfollowing_chi = density_and_derivative(
        current, following, None, target, "chi")
    charges = np.asarray((
        (np.sum(dcurrent_psi) + np.sum(dfollowing_psi)) / 2,
        (np.sum(dcurrent_chi) + np.sum(dfollowing_chi)) / 2,
    ))
    displacement = (
        center_derivative(following_psi, dfollowing_psi)
        - center_derivative(current_psi, dcurrent_psi)
    )
    return np.concatenate((charges, displacement))


def main() -> None:
    global STRENGTH, AREA_WEIGHT
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--area-weight", type=float, default=0.0)
    args = parser.parse_args()
    STRENGTH = args.strength
    AREA_WEIGHT = args.area_weight

    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    frames = [joint_frame(base[0], base[1]), joint_frame(base[1], base[2])]
    connection_norms = [
        float(torch.linalg.vector_norm(c19_connection(frame)).detach())
        for frame in frames
    ]
    interval_deltas = [
        float(connection_kinetic_delta(base[0], base[1]).detach()),
        float(connection_kinetic_delta(base[1], base[2]).detach()),
    ]
    rest_del = del_residual(*base).detach().numpy()

    dimension = previous.size
    previous_block = batched_jvp("previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = component.component_targets(axis, np.exp(1j * theta))
        images = batched_jvp("following", base, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(previous, current, following,
                     launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:] @ null
        left, singular, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-8 * max(singular[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [
                float(value) for value in singular],
            "relative_third_singular_value": float(
                singular[2] / max(singular[0], 1e-30)),
            "missing_physical_direction": [
                float(value) for value in left[:, -1]],
            "response_matrix": responses.tolist(),
        })

    robust = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    result = {
        "status": "V4_TETRAHEDRAL_FRAME_CONNECTION_TRANSPORT_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "replace the local Psi/chi kinetic squares by positive squares "
            "covariant under the joint-projector C19 commutator connection"),
        "screen_strength": STRENGTH,
        "area_projector_weight": AREA_WEIGHT,
        "cubic_covariance": (
            "Q transforms as R Q R^dagger because Psi, chi and chi wedge "
            "Psi are native C3 vectors; the complete C19 commutator sum "
            "therefore transforms as R A R^dagger"),
        "coefficient_provenance": (
            "C19 weight normalization 1/24 and chi0 normalization are exact; "
            "g is a response-rank screen and is not admitted"),
        "rest_connection_norms": connection_norms,
        "rest_interval_action_deltas": interval_deltas,
        "candidate_rest_DEL_rms": float(
            np.linalg.norm(rest_del) / np.sqrt(rest_del.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": robust,
        "verdict": (
            "TETRAHEDRAL_FRAME_CONNECTION_SUPPLIES_THIRD_TRANSPORT_CHANNEL"
            if robust else
            "TETRAHEDRAL_FRAME_CONNECTION_DOES_NOT_CLOSE_TRANSPORT"),
        "claim_boundary": (
            "Linear response-rank screen only. A pass still requires a native "
            "coefficient, unrestricted rest solve, bounded Hamiltonian/full "
            "spectrum, nonlinear movement, current matching, and Dirac gates."),
    }
    strength_suffix = ("unit" if STRENGTH == 1 else
                       f"strength_{STRENGTH:g}".replace(".", "p"))
    area_suffix = ("" if AREA_WEIGHT == 0 else
                   f"_area_{AREA_WEIGHT:g}".replace(".", "p"))
    suffix = strength_suffix + area_suffix
    output = HERE / f"tetrahedral_frame_connection_probe_{suffix}.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rest_connection_norms": connection_norms,
        "rest_interval_action_deltas": interval_deltas,
        "candidate_rest_DEL_rms": result["candidate_rest_DEL_rms"],
        "ranks": [row["fixed_charge_center_rank"] for row in rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
