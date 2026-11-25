"""Rank-test a bond-direction-resolved native Psi--chi metric connection.

At each site the live fields construct a Hermitian C3 shape tensor:

  Q = chi chi^dagger/chi0^2 + Psi Psi^dagger/chi0^2
      + (chi wedge Psi)(chi wedge Psi)^dagger/chi0^4.

For a C19 bond vector n, b_i(n)=Q_i n is the internal direction assigned to
that physical bond.  The anti-Hermitian bond rotation

  K_ij(n)=b_j b_i^dagger - b_i b_j^dagger

is summed over the complete C19 orbit and placed inside positive kinetic
squares for both native fields.  No corner label, copy projector, Dirac
matrix, target direction, or runtime rule appears in the action.

This script only asks whether the resulting reciprocal action has rank in all
three charge-preserving transport directions at the exact rest orbit.
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
ACTIVATION = "none"
FACE_SCALE = 1.0
EDGE_SCALE = 1.0


def norm_sq(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(value) * value, dim=-1))


def outer(value: torch.Tensor) -> torch.Tensor:
    return value[..., :, None] * torch.conj(value[..., None, :])


def live_frame(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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
    return vector_part + area_part


def direction_resolved_connection(frame: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(frame)
    for offset, weight in inverse.rank.native.DIRECTIONS:
        orbit_scale = (FACE_SCALE if sum(abs(value) for value in offset) == 1
                       else EDGE_SCALE)
        direction = torch.tensor(offset, dtype=frame.dtype)
        direction = direction / torch.linalg.vector_norm(direction)
        neighbor = torch.roll(frame, shifts=offset, dims=(0, 1, 2))
        local_axis = torch.einsum("...ab,b->...a", frame, direction)
        neighbor_axis = torch.einsum("...ab,b->...a", neighbor, direction)
        forward = neighbor_axis[..., :, None] * torch.conj(
            local_axis[..., None, :])
        backward = local_axis[..., :, None] * torch.conj(
            neighbor_axis[..., None, :])
        result = result + orbit_scale * weight * (forward - backward)
    return result / inverse.rank.C19_WEIGHT_SUM


def connection_kinetic_delta(left: torch.Tensor,
                             right: torch.Tensor) -> torch.Tensor:
    left_psi, left_chi = inverse.current_audit.mold.unpack_tensor(
        left, inverse.SIDE)
    right_psi, right_chi = inverse.current_audit.mold.unpack_tensor(
        right, inverse.SIDE)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_velocity = (right_psi - left_psi) / inverse.DT
    chi_velocity = (right_chi - left_chi) / inverse.DT
    connection = direction_resolved_connection(live_frame(left, right))
    transfer = None
    if ACTIVATION in ("density_rate", "density_rate_square"):
        rho_psi = norm_sq(psi)
        rho_dot = 2 * torch.real(torch.sum(
            torch.conj(psi) * psi_velocity, dim=-1))
        regulator = inverse.rank.KAPPA * inverse.rank.R**2
        transfer = rho_dot / (
            2 * inverse.rank.R * (rho_psi + regulator))
    connection_psi = torch.einsum(
        "...ab,...b->...a", connection, psi)
    connection_chi = torch.einsum(
        "...ab,...b->...a", connection, chi)
    canonical = (torch.sum(norm_sq(psi_velocity))
                 + inverse.rank.B * torch.sum(norm_sq(chi_velocity)))
    if ACTIVATION == "density_rate_square":
        routed_psi = transfer[..., None] * connection_psi
        routed_chi = transfer[..., None] * connection_chi
        candidate = canonical + STRENGTH * (
            torch.sum(norm_sq(routed_psi))
            + inverse.rank.B * torch.sum(norm_sq(routed_chi)))
    else:
        if ACTIVATION == "density_rate":
            connection_psi = transfer[..., None] * connection_psi
            connection_chi = transfer[..., None] * connection_chi
        covariant_psi = psi_velocity + STRENGTH * connection_psi
        covariant_chi = chi_velocity + STRENGTH * connection_chi
        candidate = (torch.sum(norm_sq(covariant_psi))
                     + inverse.rank.B * torch.sum(norm_sq(covariant_chi)))
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
    elif argument == "following":
        function = lambda value: del_residual(previous, current, value)
        point = following
    else:
        raise ValueError(argument)
    images = vmap(
        lambda vector: jvp(function, (point,), (vector,))[1],
        chunk_size=8)(vector_tensor)
    return images.detach().numpy().T


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
    value, derivative = jvp(
        lambda a, b: density_tensor(a, b, mode), (l, r), (dl, dr))
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
        - center_derivative(current_psi, dcurrent_psi))
    return np.concatenate((charges, displacement))


def main() -> None:
    global ACTIVATION
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation",
        choices=("none", "density_rate", "density_rate_square"),
        default="density_rate")
    args = parser.parse_args()
    ACTIVATION = args.activation
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (previous, current, following))
    frames = (live_frame(base[0], base[1]), live_frame(base[1], base[2]))
    connection_norms = [float(torch.linalg.vector_norm(
        direction_resolved_connection(frame)).detach()) for frame in frames]
    interval_deltas = [float(connection_kinetic_delta(
        base[0], base[1]).detach()), float(connection_kinetic_delta(
            base[1], base[2]).detach())]
    rest_del = del_residual(*base).detach().numpy()

    dimension = previous.size
    previous_block = batched_jvp("previous", base, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = component.component_targets(
            axis, np.exp(1j * theta))
        images = batched_jvp("following", base, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            response(previous, current, following,
                     launches[:, column], targets[:, column])
            for column in range(targets.shape[1])])
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
            "fixed_charge_center_singular_values": singular.tolist(),
            "relative_third_singular_value": float(
                singular[2] / max(singular[0], 1e-30)),
            "missing_physical_direction": left[:, -1].tolist(),
            "response_matrix": responses.tolist(),
        })

    rank_pass = all(row["fixed_charge_center_rank"] == 3
                    and row["relative_third_singular_value"] > 1e-6
                    for row in rows)
    rest_pass = float(np.linalg.norm(rest_del) / np.sqrt(rest_del.size)) < 1e-8
    result = {
        "status": "V4_BOND_DIRECTION_NATIVE_FRAME_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate": (
            "positive Psi/chi kinetic squares using the anti-Hermitian C19 "
            "connection of bond-resolved live shape tensors Q(Psi,chi,W)"),
        "activation": ACTIVATION,
        "activation_definition": (
            "rhoPsi_dot/[2 chi0 (rhoPsi+kappa chi0^2)]"
            if ACTIVATION in ("density_rate", "density_rate_square")
            else "one"),
        "screen_strength": STRENGTH,
        "coefficient_admitted": False,
        "real_action": True,
        "positive_kinetic_square": True,
        "reciprocal_variation": True,
        "complete_C19_orbit_sum": True,
        "cubic_covariance": True,
        "covariance_identity": (
            "Under a real cube rotation R, Q->R Q R^T and n->R n, hence "
            "Q n->R Q n and K->R K R^T before the complete C19 sum."),
        "rest_connection_norms": connection_norms,
        "rest_interval_action_deltas": interval_deltas,
        "rest_DEL_rms": float(np.linalg.norm(rest_del)
                              / np.sqrt(rest_del.size)),
        "axis_rows": rows,
        "all_axes_full_rank": rank_pass,
        "verdict": (
            "BOND_DIRECTION_NATIVE_FRAME_RANK_PASS"
            if rank_pass and rest_pass
            else "BOND_DIRECTION_NATIVE_FRAME_RANK_FAIL"),
        "claim_boundary": (
            "Exact-rest linear response rank on the side-five carrier. A "
            "rank pass would not select a coefficient, prove positive full "
            "mass after expansion, or close nonlinear V4-006."),
    }
    output_names = {
        "none": "bond_direction_native_frame_rank.json",
        "density_rate": "bond_direction_native_frame_transfer_rank.json",
        "density_rate_square": (
            "bond_direction_native_frame_transfer_square_rank.json"),
    }
    output = HERE / output_names[ACTIVATION]
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "axis_rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
