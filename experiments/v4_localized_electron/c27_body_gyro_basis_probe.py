"""Screen the complete reduced W-linear gyroscopic basis against C27 torque.

The positive body-diagonal Dirichlet energy of W = chi x Psi supplies the
missing third transport response, but it torques the exact rest orbit.  A
single common-W body one-form has only a small overlap with that torque.  The
lowest-degree phase-sensitive basis also contains the *relative* area
velocity

    Rdot = chi x Psidot - chidot x Psi,

whose variation distinguishes Psi from chi even when the rest values of the
common and relative area velocities coincide.  Cubic symmetry permits local
and body-diagonal contractions of W with the common/relative velocities.

This script calculates the four native force columns, solves the exact linear
range problem for the C27 rest torque, and only then evaluates fixed-charge
transport.  Coefficients are selected solely by native rest cancellation;
no Dirac velocity, spinor matrix, corner label, or preferred axis enters.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import plaquette_factorization_completion_probe as base
import c27_body_diagonal_transport_probe as c27


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))
BASIS_NAMES = (
    "local_common_area_velocity",
    "local_relative_area_velocity",
    "body_common_area_velocity",
    "body_relative_area_velocity",
)
c27.MODE = "joint_W"
COEFFICIENTS = np.zeros(len(BASIS_NAMES), dtype=np.float64)
POTENTIAL_COEFFICIENT = 1.0


def fields(raw: torch.Tensor):
    return base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)


def midpoint_kinematics(left: torch.Tensor, right: torch.Tensor):
    left_psi, left_chi = fields(left)
    right_psi, right_chi = fields(right)
    psi = (left_psi + right_psi) / 2
    chi = (left_chi + right_chi) / 2
    psi_velocity = (right_psi - left_psi) / base.inverse.DT
    chi_velocity = (right_chi - left_chi) / base.inverse.DT
    area = torch.linalg.cross(chi, psi, dim=-1) / base.inverse.rank.R
    common = (
        torch.linalg.cross(chi, psi_velocity, dim=-1)
        + torch.linalg.cross(chi_velocity, psi, dim=-1)
    ) / base.inverse.rank.R
    relative = (
        torch.linalg.cross(chi, psi_velocity, dim=-1)
        - torch.linalg.cross(chi_velocity, psi, dim=-1)
    ) / base.inverse.rank.R
    return area, common, relative


def local_pair(area: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    return base.inverse.DT * torch.sum(torch.imag(
        torch.sum(torch.conj(area) * velocity, dim=-1)))


def body_pair(area: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for offset in BODY_OFFSETS:
        neighbor_area = torch.roll(area, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2))
        first = torch.sum(torch.conj(area) * neighbor_velocity, dim=-1)
        second = torch.sum(
            torch.conj(neighbor_area) * velocity, dim=-1)
        result = result + torch.sum(torch.imag(first + second))
    # Directed offsets count the symmetric bond expression twice.
    return base.inverse.DT * result / 4


def gyro_basis(left: torch.Tensor, right: torch.Tensor):
    area, common, relative = midpoint_kinematics(left, right)
    return (
        local_pair(area, common),
        local_pair(area, relative),
        body_pair(area, common),
        body_pair(area, relative),
    )


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    potential: float | torch.Tensor,
                    coefficients: torch.Tensor) -> torch.Tensor:
    terms = gyro_basis(left, right)
    gyro = sum(coefficients[index] * term
               for index, term in enumerate(terms))
    return (
        base.inverse.interval_action(left, right)
        - base.inverse.DT * potential
        * (c27.body_potential(left) + c27.body_potential(right)) / 2
        + gyro
    )


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor,
                       potential: float | torch.Tensor,
                       coefficients: torch.Tensor) -> torch.Tensor:
    return (
        interval_action(previous, current, potential, coefficients)
        + interval_action(current, following, potential, coefficients)
    )


def make_derivatives(potential: float, coefficients: np.ndarray):
    values = torch.tensor(coefficients, dtype=torch.float64)
    interval = lambda left, right: interval_action(
        left, right, potential, values)
    three = lambda previous, current, following: three_layer_action(
        previous, current, following, potential, values)
    return grad(three, argnums=1), grad(interval, argnums=1)


def main() -> None:
    global COEFFICIENTS
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, theta = base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))

    zero = np.zeros(len(BASIS_NAMES), dtype=np.float64)
    baseline_del, _ = make_derivatives(0.0, zero)
    potential_del, _ = make_derivatives(1.0, zero)
    baseline = baseline_del(*tensors).detach().numpy()
    target = potential_del(*tensors).detach().numpy() - baseline

    columns = []
    for index in range(len(BASIS_NAMES)):
        unit = zero.copy()
        unit[index] = 1.0
        unit_del, _ = make_derivatives(0.0, unit)
        columns.append(unit_del(*tensors).detach().numpy() - baseline)
    matrix = np.column_stack(columns)
    coefficients, _, rank, singular = np.linalg.lstsq(
        matrix, -target, rcond=1e-12)
    COEFFICIENTS = coefficients
    paired_del, paired_momentum = make_derivatives(1.0, coefficients)
    paired = paired_del(*tensors).detach().numpy()
    remaining = paired - baseline
    relative = float(
        np.linalg.norm(remaining) / max(np.linalg.norm(target), 1e-300))

    # Reuse the exact action-derived fixed-charge response screen.
    base.del_residual = paired_del
    base.right_momentum = paired_momentum
    dimension = previous.size
    previous_block = base.batched_jvp("previous", tensors, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = base.component.component_targets(
            axis, np.exp(1j * theta))
        images = base.batched_jvp("following", tensors, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            base.response(previous, current, following,
                          launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:] @ null
        left, values, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-8 * max(values[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(values > threshold)),
            "fixed_charge_center_singular_values": [
                float(value) for value in values],
            "relative_third_singular_value": float(
                values[2] / max(values[0], 1e-30)),
            "missing_physical_direction": [
                float(value) for value in left[:, -1]],
        })

    rank_three = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    rest_closed = relative < 1e-8
    result = {
        "status": "V4_C27_BODY_GYRO_BASIS_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "unit positive joint-W body-diagonal Dirichlet potential plus "
            "the complete reduced local/body common/relative W-linear "
            "gyroscopic basis"),
        "basis_names": list(BASIS_NAMES),
        "basis_force_rank": int(rank),
        "basis_force_singular_values": [float(value) for value in singular],
        "coefficients": {
            name: float(value)
            for name, value in zip(BASIS_NAMES, coefficients)
        },
        "coefficient_rule": (
            "minimum-norm solution cancelling the independently calculated "
            "unit C27 native rest torque; no Dirac target is used"),
        "native_discrete_clock_rate": float(theta / base.inverse.DT),
        "rest_force_relative_remainder": relative,
        "paired_rest_DEL_rms": float(
            np.linalg.norm(paired) / np.sqrt(paired.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": rank_three,
        "verdict": (
            "C27_GYRO_BASIS_PRESERVES_REST_AND_CLOSES_TRANSPORT"
            if rest_closed and rank_three else
            "C27_GYRO_BASIS_INCOMPLETE"),
        "claim_boundary": (
            "Rest-force and linear fixed-charge rank screen only. It does not "
            "prove symplectic nondegeneracy, bounded dynamics, full spectral "
            "stability, nonlinear handoff, or Dirac current matching."),
    }
    output = HERE / "c27_body_gyro_basis_probe.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "basis_force_rank": result["basis_force_rank"],
        "coefficients": result["coefficients"],
        "rest_force_relative_remainder": relative,
        "paired_rest_DEL_rms": result["paired_rest_DEL_rms"],
        "ranks": [row["fixed_charge_center_rank"] for row in rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
