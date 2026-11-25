"""Can a direct positive chi-C19 vacuum stiffness preserve the carrier rest?

The native C27/gyro candidate leaves a large flat exterior-chi Jordan sector.
The smallest positive cure is the ordinary cubic C19 Dirichlet energy of chi.
It is not admitted unless its carrier torque and the C27 W torque both lie in
one reciprocal gyroscopic response span.  This probe augments the reduced
local/body W one-forms with their C19-bond common/relative counterparts and
solves the exact rest-force range problem before any coefficient scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import c27_body_gyro_basis_probe as body


HERE = Path(__file__).resolve().parent
BASIS_NAMES = body.BASIS_NAMES + (
    "c19_common_area_velocity",
    "c19_relative_area_velocity",
)


def squared_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.conj(value) * value)


def direct_chi_c19_potential(raw: torch.Tensor) -> torch.Tensor:
    _, chi = body.fields(raw)
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in body.base.inverse.rank.native.DIRECTIONS:
        difference = torch.roll(
            chi, shifts=offset, dims=(0, 1, 2)) - chi
        result = result + weight * torch.sum(squared_norm(difference))
    # Directed C19 offsets count undirected bonds twice; the conventional
    # positive bond energy includes a further factor 1/2.
    return result / 4


def c19_pair(area: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=torch.float64)
    for offset, weight in body.base.inverse.rank.native.DIRECTIONS:
        neighbor_area = torch.roll(area, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2))
        first = torch.sum(torch.conj(area) * neighbor_velocity, dim=-1)
        second = torch.sum(
            torch.conj(neighbor_area) * velocity, dim=-1)
        result = result + weight * torch.sum(torch.imag(first + second))
    return body.base.inverse.DT * result / 4


def gyro_basis(left: torch.Tensor, right: torch.Tensor):
    area, common, relative = body.midpoint_kinematics(left, right)
    return body.gyro_basis(left, right) + (
        c19_pair(area, common),
        c19_pair(area, relative),
    )


def interval_action(left, right, body_strength, chi_strength, coefficients):
    terms = gyro_basis(left, right)
    gyro = sum(coefficients[index] * term
               for index, term in enumerate(terms))
    return (
        body.base.inverse.interval_action(left, right)
        - body.base.inverse.DT * body_strength
        * (body.c27.body_potential(left) + body.c27.body_potential(right)) / 2
        - body.base.inverse.DT * chi_strength
        * (direct_chi_c19_potential(left)
           + direct_chi_c19_potential(right)) / 2
        + gyro
    )


def three_layer_action(previous, current, following, body_strength,
                       chi_strength, coefficients):
    return (
        interval_action(previous, current, body_strength, chi_strength,
                        coefficients)
        + interval_action(current, following, body_strength, chi_strength,
                          coefficients)
    )


def make_del(body_strength, chi_strength, coefficients):
    values = torch.tensor(coefficients, dtype=torch.float64)
    action = lambda pm, q, pp: three_layer_action(
        pm, q, pp, body_strength, chi_strength, values)
    return grad(action, argnums=1)


def main() -> None:
    body.base.inverse.rank.lock_inertia()
    body.base.inverse.rest.SIDE = body.base.inverse.SIDE
    body.base.inverse.rest.DT = body.base.inverse.DT
    previous, current, following, _ = body.base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))
    zero = np.zeros(len(BASIS_NAMES))
    baseline_del = make_del(0.0, 0.0, zero)
    baseline = baseline_del(*tensors).detach().numpy()
    targets = []
    for body_strength, chi_strength in ((1.0, 0.0), (0.0, 1.0)):
        unit = make_del(body_strength, chi_strength, zero)(
            *tensors).detach().numpy()
        targets.append(unit - baseline)
    target_matrix = np.column_stack(targets)
    columns = []
    for index in range(len(BASIS_NAMES)):
        unit = zero.copy()
        unit[index] = 1.0
        response = make_del(0.0, 0.0, unit)(*tensors).detach().numpy()
        columns.append(response - baseline)
    response_matrix = np.column_stack(columns)
    coefficients, _, rank, singular = np.linalg.lstsq(
        response_matrix, -target_matrix, rcond=1e-12)
    remainder = response_matrix @ coefficients + target_matrix
    relative = np.linalg.norm(remainder, axis=0) / np.maximum(
        np.linalg.norm(target_matrix, axis=0), 1e-300)
    result = {
        "status": "V4_C27_CHI_C19_REST_COMPLETION_RANGE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis_names": list(BASIS_NAMES),
        "basis_force_rank": int(rank),
        "basis_force_singular_values": [float(value) for value in singular],
        "target_names": ["joint_W_C27", "direct_chi_C19"],
        "coefficient_columns": {
            target: {
                name: float(coefficients[row, column])
                for row, name in enumerate(BASIS_NAMES)
            }
            for column, target in enumerate(("joint_W_C27", "direct_chi_C19"))
        },
        "relative_force_remainders": {
            "joint_W_C27": float(relative[0]),
            "direct_chi_C19": float(relative[1]),
        },
        "both_targets_in_gyro_range": bool(np.max(relative) < 1e-8),
        "verdict": (
            "CHI_C19_VACUUM_STIFFNESS_HAS_RECIPROCAL_REST_COMPLETION"
            if np.max(relative) < 1e-8 else
            "CHI_C19_VACUUM_STIFFNESS_OUTSIDE_TESTED_GYRO_RANGE"),
        "claim_boundary": (
            "Exact-rest force-range test only. It does not select the chi "
            "stiffness, prove spectrum improvement, retain transport rank, "
            "or establish nonlinear movement."),
    }
    output = HERE / "c27_chi_c19_rest_completion_probe.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "basis_force_rank": result["basis_force_rank"],
        "relative_force_remainders": result["relative_force_remainders"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
