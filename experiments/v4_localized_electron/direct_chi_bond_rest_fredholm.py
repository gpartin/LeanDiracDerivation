"""First-order solvability audit for the direct-chi-bond rest branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import direct_chi_bond_rest_continuation as continuation
import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def main() -> None:
    rank.lock_inertia()
    layers, theta = continuation.transfer_rest.rest.orbit_layers()
    reference = layers[1].detach()
    shape = (continuation.SIDE,) * 3 + (4, 3)
    fixed = np.asarray([
        np.ravel_multi_index((0, 0, 0, 1, 0), shape),
        np.ravel_multi_index((2, 2, 2, 3, 2), shape),
    ])
    free_mask = np.ones(reference.numel(), dtype=bool)
    free_mask[fixed] = False
    free = torch.tensor(np.flatnonzero(free_mask), dtype=torch.long)
    base = reference.clone()
    values = torch.tensor(np.concatenate((
        base[free].numpy(), np.asarray([float(theta), 0.0]))),
        dtype=torch.float64, requires_grad=True)

    def residual(parameters: torch.Tensor, strength: float) -> torch.Tensor:
        current = base.clone()
        current[free] = parameters[:-2]
        return continuation.del_residual(
            current, parameters[-2], parameters[-1], strength) / rank.R

    jacobian = torch.autograd.functional.jacobian(
        lambda p: residual(p, 0.0), values, vectorize=True).detach().numpy()
    forcing = (residual(values, 1.0) - residual(values, 0.0)).detach().numpy()
    u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    largest = singular[0]
    thresholds = {}
    selected = None
    for tolerance in (1e-8, 1e-10, 1e-12, 1e-14):
        numerical_rank = int(np.sum(singular > largest * tolerance))
        range_projection = u[:, :numerical_rank] @ (
            u[:, :numerical_rank].T @ forcing)
        obstruction = forcing - range_projection
        fraction = float(np.dot(obstruction, obstruction)
                         / np.dot(forcing, forcing))
        thresholds[f"relative_{tolerance:g}"] = {
            "rank": numerical_rank,
            "nullity": int(jacobian.shape[1] - numerical_rank),
            "cokernel_dimension": int(jacobian.shape[0] - numerical_rank),
            "squared_forcing_obstruction_fraction": fraction,
            "relative_forcing_obstruction": float(np.sqrt(fraction)),
        }
        if tolerance == 1e-12:
            selected = obstruction

    assert selected is not None
    packed = selected.reshape(continuation.SIDE, continuation.SIDE,
                              continuation.SIDE, 4, 3)
    total_power = float(np.sum(packed**2))
    psi_power = float(np.sum(packed[..., :2, :]**2))
    chi_power = float(np.sum(packed[..., 2:, :]**2))
    field_rows = {
        "psi_fraction": psi_power / total_power,
        "chi_fraction": chi_power / total_power,
        "chi_component_fractions": [
            float(np.sum(packed[..., 2:, component]**2) / chi_power)
            for component in range(3)
        ],
    }
    flat_index = int(np.argmax(np.abs(selected)))
    largest_coordinate = np.unravel_index(flat_index, shape)
    largest_entry = {
        "site": [int(value) for value in largest_coordinate[:3]],
        "field_part": ["psi_real", "psi_imag", "chi_real", "chi_imag"][
            largest_coordinate[3]],
        "component": int(largest_coordinate[4]),
        "value": float(selected[flat_index]),
        "fraction_of_obstruction_power": float(
            selected[flat_index]**2 / total_power),
    }
    result = {
        "status": "V4_DIRECT_CHI_BOND_REST_FREDHOLM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "jacobian_shape": list(jacobian.shape),
        "largest_singular_value": float(singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "thresholds": thresholds,
        "selected_relative_tolerance": 1e-12,
        "selected_obstruction_by_field": field_rows,
        "selected_largest_entry": largest_entry,
        "verdict": (
            "SMOOTH_REST_BRANCH_FIRST_ORDER_OBSTRUCTED"
            if thresholds["relative_1e-12"][
                "relative_forcing_obstruction"] > 1e-6
            else "SMOOTH_REST_BRANCH_FIRST_ORDER_SOLVABLE"),
        "claim_boundary": (
            "This is the first-order Fredholm test at the original rest "
            "orbit. It does not exclude a disconnected or finite-amplitude "
            "rest branch."),
    }
    (HERE / "direct_chi_bond_rest_fredholm.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
