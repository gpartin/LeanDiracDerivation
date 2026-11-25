"""Compress the joint connection into the two cubical copy tangent spaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vmap

import plaquette_factorization_completion_probe as base
import c19_bargmann_joint_connection_gate as joint


HERE = Path(__file__).resolve().parent


def connection_three(previous: torch.Tensor, current: torch.Tensor,
                     following: torch.Tensor) -> torch.Tensor:
    return (joint.connection_term(previous, current)
            + joint.connection_term(current, following))


connection_del = grad(connection_three, argnums=1)


def images(previous: torch.Tensor, current: torch.Tensor,
           following: torch.Tensor, targets: np.ndarray) -> np.ndarray:
    vectors = torch.tensor(targets.T, dtype=torch.float64)
    function = lambda value: connection_del(previous, current, value)
    return vmap(lambda vector: jvp(
        function, (following,), (vector,))[1], chunk_size=8)(
            vectors).detach().numpy().T


def main() -> None:
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, theta = base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))

    rows = []
    for axis in range(3):
        labels, targets = base.component.component_targets(
            axis, np.exp(1j * theta))
        operator_images = images(*layers, targets)
        gram = targets.T @ targets
        compressed = np.linalg.pinv(gram, rcond=1e-12) @ (
            targets.T @ operator_images)
        projected = targets @ compressed
        scale = max(float(np.linalg.norm(operator_images)), 1e-300)
        plus_minus = compressed[:4, 4:]
        minus_plus = compressed[4:, :4]
        rows.append({
            "axis": axis,
            "labels": labels,
            "operator_image_norm": scale,
            "relative_outside_eight_component_tangent": float(
                np.linalg.norm(operator_images - projected) / scale),
            "relative_plus_from_minus_block": float(
                np.linalg.norm(targets[:, :4] @ plus_minus) / scale),
            "relative_minus_from_plus_block": float(
                np.linalg.norm(targets[:, 4:] @ minus_plus) / scale),
            "compressed_matrix": compressed.tolist(),
        })

    copy_preserved = all(
        row["relative_plus_from_minus_block"] < 1e-8
        and row["relative_minus_from_plus_block"] < 1e-8
        for row in rows)
    tangent_closed = all(
        row["relative_outside_eight_component_tangent"] < 1e-8
        for row in rows)
    result = {
        "status": "V4_C19_BARGMANN_CONNECTION_COPY_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "copy_preserved": copy_preserved,
        "eight_component_tangent_closed": tangent_closed,
        "verdict": (
            "BARGMANN_CONNECTION_PRESERVES_BOTH_COPIES"
            if copy_preserved and tangent_closed else
            "BARGMANN_CONNECTION_FAILS_COPY_OR_TANGENT_CLOSURE"),
        "claim_boundary": (
            "Euclidean compression of the exact action Hessian on the "
            "prepared phase-ramp tangent. It is not a proof of the full "
            "localized phase-space embedding or a Dirac operator identity."),
    }
    output = HERE / "c19_bargmann_connection_copy_gate.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
