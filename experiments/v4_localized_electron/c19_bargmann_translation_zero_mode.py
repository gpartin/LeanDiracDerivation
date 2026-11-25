"""Test the joint Bargmann connection on the exact translation obstruction.

The candidate connection has zero discrete Euler--Lagrange force on the rest
orbit.  Here we differentiate the complete three-layer DEL equation while all
three layers are displaced by the same centered lattice translation.  A true
translation zero mode requires the resulting derivative to vanish.  The
single connection coefficient is solved from all three axes together; no
Dirac operator or target trajectory enters this calculation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import c19_bargmann_joint_connection_gate as joint
import plaquette_factorization_completion_probe as base


HERE = Path(__file__).resolve().parent


def three_layer_term(previous: torch.Tensor, current: torch.Tensor,
                     following: torch.Tensor, kind: str) -> torch.Tensor:
    interval = (joint.canonical_interval_action if kind == "canonical"
                else joint.connection_term)
    return interval(previous, current) + interval(current, following)


def del_function(kind: str):
    return grad(lambda p, c, f: three_layer_term(p, c, f, kind), argnums=1)


def translated_tangent(raw: np.ndarray, axis: int) -> np.ndarray:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        torch.tensor(raw, dtype=torch.float64), base.inverse.SIDE)
    dpsi = (torch.roll(psi, 1, axis) - torch.roll(psi, -1, axis)) / 2
    dchi = (torch.roll(chi, 1, axis) - torch.roll(chi, -1, axis)) / 2
    tangent = base.inverse.current_audit.pack_tensor(dpsi, dchi)
    tangent = tangent.detach().numpy()
    return tangent / np.linalg.norm(tangent)


def directional_del(kind: str, layers: tuple[np.ndarray, ...],
                    tangents: tuple[np.ndarray, ...]) -> np.ndarray:
    points = tuple(torch.tensor(value, dtype=torch.float64) for value in layers)
    directions = tuple(torch.tensor(value, dtype=torch.float64)
                       for value in tangents)
    return jvp(del_function(kind), points, directions)[1].detach().numpy()


def main() -> None:
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, _theta = base.inverse.rest_layers()
    layers = (previous, current, following)

    incumbent_columns = []
    connection_columns = []
    rows = []
    for axis in range(3):
        tangents = tuple(translated_tangent(layer, axis) for layer in layers)
        incumbent = directional_del("canonical", layers, tangents)
        candidate = directional_del("connection", layers, tangents)
        incumbent_columns.append(incumbent)
        connection_columns.append(candidate)
        rows.append({
            "axis": axis,
            "canonical_translation_DEL_norm": float(np.linalg.norm(incumbent)),
            "unit_connection_response_norm": float(np.linalg.norm(candidate)),
            "connection_correlation_with_required_repair": float(
                np.dot(candidate, -incumbent) / max(
                    np.linalg.norm(candidate) * np.linalg.norm(incumbent),
                    1e-300)),
        })

    target = -np.concatenate(incumbent_columns)
    column = np.concatenate(connection_columns)
    coefficient = float(np.dot(column, target) / np.dot(column, column))
    residual = column * coefficient - target
    relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    for row, incumbent, candidate in zip(
            rows, incumbent_columns, connection_columns):
        row["fitted_axis_relative_residual"] = float(np.linalg.norm(
            incumbent + coefficient * candidate) / np.linalg.norm(incumbent))

    result = {
        "status": "V4_C19_BARGMANN_TRANSLATION_ZERO_MODE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate": "sum Im(B_loop) Im(conj(B_loop) dB_loop) / chi0^12",
        "candidate_rest_DEL": "zero to the separately audited precision",
        "axis_rows": rows,
        "least_squares_connection_coefficient": coefficient,
        "combined_relative_residual": relative,
        "explained_squared_fraction": float(1 - relative**2),
        "verdict": (
            "BARGMANN_CONNECTION_CLOSES_TRANSLATION_ZERO_MODE"
            if relative < 1e-8 else
            "BARGMANN_CONNECTION_DOES_NOT_CLOSE_TRANSLATION_ZERO_MODE"),
        "claim_boundary": (
            "Exact three-layer DEL tangent screen at the localized rest "
            "orbit. A pass would still require symplectic nondegeneracy, "
            "coefficient provenance, full stability, nonlinear movement, "
            "same-state Dirac compression, and Lean proof."),
    }
    output = HERE / "c19_bargmann_translation_zero_mode.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
