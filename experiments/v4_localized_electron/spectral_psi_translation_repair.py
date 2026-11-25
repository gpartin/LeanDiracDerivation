"""Screen the positive spectral square ||(D19-27) Psi||^2 for pinning repair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import canonical_first_shell_phase_embedding as phase


HERE = Path(__file__).resolve().parent
DT = 1.0 / 1024.0


def spectral_psi_force_matrix(side: int) -> np.ndarray:
    dimension = 12 * side**3
    matrix = np.zeros((dimension, dimension))
    basis = np.zeros(dimension)
    for column in range(dimension):
        basis[column] = 1.0
        dpsi, dchi = rank.native.unpack(basis, side)
        defect = rank.native.c19_dirichlet(dpsi) - 27 * dpsi
        psi_force = -(rank.native.c19_dirichlet(defect) - 27 * defect)
        matrix[:, column] = rank.pack(psi_force, np.zeros_like(dchi))
        basis[column] = 0.0
    return matrix


def main() -> None:
    rank.lock_inertia()
    phase.DT = DT
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(phase.SIDE, amplitude_sq)
    canonical = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)[0]
    repair = spectral_psi_force_matrix(phase.SIDE)
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    rotation = rank.native.rotation_matrix(
        phase.SIDE, np.exp(1j * theta), 1.0 + 0.0j, 1)
    inverse = rotation.T
    translations = rank.native.translation_vectors(psi, chi)
    residuals, columns = [], []
    rows = []
    for axis in range(3):
        tangent = translations[:, axis]
        kinematic = ((2 * inverse - inverse @ inverse - np.eye(
            inverse.shape[0])) @ tangent) / DT**2
        incumbent = kinematic + inverse @ (canonical @ tangent)
        response = inverse @ (repair @ tangent)
        residuals.append(incumbent)
        columns.append(response)
        rows.append({
            "axis": axis,
            "incumbent_pinning_norm": float(np.linalg.norm(incumbent)),
            "unit_repair_response_norm": float(np.linalg.norm(response)),
            "repair_correlation": float(np.dot(response, -incumbent)
                / max(np.linalg.norm(response) * np.linalg.norm(incumbent),
                      1e-300)),
        })
    target = -np.concatenate(residuals)
    column = np.concatenate(columns)
    coefficient = float(np.dot(column, target) / np.dot(column, column))
    residual = column * coefficient - target
    relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    positive = coefficient >= 0
    for row, incumbent, response in zip(rows, residuals, columns):
        row["fitted_axis_relative_residual"] = float(np.linalg.norm(
            incumbent + coefficient * response)
            / max(np.linalg.norm(incumbent), 1e-300))
    result = {
        "status": "V4_SPECTRAL_PSI_TRANSLATION_REPAIR",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "candidate_energy": "eta/2 * ||(D19-27) Psi||^2",
        "geometry_provenance": (
            "27 is the proved C19 Dirichlet eigenvalue of the localized "
            "eight-corner twist and equals the full 3^3 cube count"),
        "exact_rest_first_variation": "zero because (D19-27) Psi_rest = 0",
        "force_reaches": "two successive C19 hops",
        "rows": rows,
        "least_squares_eta": coefficient,
        "eta_nonnegative": positive,
        "combined_relative_residual": relative,
        "explained_squared_fraction": float(1 - relative**2),
        "verdict": (
            "POSITIVE_SPECTRAL_SQUARE_REPAIRS_TRANSLATION_PINNING"
            if positive and relative < 1e-8 else
            "POSITIVE_SPECTRAL_SQUARE_DOES_NOT_CLOSE_TRANSLATION_PINNING"),
        "claim_boundary": (
            "Linear zero-mode inverse screen only. Even a pass would require "
            "a one-hop reduction or an explicit framework/stencil change, "
            "the complete spectrum, nonlinear movement, and Lean proofs."),
    }
    output = HERE / "spectral_psi_translation_repair.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
