"""Decompose the canonical localized translation pinning force by action term."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import canonical_first_shell_phase_embedding as phase


HERE = Path(__file__).resolve().parent
DT = 1.0 / 1024.0


def dominant_channel(raw: np.ndarray) -> dict:
    psi, chi = rank.native.unpack(raw, phase.SIDE)
    psi_power = np.sum(abs(psi)**2, axis=-1)
    chi_power = rank.B * np.sum(abs(chi)**2, axis=-1)
    if np.max(psi_power) >= np.max(chi_power):
        field, values, site_power = "Psi", psi, psi_power
    else:
        field, values, site_power = "chi", chi, chi_power
    site = tuple(int(value) for value in np.unravel_index(
        np.argmax(site_power), site_power.shape))
    component = int(np.argmax(abs(values[site])))
    total = max(float(np.sum(psi_power) + np.sum(chi_power)), 1e-300)
    return {
        "field": field,
        "site": list(site),
        "component": component,
        "psi_power_fraction": float(np.sum(psi_power) / total),
        "chi_power_fraction": float(np.sum(chi_power) / total),
    }


def main() -> None:
    rank.lock_inertia()
    phase.DT = DT
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(phase.SIDE, amplitude_sq)
    matrices = rank.native.tangent_matrices(
        psi, chi, 0.0, 0.0, 1.0, 1.0)
    total_acceleration, local_acceleration, area_acceleration = matrices[:3]
    theta = float(np.arccos(1 - DT**2 * rank.R**2 / 2))
    rotation = rank.native.rotation_matrix(
        phase.SIDE, np.exp(1j * theta), 1.0 + 0.0j, 1)
    inverse = rotation.T
    translations = rank.native.translation_vectors(psi, chi)

    rows = []
    fit_columns = []
    fit_targets = []
    for axis in range(3):
        tangent = translations[:, axis]
        kinematic = ((2 * inverse - inverse @ inverse - np.eye(
            inverse.shape[0])) @ tangent) / DT**2
        local = inverse @ (local_acceleration @ tangent)
        area = inverse @ (area_acceleration @ tangent)
        total = kinematic + inverse @ (total_acceleration @ tangent)
        scale = max(float(np.linalg.norm(total)), 1e-300)
        fit_columns.append(np.column_stack((local, area)))
        fit_targets.append(-kinematic)
        rows.append({
            "axis": axis,
            "pinning_norm": float(np.linalg.norm(total)),
            "kinematic_norm": float(np.linalg.norm(kinematic)),
            "local_norm": float(np.linalg.norm(local)),
            "area_norm": float(np.linalg.norm(area)),
            "local_correlation_with_pinning": float(
                np.dot(local, total) / max(np.linalg.norm(local) * scale, 1e-300)),
            "area_correlation_with_pinning": float(
                np.dot(area, total) / max(np.linalg.norm(area) * scale, 1e-300)),
            "reconstruction_relative_error": float(np.linalg.norm(
                total - (kinematic + local + area)) / scale),
            "dominant_pinning_channel": dominant_channel(total),
        })

    matrix = np.vstack(fit_columns)
    target = np.concatenate(fit_targets)
    coefficients, _residuals, fit_rank, singular = np.linalg.lstsq(
        matrix, target, rcond=1e-12)
    fitted = matrix @ coefficients - target
    relative = float(np.linalg.norm(fitted) / np.linalg.norm(target))
    incumbent = matrix @ np.ones(2) - target
    incumbent_relative = float(
        np.linalg.norm(incumbent) / np.linalg.norm(target))
    result = {
        "status": "V4_CANONICAL_TRANSLATION_FORCE_RCA",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "dt": DT,
        "rows": rows,
        "two_term_inverse_fit": {
            "basis": ["local_potential", "C19_area_bond"],
            "response_rank": int(fit_rank),
            "singular_values": [float(value) for value in singular],
            "incumbent_coefficients": [1.0, 1.0],
            "least_squares_coefficients": [float(value) for value in coefficients],
            "incumbent_relative_residual": incumbent_relative,
            "best_relative_residual": relative,
            "explained_squared_fraction": float(1 - relative**2),
        },
        "verdict": (
            "EXISTING_TWO_ACTION_TERMS_CAN_CANCEL_TRANSLATION_PINNING"
            if relative < 1e-8 else
            "EXISTING_TWO_ACTION_TERMS_CANNOT_CANCEL_TRANSLATION_PINNING"),
        "claim_boundary": (
            "Continuous linear inverse response at the canonical rest orbit. "
            "Changing either coefficient would require re-solving the rest "
            "orbit and all V4/Lean gates; the least-squares values are not "
            "admitted coefficients."),
    }
    output = HERE / "canonical_translation_force_rca.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
