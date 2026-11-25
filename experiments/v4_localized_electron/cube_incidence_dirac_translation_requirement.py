"""Test the low-momentum Dirac translation requirement on the current action.

In the carrier co-rotating frame, the positive-energy Dirac branch has zero
relative frequency at zero momentum and acquires its velocity continuously as
momentum is turned on.  A localized native carrier intended to supply that
branch must therefore have a neutral center-displacement tangent.  This audit
tests that necessary condition on the current research action and asks whether
the already-derived reciprocal action vocabulary can cancel its pinning force.

No Dirac matrix or target trajectory enters the action.  The Dirac equation is
used only to state the zero-momentum translation requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as box
import cube_incidence_dirac_inverse_schur as inverse
import cube_incidence_nonlinear_action as action
import localized_stability as native


HERE = Path(__file__).resolve().parent


def update_blocks(mass, gyro, stiffness, dt):
    left = mass / dt + gyro / 2
    advance = -np.linalg.solve(left, dt * stiffness - 2 * mass / dt)
    memory = -np.linalg.solve(left, mass / dt - gyro / 2)
    return advance, memory


def doubled_residual(advance, memory, vector):
    following = advance @ vector + memory @ vector
    return following - vector


def main() -> None:
    bare, matrices = action.expected_blocks()
    mass = matrices["mass"]
    gyro = matrices["gyro"]
    stiffness = matrices["stiffness"]
    psi, chi = box.unpack(
        torch.tensor(bare["rest"], dtype=torch.float64),
        5)
    psi = psi.detach().numpy()
    chi = chi.detach().numpy()
    translations = native.translation_vectors(psi, chi)

    zero_psi = np.zeros_like(psi)
    zero_chi = np.zeros_like(chi)
    phase_vectors = {
        "Psi_phase": native.pack(1j * psi, zero_chi),
        "chi_phase": native.pack(zero_psi, 1j * chi),
    }
    phase_vectors = {
        name: value / max(np.linalg.norm(value), 1e-300)
        for name, value in phase_vectors.items()
    }

    dt_rows = []
    for divisor in (1, 2, 4, 8):
        dt = native_time.DT / divisor
        advance, memory = update_blocks(mass, gyro, stiffness, dt)
        axis_rows = []
        for axis in range(3):
            vector = translations[:, axis]
            residual = doubled_residual(advance, memory, vector)
            axis_rows.append({
                "axis": axis,
                "neutral_update_relative_residual": float(
                    np.linalg.norm(residual) / np.linalg.norm(vector)),
                "residual_over_dt_squared": float(
                    np.linalg.norm(residual) / np.linalg.norm(vector) / dt**2),
            })
        controls = {}
        for name, vector in phase_vectors.items():
            residual = doubled_residual(advance, memory, vector)
            controls[name] = float(
                np.linalg.norm(residual) / np.linalg.norm(vector))
        dt_rows.append({
            "dt": dt,
            "axis_rows": axis_rows,
            "phase_symmetry_control_residuals": controls,
        })

    # Continuous zero-frequency inverse condition: (K+dK) t = 0.
    target = np.concatenate([
        -(stiffness @ translations[:, axis]) for axis in range(3)])
    candidates = inverse.action_directions((mass, gyro, stiffness))
    names = list(candidates)
    response = np.column_stack([
        np.concatenate([
            candidates[name][2] @ translations[:, axis]
            for axis in range(3)])
        for name in names
    ])
    coefficients, *_ = np.linalg.lstsq(response, target, rcond=1e-10)
    repaired = target - response @ coefficients
    singular = np.linalg.svd(response, compute_uv=False)

    curvature_rows = []
    for axis in range(3):
        vector = translations[:, axis]
        mass_norm = float(vector @ mass @ vector)
        curvature_rows.append({
            "axis": axis,
            "generalized_translation_curvature": float(
                vector @ stiffness @ vector / mass_norm),
            "stiffness_image_over_mass_norm": float(
                np.linalg.norm(stiffness @ vector) / max(mass_norm, 1e-300)),
        })

    finest = dt_rows[-1]
    maximum_fine = max(row["neutral_update_relative_residual"]
                       for row in finest["axis_rows"])
    result = {
        "status": "V4_DIRAC_LOW_MOMENTUM_TRANSLATION_REQUIREMENT",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "requirement": (
            "the co-rotating positive-energy Dirac rest branch must admit a "
            "neutral native center-displacement tangent before alpha dot p "
            "can generate continuous low-momentum motion"),
        "dt_rows": dt_rows,
        "continuous_curvature_rows": curvature_rows,
        "existing_action_inverse_screen": {
            "candidate_count": len(names),
            "response_rank": int(np.sum(
                singular > max(singular[0], 1.0) * 1e-10)),
            "unrepaired_target_norm": float(np.linalg.norm(target)),
            "best_relative_residual": float(
                np.linalg.norm(repaired) / max(np.linalg.norm(target), 1e-300)),
            "diagnostic_coefficients": {
                name: float(value) for name, value in zip(
                    names, coefficients, strict=True)
            },
            "coefficients_admitted": False,
        },
        "verdict": (
            "CURRENT_ACTION_TRANSLATION_REQUIREMENT_PASS"
            if maximum_fine < 1e-8 else
            "CURRENT_ACTION_HAS_LOCALIZED_PINNING_OBSTRUCTION"),
        "claim_boundary": (
            "Necessary low-momentum tangent condition at the exact localized "
            "rest orbit. Failure does not exclude isolated finite-speed "
            "traveling waves. Fitted inverse coefficients are diagnostic and "
            "are not an admitted action."),
    }
    (HERE / "cube_incidence_dirac_translation_requirement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
