"""Diagnose the 5e-8 unit-circle splitting of the native C27/gyro action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    a = 2 * spectrum.DT / np.sin(theta)
    coefficients = torch.tensor((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ), dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(pm, q, pp, 1.0, coefficients)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    hessian = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous_block = hessian[1][0].detach().numpy()
    current_block = hessian[1][1].detach().numpy()
    following_block = hessian[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    lhs = following_block @ rotation
    forward = -np.linalg.solve(lhs, current_block)
    backward = -np.linalg.solve(lhs, previous_block @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])
    values, left, right = eig(
        monodromy, left=True, right=True, check_finite=False)
    moduli = abs(values)
    order = np.argsort(moduli)[::-1]
    norm_matrix = np.linalg.norm(monodromy, 2)
    rows = []
    for index in order[:24]:
        rv = right[:, index]
        lv = left[:, index]
        rv = rv / np.linalg.norm(rv)
        lv = lv / np.linalg.norm(lv)
        overlap = abs(np.vdot(lv, rv))
        residual = np.linalg.norm(monodromy @ rv - values[index] * rv)
        rows.append({
            "real": float(values[index].real),
            "imag": float(values[index].imag),
            "modulus": float(moduli[index]),
            "modulus_excess": float(moduli[index] - 1),
            "phase": float(np.angle(values[index])),
            "right_residual": float(residual),
            "relative_right_residual": float(
                residual / max(norm_matrix, 1e-300)),
            "left_right_overlap": float(overlap),
            "eigenvalue_condition_estimate": float(
                1 / max(overlap, 1e-300)),
        })
    result = {
        "status": "V4_C27_BODY_GYRO_SPECTRAL_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "matrix_two_norm": float(norm_matrix),
        "maximum_modulus": float(np.max(moduli)),
        "count_modulus_above_1_plus_1e_minus_6": int(np.sum(
            moduli > 1 + 1e-6)),
        "count_modulus_above_1_plus_1e_minus_7": int(np.sum(
            moduli > 1 + 1e-7)),
        "count_modulus_above_1_plus_1e_minus_8": int(np.sum(
            moduli > 1 + 1e-8)),
        "leading_modes": rows,
        "interpretation_rule": (
            "A radius excess comparable to machine precision times the "
            "left/right eigenvalue condition is numerical splitting, not a "
            "resolved finite Floquet exponent."),
        "claim_boundary": (
            "Double-precision conditioning diagnostic; not a formal or "
            "arbitrary-precision proof of semisimple unit-circle spectrum."),
    }
    output = HERE / "c27_body_gyro_spectral_diagnostic.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "maximum_modulus": result["maximum_modulus"],
        "counts": [
            result["count_modulus_above_1_plus_1e_minus_6"],
            result["count_modulus_above_1_plus_1e_minus_7"],
            result["count_modulus_above_1_plus_1e_minus_8"],
        ],
        "leading_condition": rows[0]["eigenvalue_condition_estimate"],
        "leading_residual": rows[0]["right_residual"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
