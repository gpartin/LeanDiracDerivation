"""Scan the positive strength ray of the rest-preserving C27/gyro pair.

Both the C27 potential and its derived gyro completion scale by one strength.
The discrete action Hessian is therefore affine in that strength.  We compute
the baseline and unit Hessians once, then classify the complete positive ray
without repeatedly differentiating or fitting a stability target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
STRENGTHS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
             1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


def blocks(strength: float, layers, theta):
    coefficient = 2 * spectrum.DT / np.sin(theta)
    coefficients = torch.tensor(
        strength * np.asarray(
            (coefficient, coefficient, -coefficient / 4, -coefficient / 4)),
        dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(
            pm, q, pp, strength, coefficients)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    hessian = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    return tuple(hessian[1][index].detach().numpy() for index in range(3))


def classify(strength, baseline, delta, rotation):
    previous, current, following = tuple(
        baseline[index] + strength * delta[index] for index in range(3))
    lhs = following @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    threshold = singular[0] * 1e-12
    rank = int(np.sum(singular > threshold))
    row = {
        "strength": float(strength),
        "following_block_rank": rank,
        "following_block_dimension": int(lhs.shape[0]),
        "following_block_condition_number": float(
            singular[0] / max(singular[-1], 1e-300)),
    }
    if rank < lhs.shape[0]:
        row.update({
            "spectral_radius": None,
            "growth_rate_per_unit_time": None,
            "reciprocal_pairing_defect": None,
        })
        return row
    forward = -np.linalg.solve(lhs, current)
    backward = -np.linalg.solve(lhs, previous @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])
    values = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(values))
    radius = float(moduli[-1])
    row.update({
        "spectral_radius": radius,
        "growth_rate_per_unit_time": float(np.log(radius) / spectrum.DT),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
    })
    return row


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    zero = blocks(0.0, layers, theta)
    one = blocks(1.0, layers, theta)
    delta = tuple(one[index] - zero[index] for index in range(3))
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    rows = [classify(value, zero, delta, rotation) for value in STRENGTHS]
    finite = [row for row in rows
              if row["growth_rate_per_unit_time"] is not None]
    best = min(finite, key=lambda row: row["growth_rate_per_unit_time"])
    baseline_growth = rows[0]["growth_rate_per_unit_time"]
    improved = best["growth_rate_per_unit_time"] < baseline_growth - 1e-8
    result = {
        "status": "V4_C27_BODY_GYRO_STRENGTH_RAY_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "strengths": list(STRENGTHS),
        "rows": rows,
        "best_row": best,
        "positive_ray_improves_growth": improved,
        "verdict": (
            "C27_BODY_GYRO_STRENGTH_RAY_IMPROVES_STABILITY" if improved
            else "C27_BODY_GYRO_STRENGTH_RAY_NO_STABILITY_IMPROVEMENT"),
        "claim_boundary": (
            "Complete side-3 linear spectrum along one exact action ray. "
            "It does not test other gyro nullspace directions, nonlinear "
            "stability, or carrier movement."),
    }
    output = HERE / "c27_body_gyro_strength_scan.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "baseline_growth": baseline_growth,
        "best_row": best,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
