"""Search the exact rest-force nullspace of the C27 gyro completion.

Rest cancellation fixes only the sums of the common and relative local/body
coefficients.  Their differences are two independent gyroscopic channels:

  n_local = (1, -1, 0, 0),
  n_body  = (0, 0, 1, -1).

They exert zero force on the exact rest orbit, so varying them does not fit or
hold the carrier.  They do change the full phase-space Hessian.  This probe
tests whether stability/regularity can select either native null direction.
No Dirac observable enters the scan.
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
LOCAL_MULTIPLIERS = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
                     10000.0, 30000.0, 100000.0)
BODY_MULTIPLIERS = (-10.0, -3.0, -1.0, -0.3, -0.1, 0.0,
                    0.1, 0.3, 1.0, 3.0, 10.0)


def action_blocks(layers, coefficients, potential=1.0):
    values = torch.tensor(coefficients, dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(
            pm, q, pp, potential, values)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    hessian = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    return tuple(hessian[1][index].detach().numpy() for index in range(3))


def classify(local_multiplier, body_multiplier, center, local_delta,
             body_delta, rotation):
    blocks = tuple(
        center[index]
        + local_multiplier * local_delta[index]
        + body_multiplier * body_delta[index]
        for index in range(3))
    previous, current, following = blocks
    lhs = following @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    threshold = singular[0] * 1e-12
    rank = int(np.sum(singular > threshold))
    row = {
        "local_null_multiplier": float(local_multiplier),
        "body_null_multiplier": float(body_multiplier),
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
    a = 2 * spectrum.DT / np.sin(theta)
    center_coefficients = np.asarray((a, a, -a / 4, -a / 4))
    local_null = np.asarray((a, -a, 0.0, 0.0))
    body_null = np.asarray((0.0, 0.0, a, -a))
    center = action_blocks(layers, center_coefficients)
    local_unit = action_blocks(layers, center_coefficients + local_null)
    body_unit = action_blocks(layers, center_coefficients + body_null)
    local_delta = tuple(local_unit[index] - center[index]
                        for index in range(3))
    body_delta = tuple(body_unit[index] - center[index]
                       for index in range(3))
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    rows = [
        classify(local_value, body_value, center, local_delta, body_delta,
                 rotation)
        for local_value in LOCAL_MULTIPLIERS
        for body_value in BODY_MULTIPLIERS
    ]
    # A distinguished native normalization makes the coefficient of the
    # reduced local chi-velocity term exactly B/2; -1/4 is the body/local
    # incidence ratio already selected by rest cancellation.
    native_local_multiplier = (
        candidate.base.inverse.rank.B / (4 * a))
    native_row = classify(
        native_local_multiplier, -0.25,
        center, local_delta, body_delta, rotation)
    rows.append(native_row)
    finite = [row for row in rows
              if row["growth_rate_per_unit_time"] is not None]
    best = min(finite, key=lambda row: row["growth_rate_per_unit_time"])
    center_row = next(row for row in rows
                      if row["local_null_multiplier"] == 0.0
                      and row["body_null_multiplier"] == 0.0)
    improved = (best["growth_rate_per_unit_time"]
                < center_row["growth_rate_per_unit_time"] - 1e-8)
    stable = best["growth_rate_per_unit_time"] < 1e-8
    result = {
        "status": "V4_C27_BODY_GYRO_NULLSPACE_SCAN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "base_coefficient_a": float(a),
        "nullspace_basis": {
            "local": "a*(1,-1,0,0)",
            "body": "a*(0,0,1,-1)",
        },
        "local_multipliers": list(LOCAL_MULTIPLIERS),
        "body_multipliers": list(BODY_MULTIPLIERS),
        "rows": rows,
        "center_row": center_row,
        "best_row": best,
        "native_inertia_row": native_row,
        "native_inertia_rule": (
            "local null multiplier B/(4a), so the reduced chi-velocity "
            "coefficient is B/2; body null multiplier -1/4"),
        "nullspace_improves_growth": improved,
        "nullspace_finds_linear_stability": stable,
        "verdict": (
            "C27_GYRO_NULLSPACE_STABILIZES_REST" if stable else
            "C27_GYRO_NULLSPACE_IMPROVES_REST" if improved else
            "C27_GYRO_NULLSPACE_NO_STABILITY_IMPROVEMENT"),
        "claim_boundary": (
            "Coarse complete side-3 linear-spectrum scan of the exact "
            "rest-force nullspace. A minimum requires refinement, response "
            "rank recheck, coefficient provenance, and nonlinear gates."),
    }
    output = HERE / "c27_body_gyro_nullspace_scan.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "center_growth": center_row["growth_rate_per_unit_time"],
        "best_row": best,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
