"""Timestep refinement of the native-normalized C27/gyro candidate.

The exact rest-force family leaves two gyro null directions.  The native
inertia normalization and cube incidence select

  c = (a + B/4, a - B/4, -a/2, 0),
  a = 2 dt / sin(theta) -> 2/chi0.

Equivalently, the local chi-velocity term carries B/2 and the symmetric body
common-area term carries the -1/4 incidence completion.  This test determines
whether the tiny residual growth at rate 256 is physical or numerical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
import torch

import c27_body_gyro_basis_probe as candidate
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
SIDE = 3
RATES = (128, 256, 512, 1024)


def configure(dt: float) -> None:
    candidate.base.inverse.SIDE = SIDE
    candidate.base.inverse.DT = dt
    candidate.base.inverse.rest.SIDE = SIDE
    candidate.base.inverse.rest.DT = dt
    candidate.base.inverse.current_audit.SIDE = SIDE
    candidate.base.inverse.current_audit.DT = dt


def classify(rate: int):
    dt = 1.0 / rate
    configure(dt)
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    a = 2 * dt / np.sin(theta)
    coefficients = torch.tensor(
        (a + candidate.base.inverse.rank.B / 4,
         a - candidate.base.inverse.rank.B / 4,
         -a / 2,
         0.0), dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(pm, q, pp, 1.0, coefficients)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    scalar = action(*variables)
    gradient = torch.autograd.grad(
        scalar, variables[1], create_graph=False)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous_block = hessian[1][0].detach().numpy()
    current_block = hessian[1][1].detach().numpy()
    following_block = hessian[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(SIDE, theta, 0.0, 1)
    lhs = following_block @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    threshold = singular[0] * 1e-12
    rank_value = int(np.sum(singular > threshold))
    row = {
        "rate": rate,
        "dt": dt,
        "theta": float(theta),
        "a": float(a),
        "a_continuum_limit": 2 / candidate.base.inverse.rank.R,
        "coefficients": [float(value) for value in coefficients],
        "rest_DEL_rms": float(
            np.linalg.norm(gradient) / np.sqrt(gradient.size)),
        "following_block_rank": rank_value,
        "following_block_dimension": int(lhs.shape[0]),
        "following_block_condition_number": float(
            singular[0] / max(singular[-1], 1e-300)),
    }
    if rank_value < lhs.shape[0]:
        row.update({
            "spectral_radius": None,
            "growth_rate_per_unit_time": None,
            "reciprocal_pairing_defect": None,
        })
        return row
    forward = -np.linalg.solve(lhs, current_block)
    backward = -np.linalg.solve(lhs, previous_block @ rotation.T)
    identity = np.eye(forward.shape[0])
    monodromy = np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])
    eigenvalues = eigvals(monodromy, check_finite=False)
    moduli = np.sort(abs(eigenvalues))
    radius = float(moduli[-1])
    row.update({
        "spectral_radius": radius,
        "growth_rate_per_unit_time": float(np.log(radius) / dt),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
        "largest_moduli": [float(value) for value in moduli[-12:][::-1]],
    })
    return row


def main() -> None:
    candidate.base.inverse.rank.lock_inertia()
    rows = [classify(rate) for rate in RATES]
    finite = all(row["growth_rate_per_unit_time"] is not None for row in rows)
    growth = [row["growth_rate_per_unit_time"] for row in rows]
    pairing = [row["reciprocal_pairing_defect"] for row in rows]
    at_precision_floor = finite and all(
        growth[index] <= 20 * pairing[index] / rows[index]["dt"]
        for index in range(len(rows)))
    result = {
        "status": "V4_C27_BODY_GYRO_NATIVE_TIMESTEP_REFINEMENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "rates": list(RATES),
        "coefficient_rule": (
            "c=(a+B/4,a-B/4,-a/2,0), a=2dt/sin(theta), with "
            "continuum a=2/chi0"),
        "rows": rows,
        "growth_consistent_with_reciprocal_numerical_floor": at_precision_floor,
        "verdict": (
            "C27_BODY_GYRO_NATIVE_REFINEMENT_CONSISTENT_WITH_STABILITY"
            if at_precision_floor else
            "C27_BODY_GYRO_NATIVE_REFINEMENT_NOT_STABLE"),
        "claim_boundary": (
            "Side-3 linear timestep refinement only. Stability still requires "
            "higher-precision/structured spectral confirmation, box refinement, "
            "and nonlinear perturbation evolution."),
    }
    output = HERE / "c27_body_gyro_native_refinement.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rows": [{
            "rate": row["rate"],
            "growth": row["growth_rate_per_unit_time"],
            "pairing_defect": row["reciprocal_pairing_defect"],
            "rest_DEL_rms": row["rest_DEL_rms"],
        } for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
