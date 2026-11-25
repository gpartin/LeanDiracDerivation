"""Full two-layer rest-spectrum gate for the reduced C27/gyro action.

The preceding response screen found that four apparent gyro coefficients
collapse to

  a (local_common + local_relative)
    - a/4 (body_common + body_relative),

with a = 2 dt / sin(theta) for the exact discrete rest clock.  In the
continuum this tends to 2/chi0.  This file tests the complete implicit
variational recurrence on every Psi/chi/halo/exterior coordinate of a side-3
periodic box.  It is a spectrum gate, not an admission or movement pass.
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
DT = 1.0 / 256.0


def configure() -> None:
    candidate.base.inverse.SIDE = SIDE
    candidate.base.inverse.DT = DT
    candidate.base.inverse.rest.SIDE = SIDE
    candidate.base.inverse.rest.DT = DT
    candidate.base.inverse.current_audit.SIDE = SIDE
    candidate.base.inverse.current_audit.DT = DT


def classify(use_candidate: bool):
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    if use_candidate:
        coefficient = 2 * DT / np.sin(theta)
        coefficients = np.asarray(
            (coefficient, coefficient, -coefficient / 4, -coefficient / 4),
            dtype=np.float64)
        potential = 1.0
    else:
        coefficient = 0.0
        coefficients = np.zeros(4, dtype=np.float64)
        potential = 0.0
    values = torch.tensor(coefficients, dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(
            pm, q, pp, potential, values)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    scalar = action(*variables)
    gradient = torch.autograd.grad(
        scalar, variables[1], create_graph=False)[0].detach().numpy()
    blocks = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous_block = blocks[1][0].detach().numpy()
    current_block = blocks[1][1].detach().numpy()
    following_block = blocks[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(SIDE, theta, 0.0, 1)
    lhs = following_block @ rotation
    singular = np.linalg.svd(lhs, compute_uv=False)
    threshold = singular[0] * 1e-12
    rank = int(np.sum(singular > threshold))
    row = {
        "candidate": use_candidate,
        "continuum_gyro_coefficient": 2 / candidate.base.inverse.rank.R,
        "discrete_gyro_coefficient": float(coefficient),
        "body_to_local_ratio": -0.25 if use_candidate else 0.0,
        "rest_DEL_rms": float(
            np.linalg.norm(gradient) / np.sqrt(gradient.size)),
        "following_block_dimension": int(lhs.shape[0]),
        "following_block_rank": rank,
        "following_block_condition_number": float(
            singular[0] / max(singular[-1], 1e-300)),
        "following_block_smallest_singular_value": float(singular[-1]),
    }
    if rank < lhs.shape[0]:
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
        "growth_rate_per_unit_time": float(np.log(radius) / DT),
        "reciprocal_pairing_defect": float(np.max(
            abs(moduli * moduli[::-1] - 1))),
        "largest_moduli": [float(value) for value in moduli[-12:][::-1]],
    })
    return row


def main() -> None:
    configure()
    candidate.base.inverse.rank.lock_inertia()
    baseline = classify(False)
    trial = classify(True)
    well_posed = trial["growth_rate_per_unit_time"] is not None
    stable = (well_posed
              and trial["growth_rate_per_unit_time"] < 1e-8)
    improved = (well_posed
                and baseline["growth_rate_per_unit_time"] is not None
                and trial["growth_rate_per_unit_time"]
                < baseline["growth_rate_per_unit_time"] - 1e-8)
    result = {
        "status": "V4_C27_BODY_GYRO_FULL_REST_SPECTRUM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": SIDE,
        "dt": DT,
        "action_delta": (
            "unit positive joint-W body-diagonal Dirichlet potential plus "
            "the reduced local/body common+relative area gyro one-form"),
        "coefficient_provenance": (
            "a=2 dt/sin(theta), body/local=-1/4 from exact native discrete "
            "rest-torque cancellation; a tends to 2/chi0"),
        "baseline": baseline,
        "candidate": trial,
        "candidate_update_well_posed": well_posed,
        "candidate_linearly_stable": stable,
        "candidate_improves_growth": improved,
        "verdict": (
            "C27_BODY_GYRO_REST_SPECTRUM_PASS" if stable else
            "C27_BODY_GYRO_UPDATE_SINGULAR" if not well_posed else
            "C27_BODY_GYRO_REST_SPECTRUM_FAIL"),
        "claim_boundary": (
            "Complete side-3 two-layer linear spectrum. Passing would still "
            "require timestep/box refinement, nonlinear 100-cycle stability, "
            "movement, and same-readout Dirac gates."),
    }
    output = HERE / "c27_body_gyro_rest_spectrum.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "baseline_growth": baseline["growth_rate_per_unit_time"],
        "candidate_growth": trial["growth_rate_per_unit_time"],
        "candidate_rest_DEL_rms": trial["rest_DEL_rms"],
        "candidate_update_rank": trial["following_block_rank"],
        "candidate_update_dimension": trial["following_block_dimension"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
