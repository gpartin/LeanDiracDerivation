"""Transport-rank gate for the native-normalized C27/gyro candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import c27_body_gyro_basis_probe as candidate


HERE = Path(__file__).resolve().parent


def main() -> None:
    candidate.base.inverse.rank.lock_inertia()
    candidate.base.inverse.rest.SIDE = candidate.base.inverse.SIDE
    candidate.base.inverse.rest.DT = candidate.base.inverse.DT
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))
    dt = candidate.base.inverse.DT
    a = 2 * dt / np.sin(theta)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    del_residual, right_momentum = candidate.make_derivatives(
        1.0, coefficients)
    rest_residual = del_residual(*tensors).detach().numpy()
    candidate.base.del_residual = del_residual
    candidate.base.right_momentum = right_momentum
    dimension = previous.size
    previous_block = candidate.base.batched_jvp(
        "previous", tensors, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = candidate.base.component.component_targets(
            axis, np.exp(1j * theta))
        images = candidate.base.batched_jvp("following", tensors, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            candidate.base.response(
                previous, current, following,
                launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_center = responses[2:] @ null
        left, singular, _ = np.linalg.svd(fixed_center, full_matrices=True)
        threshold = 1e-8 * max(singular[0], 1e-30)
        rows.append({
            "phase_ramp_axis": axis,
            "component_labels": labels,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [
                float(value) for value in singular],
            "relative_third_singular_value": float(
                singular[2] / max(singular[0], 1e-30)),
            "missing_physical_direction": [
                float(value) for value in left[:, -1]],
        })
    robust = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    result = {
        "status": "V4_C27_BODY_GYRO_NATIVE_TRANSPORT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": candidate.base.inverse.SIDE,
        "dt": dt,
        "a": float(a),
        "coefficients": {
            name: float(value)
            for name, value in zip(candidate.BASIS_NAMES, coefficients)
        },
        "coefficient_rule": (
            "c=(a+B/4,a-B/4,-a/2,0), a=2dt/sin(theta); selected by "
            "native inertia and body-incidence normalization"),
        "rest_DEL_rms": float(
            np.linalg.norm(rest_residual) / np.sqrt(rest_residual.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": robust,
        "verdict": (
            "C27_BODY_GYRO_NATIVE_TRANSPORT_RANK_PASS" if robust else
            "C27_BODY_GYRO_NATIVE_TRANSPORT_RANK_FAIL"),
        "claim_boundary": (
            "Exact rest and linear fixed-charge response only. This is not "
            "nonlinear translation or Dirac-current matching."),
    }
    output = HERE / "c27_body_gyro_native_transport.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rest_DEL_rms": result["rest_DEL_rms"],
        "ranks": [row["fixed_charge_center_rank"] for row in rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
