"""Check the finite-amplitude scaling of the linked clock tangent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, theta0 = candidate.base.inverse.rest_layers()
    q0 = np.asarray(current, dtype=np.float64)
    data = np.load(HERE / "c27_body_gyro_charge_tangent_solvability.npz")
    tangent = data["linked_clock_tangent"]
    direction = data["linked_clock_direction"]
    ratio = float(direction[1] / direction[0])
    a = 2 * spectrum.DT / np.sin(theta0)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    residual, _ = candidate.make_derivatives(1.0, coefficients)

    def rms(q: np.ndarray, delta: float) -> float:
        minus = torch.tensor(relative.raw_rotation_matrix(
            spectrum.SIDE, theta0 + delta, ratio * delta, -1),
            dtype=torch.float64)
        plus = torch.tensor(relative.raw_rotation_matrix(
            spectrum.SIDE, theta0 + delta, ratio * delta, 1),
            dtype=torch.float64)
        qt = torch.tensor(q, dtype=torch.float64)
        value = residual(minus @ qt, qt, plus @ qt).detach().numpy()
        return float(np.linalg.norm(value) / np.sqrt(value.size))

    baseline = rms(q0, 0.0)
    rows = []
    for sign in (-1.0, 1.0):
        for fraction in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
            delta = sign * fraction * theta0
            predicted = q0 + delta * tangent
            value = rms(predicted, delta)
            rows.append({
                "sign": sign,
                "fraction": fraction,
                "delta_theta_psi": delta,
                "predicted_DEL_rms": value,
                "excess_over_baseline": value - baseline,
                "residual_over_delta_squared": (
                    (value - baseline) / max(delta * delta, 1e-300)),
                "relative_state_change": float(
                    np.linalg.norm(predicted - q0) / np.linalg.norm(q0)),
            })
    result = {
        "status": "V4_C27_BODY_GYRO_LINKED_TANGENT_SCALING",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "baseline_DEL_rms": baseline,
        "linked_clock_ratio_dthetaChi_over_dthetaPsi": ratio,
        "linked_tangent_norm": float(np.linalg.norm(tangent)),
        "rows": rows,
        "verdict": "C27_BODY_GYRO_LINKED_TANGENT_SCALING_MEASURED",
        "claim_boundary": (
            "Local consistency diagnostic only; it is not finite continuation, "
            "stability, translation, or Dirac-current closure."),
    }
    output = HERE / "c27_body_gyro_linked_tangent_scaling.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
