"""Rest and retained-carrier positivity gate for the live-beta metric."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as native_action
import density_mold_connection_probe as local
import live_relative_grade_action as grade
import live_relative_grade_metric_action as candidate


HERE = Path(__file__).resolve().parent


def main() -> None:
    rest = native_action.expected_blocks()[0]["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    residual = candidate.del_residual(*layers, dt).detach().numpy()
    psi, chi = local.unpack_tensor(layers[1], 5)
    live_grade = grade.live_grade(psi, chi).detach().numpy()
    occupied = torch.real(torch.sum(torch.conj(psi)*psi, dim=-1)).numpy() > 1e-12
    metric_values = 1+candidate.METRIC_SCALE*live_grade[occupied]
    result = {
        "status": "V4_LIVE_RELATIVE_GRADE_METRIC_REST_GATE",
        "metric_scale": candidate.METRIC_SCALE,
        "coefficient_provenance": "1/chi0 = 1/19 C19 mode normalization",
        "rest_DEL_RMS": float(np.linalg.norm(residual)/np.sqrt(residual.size)),
        "retained_metric_eigenvalue_range": [
            float(np.min(metric_values)), float(np.max(metric_values))],
        "verdict": (
            "LIVE_RELATIVE_GRADE_METRIC_REST_POSITIVE"
            if np.linalg.norm(residual)/np.sqrt(residual.size) < 1e-9
            and np.min(metric_values) > 0 else
            "LIVE_RELATIVE_GRADE_METRIC_REJECTED"),
        "claim_boundary": "Exact retained rest and local carrier metric only.",
    }
    (HERE/"live_relative_grade_metric_rest_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
