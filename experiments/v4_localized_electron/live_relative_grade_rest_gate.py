"""Exact retained-rest test for the live relative-grade action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_nonlinear_action as native_action
import live_relative_grade_action as candidate


HERE = Path(__file__).resolve().parent


def main() -> None:
    rest = native_action.expected_blocks()[0]["rest"]
    dt = native_time.DT
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (rotation.T@rest, rest, rotation@rest))
    residual = candidate.del_residual(*layers, dt).detach().numpy()
    result = {
        "status": "V4_LIVE_RELATIVE_GRADE_REST_GATE",
        "grade_scale": candidate.GRADE_SCALE,
        "rest_DEL_norm": float(np.linalg.norm(residual)),
        "rest_DEL_RMS": float(np.linalg.norm(residual)/np.sqrt(residual.size)),
        "verdict": (
            "LIVE_RELATIVE_GRADE_REST_NULL"
            if np.linalg.norm(residual)/np.sqrt(residual.size) < 1e-9 else
            "LIVE_RELATIVE_GRADE_DEFORMS_REST"),
        "claim_boundary": "Retained three-layer rest orbit only.",
    }
    (HERE/"live_relative_grade_rest_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
