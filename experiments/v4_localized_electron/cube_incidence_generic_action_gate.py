"""Identity and larger-box rest gate for the generic action adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import cube_incidence_generic_action as generic
import cube_incidence_nonlinear_action as side5
import localized_stability as native


HERE = Path(__file__).resolve().parent


def layers(side):
    psi, chi = native.carrier_fields(side, native_time.AMPLITUDE_SQ)
    current = torch.tensor(native_box.pack(psi, chi), dtype=torch.float64,
                           requires_grad=True)
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    phase = np.exp(1j*theta)
    previous = native_box.phase_rotate(
        current.detach(), np.conjugate(phase), side)
    following = native_box.phase_rotate(current.detach(), phase, side)
    return previous, current, following


def main():
    rows = []
    side5_identity = {}
    for side in (5, 6, 7):
        previous, current, following = layers(side)
        residual = generic.del_residual(
            previous, current, following, side).detach().numpy()
        row = {
            "side": side,
            "configuration_dimension": int(current.numel()),
            "rest_DEL_RMS": float(np.linalg.norm(residual)/np.sqrt(residual.size)),
            "rest_DEL_maximum_absolute": float(np.max(abs(residual))),
        }
        if side == 5:
            old_value = side5.interval_action(previous, current)
            new_value = generic.interval_action(previous, current, side)
            old_residual = side5.del_residual(
                previous, current, following).detach().numpy()
            side5_identity = {
                "interval_action_absolute_difference": float(
                    abs(old_value.detach()-new_value.detach())),
                "DEL_relative_difference": float(
                    np.linalg.norm(old_residual-residual)
                    / max(np.linalg.norm(old_residual), 1e-30)),
                "DEL_absolute_difference": float(
                    np.linalg.norm(old_residual-residual)),
            }
        rows.append(row)
    maximum = max(row["rest_DEL_RMS"] for row in rows)
    passed = (side5_identity["interval_action_absolute_difference"] < 1e-12
              and side5_identity["DEL_absolute_difference"] < 1e-12
              and maximum < 1e-9)
    result = {
        "status": "V4_CUBE_INCIDENCE_GENERIC_ACTION_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "side5_identity": side5_identity,
        "rows": rows,
        "verdict": ("GENERIC_ACTION_IDENTITY_AND_REST_PASS"
                    if passed else "GENERIC_ACTION_GATE_FAIL"),
        "claim_boundary": (
            "Exact formula identity on side five and rest-orbit transfer to "
            "larger boxes. It does not prove a moving branch, box-converged "
            "transport, or V4-006."),
    }
    (HERE/"cube_incidence_generic_action_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
