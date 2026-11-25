"""Measure the first unrestricted receiver response of the unit action.

The quiet receiver has zero instantaneous bond current.  This gate therefore
solves the next discrete Euler--Lagrange layer and asks whether a native phase
launch creates a *directed* exterior response.  No receiver field is preloaded
and no state is projected after the solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_implicit_update as update
import cube_incidence_nonlinear_action as action
import cube_incidence_noether_launch as launch


HERE = Path(__file__).resolve().parent
SIDE = 5


def density(raw: np.ndarray) -> np.ndarray:
    psi, _chi = native_box.unpack(torch.tensor(raw, dtype=torch.float64), SIDE)
    return torch.sum(torch.abs(psi)**2, dim=-1).detach().numpy()


def exterior_mask(axis: int, sign: int) -> np.ndarray:
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    return coordinate >= 3 if sign > 0 else coordinate <= 0


def center_of_density(raw: np.ndarray) -> np.ndarray:
    rho = density(raw)
    grid = np.indices(rho.shape)
    total = max(float(np.sum(rho)), 1e-30)
    return np.asarray([np.sum(grid[axis]*rho)/total for axis in range(3)])


def main() -> None:
    bare, matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    factor = lu_factor(update.rest_following_jacobian(rotation, matrices))
    previous0 = rotation.T@bare["rest"]
    current = bare["rest"]
    expected = rotation@current
    momentum = 0.1
    initial_center = center_of_density(current)
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            target = exterior_mask(axis, sign)
            opposite = exterior_mask(axis, -sign)
            for mode in launch.MODES:
                previous = launch.launch_previous(
                    torch.tensor(previous0, dtype=torch.float64), axis, sign,
                    mode, momentum).detach().numpy()
                following, history, converged = update.solve_following(
                    previous, current, 2*current-previous, factor,
                    tolerance=1e-8, maximum_iterations=20)
                rho = density(following)
                target_power = float(np.sum(rho[target]))
                opposite_power = float(np.sum(rho[opposite]))
                total_power = max(float(np.sum(rho)), 1e-30)
                shift = center_of_density(following)-initial_center
                common_flux = launch.flux(
                    torch.tensor(current, dtype=torch.float64),
                    torch.tensor(following, dtype=torch.float64), target,
                    "common")
                rows.append({
                    "axis": axis,
                    "sign": sign,
                    "launch_mode": mode,
                    "solver_converged": converged,
                    "solver_iterations": len(history)-1,
                    "solver_residual_RMS": history[-1],
                    "target_exterior_Psi_fraction": target_power/total_power,
                    "opposite_exterior_Psi_fraction": opposite_power/total_power,
                    "directed_exterior_contrast": (
                        (target_power-opposite_power)/total_power),
                    "signed_center_shift": float(sign*shift[axis]),
                    "signed_common_flux": float(sign*common_flux),
                    "full_center_shift": shift.tolist(),
                    "corotating_relative_defect": float(
                        np.linalg.norm(following-expected)
                        / np.linalg.norm(expected)),
                })

    per_mode = {}
    for mode in launch.MODES:
        selected = [row for row in rows if row["launch_mode"] == mode]
        per_mode[mode] = {
            "all_updates_converged": all(
                row["solver_converged"] for row in selected),
            "minimum_directed_exterior_contrast": min(
                row["directed_exterior_contrast"] for row in selected),
            "maximum_directed_exterior_contrast": max(
                row["directed_exterior_contrast"] for row in selected),
            "minimum_signed_center_shift": min(
                row["signed_center_shift"] for row in selected),
            "maximum_signed_center_shift": max(
                row["signed_center_shift"] for row in selected),
            "minimum_signed_common_flux": min(
                row["signed_common_flux"] for row in selected),
            "maximum_signed_common_flux": max(
                row["signed_common_flux"] for row in selected),
        }
    candidates = [
        mode for mode, row in per_mode.items()
        if row["all_updates_converged"]
        and row["minimum_directed_exterior_contrast"] > 0
        and row["minimum_signed_center_shift"] > 0
        and row["minimum_signed_common_flux"] > 0
    ]
    selected_mode = max(
        candidates,
        key=lambda mode: per_mode[mode]["minimum_signed_center_shift"]
    ) if candidates else None
    result = {
        "status": "V4_CUBE_INCIDENCE_FIRST_RECEIVER_RESPONSE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "momentum": momentum,
        "initial_instantaneous_exterior_current": "zero within numerical precision",
        "per_launch_mode": per_mode,
        "selected_native_launch_mode": selected_mode,
        "rows": rows,
        "verdict": (
            "DIRECTED_FIRST_RECEIVER_RESPONSE_PASS"
            if selected_mode is not None
            else "DIRECTED_FIRST_RECEIVER_RESPONSE_FAIL"),
        "claim_boundary": (
            "One freely solved next layer from a quiet receiver.  It does not "
            "prove a complete handoff, persistent translation, same-state "
            "Dirac current, or V4-006."),
    }
    (HERE/"cube_incidence_launch_response.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
