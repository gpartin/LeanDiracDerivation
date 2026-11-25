"""Eight-update delayed-handoff gate for the unit cube-incidence action."""

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
STEPS = 8


def rho(raw):
    psi, _chi = native_box.unpack(torch.tensor(raw, dtype=torch.float64), SIDE)
    return torch.sum(torch.abs(psi)**2, dim=-1).detach().numpy()


def center(raw):
    density = rho(raw)
    grid = np.indices(density.shape)
    total = max(float(np.sum(density)), 1e-30)
    return np.asarray([float(np.sum(grid[a]*density)/total)
                       for a in range(3)])


def mask(axis, sign):
    coordinate = np.indices((SIDE, SIDE, SIDE))[axis]
    return coordinate >= 3 if sign > 0 else coordinate <= 0


def main():
    bare, matrices = action.expected_blocks()
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    factor = lu_factor(update.rest_following_jacobian(rotation, matrices))
    initial_center = center(bare["rest"])
    rows = []
    for axis in range(3):
        for sign in (-1, 1):
            previous = launch.launch_previous(
                torch.tensor(rotation.T@bare["rest"], dtype=torch.float64),
                axis, sign, "common", 0.1).detach().numpy()
            current = bare["rest"].copy()
            samples = []
            converged = True
            target = mask(axis, sign)
            opposite = mask(axis, -sign)
            for step in range(1, STEPS+1):
                following, history, passed = update.solve_following(
                    previous, current, 2*current-previous, factor,
                    tolerance=1e-8, maximum_iterations=20)
                converged = converged and passed
                density = rho(following)
                total = max(float(np.sum(density)), 1e-30)
                displacement = center(following)-initial_center
                samples.append({
                    "step": step,
                    "solver_converged": passed,
                    "solver_iterations": len(history)-1,
                    "target_exterior_fraction": float(
                        np.sum(density[target])/total),
                    "opposite_exterior_fraction": float(
                        np.sum(density[opposite])/total),
                    "directed_exterior_contrast": float(
                        (np.sum(density[target])-np.sum(density[opposite]))
                        / total),
                    "signed_center_shift": float(sign*displacement[axis]),
                    "full_center_shift": displacement.tolist(),
                })
                if not passed:
                    break
                previous, current = current, following
            rows.append({
                "axis": axis,
                "sign": sign,
                "all_updates_converged": converged,
                "samples": samples,
                "maximum_directed_exterior_contrast": max(
                    row["directed_exterior_contrast"] for row in samples),
                "final_signed_center_shift": samples[-1][
                    "signed_center_shift"],
            })
    all_pass = all(row["all_updates_converged"] for row in rows)
    minimum_final_shift = min(row["final_signed_center_shift"] for row in rows)
    minimum_maximum_contrast = min(
        row["maximum_directed_exterior_contrast"] for row in rows)
    result = {
        "status": "V4_CUBE_INCIDENCE_SHORT_HANDOFF",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "launch_mode": "common",
        "momentum": 0.1,
        "updates": STEPS,
        "all_updates_converged": all_pass,
        "minimum_final_signed_center_shift": minimum_final_shift,
        "minimum_maximum_directed_exterior_contrast": minimum_maximum_contrast,
        "rows": rows,
        "verdict": (
            "DELAYED_ALL_AXIS_HANDOFF_RESPONSE_PASS"
            if all_pass and minimum_final_shift > 0
            and minimum_maximum_contrast > 0
            else "DELAYED_HANDOFF_RESPONSE_FAIL"),
        "claim_boundary": (
            "Eight-update unrestricted launch test only.  A response pass "
            "would not establish a complete translation, carrier identity, "
            "Dirac-current matching, or V4-006."),
    }
    (HERE/"cube_incidence_short_handoff.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
