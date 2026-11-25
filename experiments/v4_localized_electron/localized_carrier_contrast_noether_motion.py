"""Measure native Noether-charge motion on the solved edge launch.

The unrestricted inverse launch makes the linearized DEL equation produce a
literal translated carrier profile.  Shape translation is not yet physical
charge translation.  This audit derives the sitewise discrete phase momentum
from the same interval action and compares its center on the current and next
layers for all six signed directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp

import localized_carrier_contrast_full_inverse_launch as full
import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent
STEP = 1e-5


right_momentum = grad(inverse.interval_action, argnums=1)


def phase_charge_density_tensor(left_raw: torch.Tensor,
                                right_raw: torch.Tensor,
                                mode: str = "common") -> torch.Tensor:
    gradient = right_momentum(left_raw, right_raw)
    psi, chi = inverse.current_audit.mold.unpack_tensor(
        right_raw, inverse.SIDE)
    zero_psi = torch.zeros_like(psi)
    zero_chi = torch.zeros_like(chi)
    generator = inverse.current_audit.pack_tensor(
        1j*psi if mode in ("Psi", "common") else zero_psi,
        1j*chi if mode in ("chi", "common") else zero_chi)
    site = (gradient*generator).reshape(
        inverse.SIDE, inverse.SIDE, inverse.SIDE, 4, 3).sum(dim=(-1, -2))
    return site


def phase_charge_density(left: np.ndarray, right: np.ndarray,
                         mode: str = "common") -> np.ndarray:
    left_raw = torch.tensor(left, dtype=torch.float64)
    right_raw = torch.tensor(right, dtype=torch.float64)
    return phase_charge_density_tensor(left_raw, right_raw, mode).detach().numpy()


def phase_charge_density_derivative(
        left: np.ndarray, right: np.ndarray,
        left_direction: np.ndarray | None,
        right_direction: np.ndarray | None,
        mode: str = "common") -> np.ndarray:
    left_raw = torch.tensor(left, dtype=torch.float64)
    right_raw = torch.tensor(right, dtype=torch.float64)
    dleft = torch.zeros_like(left_raw) if left_direction is None else torch.tensor(
        left_direction, dtype=torch.float64)
    dright = torch.zeros_like(right_raw) if right_direction is None else torch.tensor(
        right_direction, dtype=torch.float64)
    function = lambda l, r: phase_charge_density_tensor(l, r, mode)
    derivative_value = jvp(
        function, (left_raw, right_raw), (dleft, dright))[1]
    return derivative_value.detach().numpy()


def center(density: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    coordinates = np.indices(density.shape)
    return np.array([float(np.sum(coordinates[axis]*density)/total)
                     for axis in range(3)])


def center_derivative_exact(density: np.ndarray,
                            density_derivative: np.ndarray) -> np.ndarray:
    total = float(np.sum(density))
    dtotal = float(np.sum(density_derivative))
    coordinates = np.indices(density.shape)
    values = []
    for axis in range(3):
        moment = float(np.sum(coordinates[axis]*density))
        dmoment = float(np.sum(coordinates[axis]*density_derivative))
        values.append((dmoment*total-moment*dtotal)/(total**2))
    return np.asarray(values)


def derivative(function, step: float = STEP) -> np.ndarray:
    return (function(step)-function(-step))/(2*step)


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    data = np.load(
        HERE / "localized_carrier_contrast_full_inverse_launch.npz")
    previous = data["previous"]
    current = data["current"]
    following = data["following"]
    targets = data["targets"]
    launches = data["launches"]
    baseline_current_density = phase_charge_density(previous, current)
    baseline_following_density = phase_charge_density(current, following)
    baseline_total = float(np.sum(baseline_current_density))
    labels = [(axis, sign) for axis in range(3) for sign in (-1, 1)]
    rows = []
    for column, (axis, sign) in enumerate(labels):
        launch = launches[:, column]
        target = targets[:, column]
        current_density_derivative = derivative(
            lambda epsilon: phase_charge_density(
                previous+epsilon*launch, current))
        following_density_derivative = derivative(
            lambda epsilon: phase_charge_density(
                current, following+epsilon*target))
        current_center_derivative = derivative(
            lambda epsilon: center(phase_charge_density(
                previous+epsilon*launch, current)))
        following_center_derivative = derivative(
            lambda epsilon: center(phase_charge_density(
                current, following+epsilon*target)))
        displacement = following_center_derivative-current_center_derivative
        total_current = float(np.sum(current_density_derivative))
        total_following = float(np.sum(following_density_derivative))
        conservation_scale = max(
            np.linalg.norm(current_density_derivative),
            np.linalg.norm(following_density_derivative), 1e-30)
        rows.append({
            "axis": axis,
            "sign": sign,
            "current_charge_derivative": total_current,
            "following_charge_derivative": total_following,
            "global_charge_derivative_defect": float(
                abs(total_following-total_current)
                / max(abs(total_current), abs(total_following), 1e-30)),
            "local_charge_density_continuity_relative_defect": float(
                np.linalg.norm(
                    following_density_derivative-current_density_derivative)
                / conservation_scale),
            "current_center_derivative": [float(value)
                                           for value in current_center_derivative],
            "following_center_derivative": [float(value)
                                             for value in following_center_derivative],
            "one_interval_center_displacement_response": [
                float(value) for value in displacement],
            "signed_requested_axis_response": float(sign*displacement[axis]),
            "moves_requested_way": bool(sign*displacement[axis] > 0),
            "transverse_center_response_norm": float(np.linalg.norm(
                np.delete(displacement, axis))),
        })
    max_global = max(row["global_charge_derivative_defect"] for row in rows)
    all_move = all(row["moves_requested_way"] for row in rows)
    result = {
        "status": "V4_CARRIER_CONTRAST_NOETHER_MOTION_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "same complete discrete interval action as the unrestricted "
            "inverse-launch audit"),
        "phase_symmetry": "common native Psi--chi U(1)",
        "baseline_total_phase_charge": baseline_total,
        "baseline_current_center": [float(value) for value in center(
            baseline_current_density)],
        "baseline_following_center": [float(value) for value in center(
            baseline_following_density)],
        "rows": rows,
        "max_global_charge_derivative_defect": max_global,
        "all_six_charge_centers_move_requested_way": all_move,
        "verdict": (
            "LINEAR_NATIVE_EDGE_LAUNCH_MOVES_NOETHER_CHARGE"
            if max_global < 1e-8 and all_move else
            "SHAPE_SNAP_AND_NOETHER_CHARGE_MOTION_DISAGREE"),
        "claim_boundary": (
            "This is a first-order discrete Noether-density result on one "
            "solved launch. It is not a nonlinear repeated hop or a Dirac "
            "current match."),
        "corrective_action": (
            "On pass, seed a nonlinear two-handoff relative-periodic solve. "
            "On failure, decompose the target into internal phase/copy "
            "directions and solve the native positive-frequency graph before "
            "changing the action."),
    }
    (HERE / "localized_carrier_contrast_noether_motion.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
