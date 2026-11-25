"""Solve the unrestricted native linear launch for all six edge snaps.

This is the control following the two-copy and first-shell inverse solves.  It
factorizes the complete previous-layer Euler--Lagrange block once, solves all
six literal next-edge targets, and measures how much launch power lies in the
carrier contrast, the complete first shell, and the remaining periodic box.
No action coefficient or Dirac object is adjusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import localized_carrier_contrast_inverse_launch as inverse


HERE = Path(__file__).resolve().parent


def site_power(raw: np.ndarray) -> np.ndarray:
    psi, chi = inverse.rank.native.unpack(raw, inverse.SIDE)
    return np.sum(abs(psi)**2+abs(chi)**2, axis=-1)


def support_fractions(raw: np.ndarray) -> dict:
    power = site_power(raw)
    total = max(float(np.sum(power)), 1e-30)
    core = np.zeros(power.shape, dtype=bool)
    core[1:3, 1:3, 1:3] = True
    first = np.zeros(power.shape, dtype=bool)
    first[0:4, 0:4, 0:4] = True
    return {
        "eight_site_core": float(np.sum(power[core])/total),
        "first_shell_including_core": float(np.sum(power[first])/total),
        "outside_first_shell": float(np.sum(power[~first])/total),
    }


def main() -> None:
    inverse.rank.lock_inertia()
    inverse.rest.SIDE = inverse.SIDE
    inverse.rest.DT = inverse.DT
    inverse.current_audit.SIDE = inverse.SIDE
    inverse.current_audit.DT = inverse.DT
    previous, current, following, theta = inverse.rest_layers()
    base = tuple(inverse.torch.tensor(value, dtype=inverse.torch.float64)
                 for value in (previous, current, following))
    dimension = previous.size
    identity = np.eye(dimension)
    previous_block = inverse.batched_jvp("previous", base, identity)
    factor = lu_factor(previous_block)

    q_copies = inverse.envelope(
        np.exp(-1j*theta), "three_by_three_by_three")
    q_shell = inverse.native_first_shell_envelope()
    targets = []
    labels = []
    for axis in range(3):
        for sign in (-1, 1):
            target = inverse.shifted_difference(following, axis, sign)
            targets.append(target)
            labels.append((axis, sign))
    target_matrix = np.column_stack(targets)
    target_images = inverse.batched_jvp("following", base, target_matrix)
    launches = lu_solve(factor, -target_images)
    rows = []
    for column, (axis, sign) in enumerate(labels):
        launch = launches[:, column]
        target = targets[column]
        residual = previous_block@launch+target_images[:, column]
        launch_norm = max(float(np.linalg.norm(launch)), 1e-30)
        copy_defect = float(np.linalg.norm(
            launch-q_copies@(q_copies.T@launch))/launch_norm)
        shell_defect = float(np.linalg.norm(
            launch-q_shell@(q_shell.T@launch))/launch_norm)
        receiver = inverse.receiver_charge_rows(
            current, following, target, axis, sign)
        rows.append({
            "axis": axis,
            "sign": sign,
            "DEL_relative_residual": float(
                np.linalg.norm(residual)
                / max(np.linalg.norm(target_images[:, column]), 1e-30)),
            "launch_norm": launch_norm,
            "launch_field_power_fractions": inverse.field_power(launch),
            "launch_support_power_fractions": support_fractions(launch),
            "distance_from_two_copy_3x3x3_envelope": copy_defect,
            "distance_from_full_native_first_shell": shell_defect,
            "receiver_interval_charge_rows": receiver,
        })
    max_residual = max(row["DEL_relative_residual"] for row in rows)
    max_tail = max(row["launch_support_power_fractions"][
        "outside_first_shell"] for row in rows)
    result = {
        "status": "V4_CARRIER_CONTRAST_FULL_INVERSE_LAUNCH_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": (
            "complete transfer-softened interval action plus half-weight "
            "transfer-activated chi bond and 1/(2 chi0) frame reflection"),
        "box": "side-five periodic unrestricted linear response control",
        "dt": inverse.DT,
        "previous_block_dimension": dimension,
        "rows": rows,
        "max_DEL_relative_residual": max_residual,
        "max_outside_first_shell_launch_power": max_tail,
        "verdict": (
            "UNRESTRICTED_NATIVE_LINEAR_EDGE_LAUNCH_EXISTS"
            if max_residual < 1e-10 else
            "UNRESTRICTED_NATIVE_LINEAR_EDGE_LAUNCH_FAILS"),
        "claim_boundary": (
            "An invertible linear launch is not a localized moving orbit. "
            "The support fractions determine whether it is a small native "
            "dressing or an extended box-dependent preparation."),
        "corrective_action": (
            "If the unrestricted launch closes and its tail is localized, "
            "use it as the seed for a nonlinear relative-periodic solve. If "
            "the tail is extended, repeat on a larger box and derive the "
            "missing action response from the nondecaying component."),
    }
    (HERE / "localized_carrier_contrast_full_inverse_launch.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    np.savez_compressed(
        HERE / "localized_carrier_contrast_full_inverse_launch.npz",
        previous=previous, current=current, following=following,
        targets=target_matrix, launches=launches)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
