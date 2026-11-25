"""Compare native Noether-charge-center motion with discrete Dirac.

The even combination of equal positive/negative packet amplitudes removes the
linear interference with the finite rest carrier and leaves the packet's
quadratic phase charge.  This avoids reconstructing spinor coordinates from
the ill-conditioned overlapping-cube synthesis and also separates net
translation from the carrier's internal circulating cut current.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import live_clifford_bond_action as action
import live_face_noether_continuity_gate as noether


HERE = Path(__file__).resolve().parent
SIDE = 5


def local_charge(left: torch.Tensor, right: torch.Tensor,
                 field: str) -> np.ndarray:
    _grad_left, grad_right = torch.func.grad(
        action.interval, argnums=(0, 1))(left, right, native_time.DT)
    shaped_gradient = grad_right.reshape(SIDE, SIDE, SIDE, 12)
    shaped_generator = noether.generator(right, field).reshape(
        SIDE, SIDE, SIDE, 12)
    return torch.sum(shaped_gradient*shaped_generator, dim=-1).detach().numpy()


def circular_center(density: np.ndarray) -> np.ndarray:
    result = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = SIDE
        phase = np.exp(2j*np.pi*np.arange(SIDE)/SIDE).reshape(shape)
        moment = np.sum(density*phase)
        result.append(float(np.angle(moment)*SIDE/(2*np.pi)))
    return np.asarray(result)


def wrapped(value: np.ndarray) -> np.ndarray:
    return (value+SIDE/2) % SIDE-SIDE/2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frequency-branch",
                        choices=("positive", "negative"),
                        default="positive")
    args = parser.parse_args()
    positive = np.load(HERE/args.positive)
    negative = np.load(HERE/args.negative)
    plus_states = [torch.tensor(value, dtype=torch.float64)
                   for value in positive["states"]]
    minus_states = [torch.tensor(value, dtype=torch.float64)
                    for value in negative["states"]]
    rest = positive["rest"]
    rotation = positive["rotation"]
    rest_states = [torch.tensor(
        np.linalg.matrix_power(rotation, index-1)@rest,
        dtype=torch.float64) for index in range(len(plus_states))]

    basis = dirac.flavor_bases()["plus"]
    alpha, beta = dirac.compressed_clifford(basis)
    _previous, target, _following, momentum = dirac.packet(
        0, -1, alpha, beta, native_time.DT, centered_time=True,
        frequency_branch=args.frequency_branch)
    targets = [target]
    for _ in range(len(plus_states)-2):
        targets.append(dirac.centered_dirac_step(
            targets[-1], alpha, beta, 1, native_time.DT,
            args.frequency_branch))
    target_centers = [dirac.circular_center(value) for value in targets]

    rows = []
    for field in ("Psi", "common"):
        densities = []
        totals = []
        centers = []
        for index in range(1, len(plus_states)):
            qp = local_charge(plus_states[index-1], plus_states[index], field)
            qm = local_charge(minus_states[index-1], minus_states[index], field)
            qr = local_charge(rest_states[index-1], rest_states[index], field)
            density = (qp+qm-2*qr)/2
            densities.append(density)
            totals.append(float(np.sum(density)))
            centers.append(circular_center(density))
        native_displacements = [wrapped(value-centers[0]) for value in centers]
        target_displacements = [wrapped(value-target_centers[0])
                                for value in target_centers]
        count = min(len(native_displacements), len(target_displacements))
        errors = [float(np.linalg.norm(native_displacements[index]
                                       - target_displacements[index]))
                  for index in range(count)]
        rows.append({
            "phase_symmetry": field,
            "total_quadratic_charges": totals,
            "native_charge_center_displacements": [v.tolist()
                                                    for v in native_displacements],
            "dirac_probability_center_displacements": [v.tolist()
                                                        for v in target_displacements],
            "maximum_absolute_center_displacement_error": max(errors),
            "final_native_direction_dot_dirac": float(np.dot(
                native_displacements[count-1], target_displacements[count-1])),
        })
    result = {
        "status": "V4_LIVE_FACE_NOETHER_CENTER_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "frequency_branch": args.frequency_branch,
        "momentum": momentum.tolist(),
        "rows": rows,
        "verdict": (
            "NOETHER_CENTER_DIRECTION_MATCHES_DIRAC"
            if all(row["final_native_direction_dot_dirac"] > 0 for row in rows)
            else "NOETHER_CENTER_DIRECTION_MISMATCH"),
        "claim_boundary": (
            "Four-update quadratic Noether-charge-center comparison for one "
            "copy and one direction. It is not a full handoff or V4-006 pass."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
