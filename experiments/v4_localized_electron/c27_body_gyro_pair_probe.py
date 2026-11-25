"""Pair the transport-positive C27 potential with a native bond one-form.

The joint-W body-diagonal Dirichlet energy raises fixed-charge transport to
rank three but exerts an in-plane rest torque.  On every occupied body
diagonal the resting W overlap is purely imaginary.  The reciprocal cubic
bond one-form

    L_g = g sum_body Im[
        W_i^dagger D_t W_j + W_j^dagger D_t W_i]

has exactly the complementary timing parity: the native internal phase clock
turns its variation into a rest force without adding a holding potential or a
preferred quarter-turn.  This script derives the coefficient solely by
cancelling the measured native rest force, then repeats the fixed-charge rank
gate.  No Dirac target enters the coefficient or action.

The one-form does not alter the potential energy, but a surviving candidate
would still require a nondegenerate symplectic-form proof, canonical momenta,
reversible update, full spectrum, and nonlinear movement gates.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from torch.func import grad

import plaquette_factorization_completion_probe as base
import c27_body_diagonal_transport_probe as c27


HERE = Path(__file__).resolve().parent
BODY_OFFSETS = tuple(itertools.product((-1, 1), repeat=3))
c27.MODE = "joint_W"
GYRO_COEFFICIENT = 0.0
POTENTIAL_COEFFICIENT = 1.0


def area(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = base.inverse.current_audit.mold.unpack_tensor(
        raw, base.inverse.SIDE)
    return torch.linalg.cross(chi, psi, dim=-1) / base.inverse.rank.R


def body_gyro_interval(left: torch.Tensor,
                       right: torch.Tensor) -> torch.Tensor:
    left_area = area(left)
    right_area = area(right)
    midpoint = (left_area + right_area) / 2
    velocity = (right_area - left_area) / base.inverse.DT
    result = torch.zeros((), dtype=torch.float64)
    for offset in BODY_OFFSETS:
        neighbor_midpoint = torch.roll(
            midpoint, shifts=offset, dims=(0, 1, 2))
        neighbor_velocity = torch.roll(
            velocity, shifts=offset, dims=(0, 1, 2))
        first = torch.sum(torch.conj(midpoint) * neighbor_velocity, dim=-1)
        second = torch.sum(
            torch.conj(neighbor_midpoint) * velocity, dim=-1)
        result = result + torch.sum(torch.imag(first + second))
    # Directed offsets count the symmetric bond expression twice.
    return base.inverse.DT * result / 4


def interval_action(left: torch.Tensor, right: torch.Tensor,
                    potential: float | torch.Tensor,
                    gyro: float | torch.Tensor) -> torch.Tensor:
    return (
        base.inverse.interval_action(left, right)
        - base.inverse.DT * potential
        * (c27.body_potential(left) + c27.body_potential(right)) / 2
        + gyro * body_gyro_interval(left, right)
    )


def three_layer_action(previous: torch.Tensor, current: torch.Tensor,
                       following: torch.Tensor,
                       potential: float | torch.Tensor,
                       gyro: float | torch.Tensor) -> torch.Tensor:
    return interval_action(previous, current, potential, gyro) + interval_action(
        current, following, potential, gyro)


def make_derivatives(potential: float, gyro: float):
    interval = lambda left, right: interval_action(
        left, right, potential, gyro)
    three = lambda previous, current, following: three_layer_action(
        previous, current, following, potential, gyro)
    return grad(three, argnums=1), grad(interval, argnums=1)


def main() -> None:
    global GYRO_COEFFICIENT
    base.inverse.rank.lock_inertia()
    base.inverse.rest.SIDE = base.inverse.SIDE
    base.inverse.rest.DT = base.inverse.DT
    previous, current, following, theta = base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))

    baseline_del, _ = make_derivatives(0.0, 0.0)
    potential_del, _ = make_derivatives(1.0, 0.0)
    gyro_del, _ = make_derivatives(0.0, 1.0)
    baseline = baseline_del(*tensors).detach().numpy()
    potential_force = potential_del(*tensors).detach().numpy() - baseline
    gyro_force = gyro_del(*tensors).detach().numpy() - baseline
    coefficient = float(
        -np.vdot(gyro_force, potential_force).real
        / max(np.vdot(gyro_force, gyro_force).real, 1e-300))
    GYRO_COEFFICIENT = coefficient
    paired_del, paired_momentum = make_derivatives(1.0, coefficient)
    paired = paired_del(*tensors).detach().numpy()
    remaining = paired - baseline
    correlation = float(
        np.vdot(gyro_force, potential_force).real
        / max(np.linalg.norm(gyro_force) * np.linalg.norm(potential_force),
              1e-300))
    rest_relative = float(
        np.linalg.norm(remaining) / max(np.linalg.norm(potential_force), 1e-300))

    # Reuse the exact action-derived charge response with paired momenta.
    base.del_residual = paired_del
    base.right_momentum = paired_momentum
    dimension = previous.size
    previous_block = base.batched_jvp("previous", tensors, np.eye(dimension))
    factor = lu_factor(previous_block)
    rows = []
    for axis in range(3):
        labels, targets = base.component.component_targets(
            axis, np.exp(1j * theta))
        images = base.batched_jvp("following", tensors, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            base.response(previous, current, following,
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

    rank_three = all(
        row["fixed_charge_center_rank"] == 3
        and row["relative_third_singular_value"] > 1e-6
        for row in rows)
    rest_closed = rest_relative < 1e-8
    omega = theta / base.inverse.DT
    result = {
        "status": "V4_C27_BODY_GYRO_PAIRED_ACTION_SCREEN",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action_delta": (
            "unit positive joint-W body-diagonal Dirichlet potential plus "
            "a symmetric joint-W body-diagonal gyroscopic one-form"),
        "coefficient_rule": (
            "gyro coefficient is the least-squares cancellation of the unit "
            "C27 rest force; no transport or Dirac target is used"),
        "gyro_coefficient": coefficient,
        "native_discrete_clock_rate": float(omega),
        "coefficient_times_clock": float(coefficient * omega),
        "rest_force_correlation": correlation,
        "rest_force_relative_remainder": rest_relative,
        "paired_rest_DEL_rms": float(
            np.linalg.norm(paired) / np.sqrt(paired.size)),
        "axis_rows": rows,
        "all_axes_robust_rank_three": rank_three,
        "verdict": (
            "C27_GYRO_PAIR_PRESERVES_REST_AND_CLOSES_TRANSPORT"
            if rest_closed and rank_three else
            "C27_GYRO_PAIR_INCOMPLETE"),
        "claim_boundary": (
            "Rest-force and linear fixed-charge rank screen only. It does not "
            "prove symplectic nondegeneracy, bounded dynamics, full spectral "
            "stability, nonlinear handoff, or Dirac current matching."),
    }
    output = HERE / "c27_body_gyro_pair_probe.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "gyro_coefficient": coefficient,
        "coefficient_times_clock": result["coefficient_times_clock"],
        "rest_force_correlation": correlation,
        "rest_force_relative_remainder": rest_relative,
        "paired_rest_DEL_rms": result["paired_rest_DEL_rms"],
        "ranks": [row["fixed_charge_center_rank"] for row in rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
