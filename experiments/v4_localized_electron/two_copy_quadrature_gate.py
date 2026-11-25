"""Test whether both native cubical copies form a transport quadrature relay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--relative-phase", type=float, default=np.pi/2)
    parser.add_argument("--output", default="two_copy_quadrature_gate.json")
    args = parser.parse_args()
    rank.lock_inertia()
    with np.load(args.cache) as archive:
        previous_block = archive["previous"]
        current_block = archive["current"]
        following_block = archive["following"]
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    dt = gate.DT
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    factor = splu(gate.sparse(following_block@rotate_plus))
    current_operator = gate.sparse(current_block)
    previous_operator = gate.sparse(previous_block@rotate_minus)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)

    pieces = []
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = gate.packet_columns(basis, psi_frame, chi_frame, sites, 0j)
        p, q, _, momentum = dirac.packet(
            0, 1, alpha, beta, dt, centered_time=True,
            frequency_branch="negative")
        pieces.append((flavor, alpha, beta, raw, p, q, momentum))
    phase = np.exp(1j*args.relative_phase)
    weights = (1/np.sqrt(2), phase/np.sqrt(2))

    def lift(index: int, field_name: str) -> np.ndarray:
        total = np.zeros(rest.size)
        for weight, piece in zip(weights, pieces):
            field = piece[4] if field_name == "previous" else piece[5]
            total += piece[3]@dirac.real_coordinates(weight*field)
        return total

    q_previous = lift(0, "previous")
    q_current = lift(0, "current")
    raw_all = np.column_stack((pieces[0][3], pieces[1][3]))
    retained_basis = envelope.orth(raw_all)
    steps = int(np.ceil(2*np.pi/rank.R/dt))
    dirac_fields = [piece[5].copy() for piece in pieces]
    maximum_shape = 0.0
    maximum_cross = 0.0
    maximum_norm = 0.0
    for step in range(steps+1):
        q_following = factor.solve(
            -(current_operator@q_current+previous_operator@q_previous))
        targets = []
        target_previous = np.zeros(rest.size)
        target_current = np.zeros(rest.size)
        target_following = np.zeros(rest.size)
        for field_index, (weight, piece) in enumerate(zip(weights, pieces)):
            _, alpha, beta, raw, _, initial, _ = piece
            target = dirac_fields[field_index]
            before = dirac.centered_dirac_step(
                target, alpha, beta, -1, dt, "negative", 1.0)
            after = dirac.centered_dirac_step(
                target, alpha, beta, 1, dt, "negative", 1.0)
            target_previous += raw@dirac.real_coordinates(weight*before)
            target_current += raw@dirac.real_coordinates(weight*target)
            target_following += raw@dirac.real_coordinates(weight*after)
        native_velocity = (
            gate.rotate_native_phase(q_following, theta)
            - gate.rotate_native_phase(q_previous, -theta))/(2*dt)
        target_velocity = (
            gate.rotate_native_phase(target_following, theta)
            - gate.rotate_native_phase(target_previous, -theta))/(2*dt)
        native_amplitude = 0.5*(
            q_current-gate.rotate_native_phase(native_velocity, np.pi/2)/rank.R)
        target_amplitude = 0.5*(
            target_current-gate.rotate_native_phase(target_velocity, np.pi/2)/rank.R)
        target_i = gate.rotate_native_phase(target_amplitude, np.pi/2)
        basis = np.column_stack((target_amplitude, target_i))
        coefficients, *_ = np.linalg.lstsq(basis, native_amplitude, rcond=1e-12)
        aligned = basis@(coefficients/max(np.linalg.norm(coefficients), 1e-30))
        retained = retained_basis@(retained_basis.T@native_amplitude)
        maximum_shape = max(maximum_shape, float(
            np.linalg.norm(native_amplitude-aligned)
            / max(np.linalg.norm(target_amplitude), 1e-30)))
        maximum_cross = max(maximum_cross, float(
            np.linalg.norm(native_amplitude-retained)
            / max(np.linalg.norm(native_amplitude), 1e-30)))
        maximum_norm = max(maximum_norm, abs(float(
            np.linalg.norm(native_amplitude)
            / max(np.linalg.norm(target_amplitude), 1e-30))-1))
        if step == steps:
            break
        q_previous, q_current = q_current, q_following
        for field_index, piece in enumerate(pieces):
            dirac_fields[field_index] = dirac.centered_dirac_step(
                dirac_fields[field_index], piece[1], piece[2], 1, dt,
                "negative", 1.0)
    result = {
        "status": "V4_TWO_COPY_QUADRATURE_RELAY_TEST",
        "relative_phase": args.relative_phase,
        "cycles": 1.0,
        "maximum_phase_space_shape_error": maximum_shape,
        "maximum_phase_space_cross_block": maximum_cross,
        "maximum_phase_space_norm_drift": maximum_norm,
        "verdict": ("TWO_COPY_RELAY_PASS" if maximum_shape < 0.05
                    and maximum_cross < 0.02 and maximum_norm < 0.05
                    else "TWO_COPY_RELAY_OPEN"),
    }
    (HERE/args.output).write_text(json.dumps(result, indent=2)+"\n",
                                 encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
