"""Diagnose the missing compact-gradient part of the live face readout.

The candidate lift is deliberately small: the historical local Psi lift plus
live chi with a constant response and one positive face-Laplacian response,

    chi_lift = a * f + b * L_face f.

This file is a discovery diagnostic, not an admitted readout.  Parameters are
identified on axis packets and evaluated on held-out diagonal, width and wave
number controls.  Dirac is used only after the native DEL step as a comparator;
it never enters the action, force, or implicit solve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import live_clifford_tangent_gate as tangent


HERE = Path(__file__).resolve().parent


def center_face_laplacian() -> np.ndarray:
    """Positive six-face graph Laplacian in the packet coordinate order."""
    side = gate.SIDE
    count = side**3
    result = np.zeros((count, count))
    for flat in range(count):
        point = np.unravel_index(flat, (side, side, side))
        result[flat, flat] = 6.0
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(point)
                neighbor[axis] = (neighbor[axis]+sign) % side
                other = np.ravel_multi_index(tuple(neighbor), (side,)*3)
                result[flat, other] -= 1.0
    # real_coordinates is center-major, then four components, then re/im.
    return np.kron(result, np.eye(8))


def components(basis, psi_frame, chi_frame, sites):
    zero = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, 0j)
    real = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, 1+0j)-zero
    imag = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, 1j)-zero
    return zero, real, imag


def lift(parts, laplacian, values, coordinate):
    a_real, a_imag, b_real, b_imag = values
    psi, chi_real, chi_imag = parts
    gradient = laplacian@coordinate
    return (psi@coordinate
            + a_real*(chi_real@coordinate)
            + a_imag*(chi_imag@coordinate)
            + b_real*(chi_real@gradient)
            + b_imag*(chi_imag@gradient))


def packet_coordinates(axis, sign, alpha, beta, width, wave_number):
    previous, current, following, momentum = gate.packet(
        axis, sign, alpha, beta, native_time.DT,
        width=width, carrier_wave_number=wave_number)
    return tuple(gate.real_coordinates(value)
                 for value in (previous, current, following)), momentum


def main() -> None:
    os.environ.setdefault("LFM_LIVE_CLIFFORD_LINK",
                          "joint_face_wedge_laplacian_smooth")
    os.environ.setdefault("LFM_LIVE_CLIFFORD_SCALE", str(1/19))
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    blocks = tangent.tangent_blocks((rotation.T@rest, rest, rotation@rest))
    previous_block, current_block, following_block = blocks
    factor = lu_factor(following_block)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    laplacian = center_face_laplacian()
    result_rows = []

    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        parts = components(basis, psi_frame, chi_frame, sites)
        # Train on the six signed coordinate axes.  Diagonals and alternate
        # localization scales are not used to determine the coefficients.
        training = []
        for axis in range(3):
            for sign in (-1, 1):
                coordinates, _ = packet_coordinates(
                    axis, sign, alpha, beta, 1.35, 0.35)
                training.append(coordinates)

        def residual(values):
            columns = []
            for previous, current, following in training:
                q_previous = lift(parts, laplacian, values, previous)
                q_current = lift(parts, laplacian, values, current)
                q_target = lift(parts, laplacian, values, following)
                q_native = lu_solve(
                    factor,
                    -(previous_block@q_previous+current_block@q_current))
                columns.append((q_native-q_target)
                               / max(np.linalg.norm(q_target), 1e-30))
            return np.concatenate(columns)

        prior = json.loads(
            (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
                encoding="utf-8"))
        old = next(row for row in prior["rows"] if row["flavor"] == flavor)
        old_case = next(case for case in old["cases"]
                        if case["envelope"] == "all_125_centers")
        seed = np.asarray((old_case["chi_scale_real"],
                           old_case["chi_scale_imag"], 0.0, 0.0))
        # The joint tangent is normalized by the native kinetic metric.  A
        # coefficient much larger than 1/sqrt(B) replaces the Psi carrier by
        # a chi-dominated disturbance and is not the same electron.  The
        # canonical inertia therefore supplies an independent admissible
        # range rather than a fitted numerical cutoff.
        natural_limit = 1/np.sqrt(native_time.B)
        fit = least_squares(
            residual, seed, method="trf", max_nfev=120,
            bounds=(-natural_limit, natural_limit),
            xtol=1e-11, ftol=1e-11, gtol=1e-11)

        controls = []
        specifications = [
            (0, -1, 1.35, 0.35, "train_axis"),
            (1, 1, 1.35, 0.35, "train_axis"),
            (2, -1, 1.35, 0.35, "train_axis"),
            (3, -1, 1.35, 0.35, "held_diagonal"),
            (3, 1, 0.95, 0.20, "held_narrow_slow"),
            (0, 1, 1.60, 0.20, "held_wide_slow"),
        ]
        for axis, sign, width, wave_number, role in specifications:
            coordinates, momentum = packet_coordinates(
                axis, sign, alpha, beta, width, wave_number)
            previous, current, following = coordinates
            q_previous = lift(parts, laplacian, fit.x, previous)
            q_current = lift(parts, laplacian, fit.x, current)
            q_target = lift(parts, laplacian, fit.x, following)
            q_native = lu_solve(
                factor, -(previous_block@q_previous+current_block@q_current))
            controls.append({
                "role": role,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "width": width,
                "wave_number": wave_number,
                "momentum": momentum.tolist(),
                "relative_error": float(
                    np.linalg.norm(q_native-q_target)
                    / max(np.linalg.norm(q_target), 1e-30)),
            })
        result_rows.append({
            "flavor": flavor,
            "coefficients": {
                "constant_real": float(fit.x[0]),
                "constant_imag": float(fit.x[1]),
                "face_laplacian_real": float(fit.x[2]),
                "face_laplacian_imag": float(fit.x[3]),
            },
            "optimizer_success": bool(fit.success),
            "optimizer_evaluations": int(fit.nfev),
            "native_kinetic_coefficient_limit": float(natural_limit),
            "coefficient_at_limit": bool(
                np.max(np.abs(fit.x)) > 0.999*natural_limit),
            "training_rms": float(np.linalg.norm(fit.fun)
                                  / np.sqrt(fit.fun.size)),
            "controls": controls,
        })

    held = [control for row in result_rows for control in row["controls"]
            if control["role"].startswith("held")]
    result = {
        "status": "V4_FACE_COMPACT_GRADIENT_BRIDGE_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "readout_admitted": False,
        "readout_family": "chi = a envelope + b positive_face_laplacian(envelope)",
        "rows": result_rows,
        "maximum_held_out_relative_error": max(
            control["relative_error"] for control in held),
        "verdict": "DIAGNOSTIC_ONLY",
        "claim_boundary": (
            "The coefficients were selected with a Dirac comparator. They "
            "identify whether one compact-gradient response has sufficient "
            "rank, but are not an independently derived native readout and "
            "cannot close V4-006."),
    }
    path = HERE/"live_face_compact_gradient_bridge.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
