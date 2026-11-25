"""Pull the action-derived Noether charge through the extended packet.

The local discrete momentum is D_2 L(q_{n-1},q_n). Contracting it with the
local phase generator gives the native discrete Noether density. Symmetric
plus/minus perturbations remove the rotating carrier's background and linear
cross term, leaving the packet's quadratic charge without a fitted formula.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as packet_gate
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent
SIDE = 5


def generator(raw, mode):
    shaped = raw.reshape(SIDE, SIDE, SIDE, 4, 3)
    value = torch.zeros_like(shaped)
    value[..., 0, :] = -shaped[..., 1, :]
    value[..., 1, :] = shaped[..., 0, :]
    if mode == "common":
        value[..., 2, :] = -shaped[..., 3, :]
        value[..., 3, :] = shaped[..., 2, :]
    return value


def tangent_generator(raw, mode):
    """Generator in the co-rotating tangent coordinates."""
    return generator(torch.tensor(raw, dtype=torch.float64), mode).numpy()


def tangent_charge_density(left, right, mode, mass, gyro, stiffness, dt):
    """Exact discrete Noether density of the quadratic tangent action.

    The linear update is the DEL equation of

      Ld(x,y) = |y-x|_M^2/(2 dt) - x^T G y/2
                 - dt (x^T K x + y^T K y)/4.

    Contracting D2 Ld with the phase generator at ``right`` gives the
    conserved discrete charge.  This is the appropriate current bridge for
    the tangent packet; the nonlinear finite-amplitude charge is retained
    below as a separate diagnostic rather than being conflated with it.
    """
    momentum = (mass@(right-left)/dt + gyro@left/2
                - dt*stiffness@right/2)
    image = tangent_generator(right, mode).reshape(-1)
    return (momentum*image).reshape(SIDE, SIDE, SIDE, 12).sum(axis=-1)


def charge_density(left, right, mode):
    left_tensor = torch.tensor(left, dtype=torch.float64)
    right_tensor = torch.tensor(
        right, dtype=torch.float64, requires_grad=True)
    value = action.interval_action(left_tensor, right_tensor)
    momentum = torch.autograd.grad(value, right_tensor)[0].reshape(
        SIDE, SIDE, SIDE, 4, 3)
    return torch.sum(momentum*generator(right_tensor, mode),
                     dim=(-1, -2)).detach().numpy()


def packet_charge(rest_left, rest_right, tangent_left, tangent_right,
                  mode, amplitude):
    plus = charge_density(
        rest_left+amplitude*tangent_left,
        rest_right+amplitude*tangent_right, mode)
    minus = charge_density(
        rest_left-amplitude*tangent_left,
        rest_right-amplitude*tangent_right, mode)
    base = charge_density(rest_left, rest_right, mode)
    return (plus+minus-2*base)/(2*amplitude**2)


def center(density):
    result = []
    total = np.sum(density)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = SIDE
        phase = np.exp(2j*np.pi*np.arange(SIDE)/SIDE).reshape(shape)
        value = np.sum(density*phase)
        result.append(float(np.angle(value)*SIDE/(2*np.pi)))
    return np.asarray(result), float(total)


def wrapped_shift(first, second):
    return (second-first+SIDE/2) % SIDE-SIDE/2


def main():
    bare, matrices = action.expected_blocks()
    mass, gyro, stiffness = (matrices[name]
                             for name in ("mass", "gyro", "stiffness"))
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)
    theta = float(np.arccos(1-dt**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(bare["rest"].size, theta)
    rest_previous = rotation.T@bare["rest"]
    rest_current = bare["rest"]
    rest_following = rotation@bare["rest"]
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for flavor, basis in packet_gate.flavor_bases().items():
        alpha, beta = packet_gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, packet_gate.OFFSETS,
            scales[flavor])
        for axis, sign in itertools.product(range(4), (-1, 1)):
            previous, current, following, momentum = packet_gate.packet(
                axis, sign, alpha, beta, dt)
            q_previous = raw@packet_gate.real_coordinates(previous)
            q_current = raw@packet_gate.real_coordinates(current)
            q_target = raw@packet_gate.real_coordinates(following)
            q_native = advance@q_current+memory@q_previous
            scale = 0.01*np.linalg.norm(rest_current)/np.linalg.norm(q_current)
            q_previous *= scale
            q_current *= scale
            q_target *= scale
            q_native *= scale
            # Convert co-rotating tangent layers back to laboratory layers.
            laboratory_previous = rotation.T@q_previous
            laboratory_current = q_current
            laboratory_target = rotation@q_target
            laboratory_native = rotation@q_native
            row = {
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "xyz"[axis],
                "sign": sign,
                "momentum": momentum.tolist(),
                "charges": {},
            }
            dirac_shift = wrapped_shift(
                packet_gate.circular_center(current),
                packet_gate.circular_center(following))
            row["dirac_probability_center_shift"] = dirac_shift.tolist()
            for mode in ("Psi", "common"):
                tangent_before = tangent_charge_density(
                    q_previous, q_current, mode, mass, gyro, stiffness, dt)
                tangent_native = tangent_charge_density(
                    q_current, q_native, mode, mass, gyro, stiffness, dt)
                tangent_target = tangent_charge_density(
                    q_current, q_target, mode, mass, gyro, stiffness, dt)
                tangent_center_before, tangent_total_before = center(
                    tangent_before)
                tangent_center_native, tangent_total_native = center(
                    tangent_native)
                tangent_center_target, tangent_total_target = center(
                    tangent_target)
                row.setdefault("tangent_charges", {})[mode] = {
                    "total_charge_before": tangent_total_before,
                    "native_total_charge_relative_change": float(
                        (tangent_total_native-tangent_total_before)
                        / max(abs(tangent_total_before), 1e-30)),
                    "dirac_target_total_charge_relative_change": float(
                        (tangent_total_target-tangent_total_before)
                        / max(abs(tangent_total_before), 1e-30)),
                    "native_charge_center_shift": wrapped_shift(
                        tangent_center_before, tangent_center_native).tolist(),
                    "embedded_dirac_charge_center_shift": wrapped_shift(
                        tangent_center_before, tangent_center_target).tolist(),
                    "native_vs_embedded_dirac_shift_error": float(
                        np.linalg.norm(
                            wrapped_shift(tangent_center_before,
                                          tangent_center_native)
                            - wrapped_shift(tangent_center_before,
                                            tangent_center_target))),
                }
                amplitude_rows = []
                for amplitude in (1.0, 0.5):
                    before = packet_charge(
                        rest_previous, rest_current,
                        laboratory_previous, laboratory_current,
                        mode, amplitude)
                    after_native = packet_charge(
                        rest_current, rest_following,
                        laboratory_current, laboratory_native,
                        mode, amplitude)
                    after_target = packet_charge(
                        rest_current, rest_following,
                        laboratory_current, laboratory_target,
                        mode, amplitude)
                    center_before, total_before = center(before)
                    center_native, total_native = center(after_native)
                    center_target, total_target = center(after_target)
                    native_shift = wrapped_shift(center_before, center_native)
                    target_shift = wrapped_shift(center_before, center_target)
                    amplitude_rows.append({
                        "amplitude": amplitude,
                        "total_charge_before": total_before,
                        "native_total_charge_relative_change": (
                            (total_native-total_before)
                            / max(abs(total_before), 1e-30)),
                        "dirac_target_total_charge_relative_change": (
                            (total_target-total_before)
                            / max(abs(total_before), 1e-30)),
                        "native_charge_center_shift": native_shift.tolist(),
                        "embedded_dirac_charge_center_shift": target_shift.tolist(),
                        "native_vs_embedded_dirac_shift_error": float(
                            np.linalg.norm(native_shift-target_shift)),
                    })
                row["charges"][mode] = amplitude_rows
            rows.append(row)
    maximum_conservation = max(
        abs(item["native_total_charge_relative_change"])
        for row in rows for item in row["tangent_charges"].values())
    maximum_shift_error = max(
        item["native_vs_embedded_dirac_shift_error"]
        for row in rows for item in row["tangent_charges"].values())
    nonlinear_maximum_conservation = max(
        abs(item["native_total_charge_relative_change"])
        for row in rows for mode in row["charges"].values() for item in mode)
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_CURRENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rows": rows,
        "maximum_native_charge_relative_change": maximum_conservation,
        "maximum_native_vs_embedded_dirac_charge_shift_error": maximum_shift_error,
        "nonlinear_finite_amplitude_maximum_charge_relative_change": (
            nonlinear_maximum_conservation),
        "verdict": (
            "EXTENDED_PACKET_NATIVE_CURRENT_PASS"
            if maximum_conservation < 1e-10 and maximum_shift_error < 5e-3
            else "EXTENDED_PACKET_NATIVE_CURRENT_OPEN"),
        "claim_boundary": (
            "Quadratic tangent charge from the nonlinear discrete action on "
            "a side-five one-step packet. It is not a nonlinear long-time or "
            "large-box transport proof."),
    }
    (HERE/"cube_incidence_extended_packet_current.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
