"""Test lattice-native motion without an intravoxel position.

The four-component carrier is an internal fiber on an extended envelope.  The
native and independently evolved C19-Dirac packets redistribute amplitudes
only among lattice sites.  This gate compares both literal Psi site density
and the action-derived tangent Noether density over one internal cycle.  It
does not require a fractionally translated compact eight-site shape at
intermediate updates.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_current as current_bridge
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import localized_stability as c19


HERE = Path(__file__).resolve().parent


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def propagators(alpha, beta, dt):
    frequencies = 2 * np.pi * np.fft.fftfreq(gate.SIDE)
    result = np.empty((gate.SIDE, gate.SIDE, gate.SIDE, 4, 4), complex)
    for index in itertools.product(range(gate.SIDE), repeat=3):
        momentum = np.asarray([frequencies[value] for value in index])
        transport = gate.c19_first_symbol(momentum)
        hamiltonian = native_time.CHI0 * (beta - np.eye(4))
        for axis in range(3):
            hamiltonian = hamiltonian + transport[axis] * alpha[axis]
        result[index] = expm(-1j * dt * hamiltonian)
    return result


def advance_dirac(field, operators):
    fourier = np.fft.fftn(field, axes=(0, 1, 2))
    advanced = np.einsum("...ab,...b->...a", operators, fourier)
    return np.fft.ifftn(advanced, axes=(0, 1, 2))


def c19_average(density):
    result = density.copy()
    normalization = 1.0
    for offset, weight in c19.DIRECTIONS:
        result = result + weight * np.roll(
            density, shift=offset, axis=(0, 1, 2))
        normalization += weight
    return result / normalization


def normalized_density_error(actual, target):
    actual_scale = np.sum(actual)
    target_scale = np.sum(target)
    if abs(actual_scale) < 1e-30 or abs(target_scale) < 1e-30:
        return float("inf")
    actual_normalized = actual / actual_scale
    target_normalized = target / target_scale
    return float(np.linalg.norm(actual_normalized - target_normalized)
                 / max(np.linalg.norm(target_normalized), 1e-30))


def psi_pixel_density(raw):
    shaped = raw.reshape(gate.SIDE, gate.SIDE, gate.SIDE, 4, 3)
    psi = shaped[..., 0, :] + 1j * shaped[..., 1, :]
    return np.sum(abs(psi)**2, axis=-1)


def main() -> None:
    bare, _ = action.expected_blocks()
    mass0, gyro0, stiffness0 = ablation.blocks()[
        "complete_research_action"]
    mass, gyro, stiffness = map(sparse, (mass0, gyro0, stiffness0))
    dt = native_time.DT / 16
    lhs = (mass / dt + gyro / 2).tocsc()
    current_block = (dt * stiffness - 2 * mass / dt).tocsc()
    previous_block = (mass / dt - gyro / 2).tocsc()
    factor = splu(lhs)
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))
    sample_steps = set(np.linspace(0, steps, 65, dtype=int).tolist())

    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        operators = propagators(alpha, beta, dt)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS,
            scales[flavor])
        for axis, sign in ((0, -1), (0, 1), (3, 1)):
            dirac_previous, dirac_current, _next, momentum = gate.packet(
                axis, sign, alpha, beta, dt)
            q_previous = raw @ gate.real_coordinates(dirac_previous)
            q_current = raw @ gate.real_coordinates(dirac_current)
            initial_density = None
            initial_target_density = None
            initial_pixel_density = None
            initial_target_pixel_density = None
            samples = []
            for step in range(steps + 1):
                if step in sample_steps:
                    native_density = current_bridge.tangent_charge_density(
                        q_previous, q_current, "Psi",
                        mass, gyro, stiffness, dt)
                    target_previous = raw @ gate.real_coordinates(
                        dirac_previous)
                    target_current = raw @ gate.real_coordinates(
                        dirac_current)
                    target_density = current_bridge.tangent_charge_density(
                        target_previous, target_current, "Psi",
                        mass, gyro, stiffness, dt)
                    native_pixel_density = psi_pixel_density(q_current)
                    target_pixel_density = psi_pixel_density(target_current)
                    if initial_density is None:
                        initial_density = native_density
                        initial_target_density = target_density
                        initial_pixel_density = native_pixel_density
                        initial_target_pixel_density = target_pixel_density
                    native_center, native_total = current_bridge.center(
                        native_density)
                    target_center, target_total = current_bridge.center(
                        target_density)
                    initial_center, _ = current_bridge.center(initial_density)
                    initial_target_center, _ = current_bridge.center(
                        initial_target_density)
                    native_shift = current_bridge.wrapped_shift(
                        initial_center, native_center)
                    target_shift = current_bridge.wrapped_shift(
                        initial_target_center, target_center)
                    native_pixel_center, _ = current_bridge.center(
                        native_pixel_density)
                    target_pixel_center, _ = current_bridge.center(
                        target_pixel_density)
                    initial_pixel_center, _ = current_bridge.center(
                        initial_pixel_density)
                    initial_target_pixel_center, _ = current_bridge.center(
                        initial_target_pixel_density)
                    native_pixel_shift = current_bridge.wrapped_shift(
                        initial_pixel_center, native_pixel_center)
                    target_pixel_shift = current_bridge.wrapped_shift(
                        initial_target_pixel_center, target_pixel_center)
                    samples.append({
                        "step": step,
                        "time": step * dt,
                        "cycles": step * dt * native_time.CHI0 / (2*np.pi),
                        "raw_density_relative_error": normalized_density_error(
                            native_density, target_density),
                        "C19_averaged_density_relative_error": normalized_density_error(
                            c19_average(native_density),
                            c19_average(target_density)),
                        "Psi_pixel_density_relative_error": normalized_density_error(
                            native_pixel_density, target_pixel_density),
                        "C19_averaged_Psi_pixel_density_relative_error": normalized_density_error(
                            c19_average(native_pixel_density),
                            c19_average(target_pixel_density)),
                        "native_center_shift": native_shift.tolist(),
                        "dirac_center_shift": target_shift.tolist(),
                        "center_shift_error": float(np.linalg.norm(
                            native_shift - target_shift)),
                        "native_Psi_pixel_center_shift": (
                            native_pixel_shift.tolist()),
                        "dirac_Psi_pixel_center_shift": (
                            target_pixel_shift.tolist()),
                        "Psi_pixel_center_shift_error": float(np.linalg.norm(
                            native_pixel_shift - target_pixel_shift)),
                        "native_charge_relative_change": float(
                            (native_total - samples[0]["native_total"])
                            / max(abs(samples[0]["native_total"]), 1e-30)
                            if samples else 0.0),
                        "native_total": native_total,
                        "target_total": target_total,
                    })
                if step == steps:
                    break
                q_following = factor.solve(
                    -(current_block @ q_current
                      + previous_block @ q_previous))
                q_previous, q_current = q_current, q_following
                dirac_previous, dirac_current = (
                    dirac_current,
                    advance_dirac(dirac_current, operators))
            rows.append({
                "flavor": flavor,
                "direction": "diagonal" if axis == 3 else "x",
                "sign": sign,
                "momentum": momentum.tolist(),
                "samples": samples,
                "maximum_raw_density_relative_error": max(
                    sample["raw_density_relative_error"]
                    for sample in samples),
                "maximum_C19_averaged_density_relative_error": max(
                    sample["C19_averaged_density_relative_error"]
                    for sample in samples),
                "maximum_center_shift_error": max(
                    sample["center_shift_error"] for sample in samples),
                "maximum_Psi_pixel_center_shift_error": max(
                    sample["Psi_pixel_center_shift_error"]
                    for sample in samples),
                "maximum_C19_averaged_Psi_pixel_density_relative_error": max(
                    sample["C19_averaged_Psi_pixel_density_relative_error"]
                    for sample in samples),
                "maximum_charge_drift": max(abs(
                    sample["native_charge_relative_change"])
                    for sample in samples),
            })

    worst_density = max(
        row["maximum_C19_averaged_density_relative_error"] for row in rows)
    worst_center = max(row["maximum_center_shift_error"] for row in rows)
    worst_charge = max(row["maximum_charge_drift"] for row in rows)
    worst_pixel_density = max(
        row["maximum_C19_averaged_Psi_pixel_density_relative_error"]
        for row in rows)
    worst_pixel_center = max(
        row["maximum_Psi_pixel_center_shift_error"] for row in rows)
    result = {
        "status": "V4_DISCRETE_SITE_TRANSFER_OBSERVABLE_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "interpretation": (
            "No intravoxel coordinate: motion is site-amplitude transfer; "
            "literal Psi site density is the direct lattice observable, "
            "while native Noether density/current is reported separately."),
        "dt": dt,
        "cycles": 1.0,
        "rows": rows,
        "maximum_C19_averaged_density_relative_error": worst_density,
        "maximum_center_shift_error": worst_center,
        "maximum_charge_drift": worst_charge,
        "maximum_C19_averaged_Psi_pixel_density_relative_error": (
            worst_pixel_density),
        "maximum_Psi_pixel_center_shift_error": worst_pixel_center,
        "verdict": (
            "DISCRETE_SITE_TRANSFER_OBSERVABLE_PASS"
            if worst_pixel_density < 0.05 and worst_pixel_center < 0.01
            else "DISCRETE_SITE_TRANSFER_OBSERVABLE_OPEN"),
        "claim_boundary": (
            "One-cycle linear tangent site-density and Noether-density test "
            "on side five. It does not prove nonlinear 100-cycle movement, "
            "box refinement, or V4-006."),
    }
    (HERE / "discrete_site_transfer_observable_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
