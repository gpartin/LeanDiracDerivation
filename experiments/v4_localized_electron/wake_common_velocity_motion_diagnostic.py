"""Recover native four-component motion for the stable common-velocity action."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import mixed_prong_dirac_long_tangent as gate
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
CACHE = HERE/os.environ.get(
    "LFM_TANGENT_CACHE",
    "wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
OUTPUT = os.environ.get(
    "LFM_MOTION_OUTPUT", "wake_common_velocity_motion_diagnostic.json")


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def unpack_coordinates(values: np.ndarray) -> np.ndarray:
    shaped = values.reshape(dirac.SIDE, dirac.SIDE, dirac.SIDE, 4, 2)
    return shaped[..., 0]+1j*shaped[..., 1]


def circular_density_center(density: np.ndarray) -> np.ndarray:
    result = []
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = dirac.SIDE
        phase = np.exp(2j*np.pi*np.arange(dirac.SIDE)/dirac.SIDE).reshape(shape)
        result.append(float(np.angle(np.sum(density*phase))*dirac.SIDE/(2*np.pi)))
    return np.asarray(result)


def native_centers(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi, chi = relative.unpack_raw(raw, dirac.SIDE)
    psi_density = np.sum(abs(psi)**2, axis=-1)
    joint_density = psi_density+rank.B*np.sum(abs(chi)**2, axis=-1)
    return (circular_density_center(psi_density),
            circular_density_center(joint_density))


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-gate.DT**2*rank.R**2/2))
    rminus = gate.phase_rotation_matrix(rest.size, -theta)
    rplus = gate.phase_rotation_matrix(rest.size, theta)
    data = np.load(CACHE)
    previous = sparse(data["previous"]@rminus)
    current = sparse(data["current"])
    following = sparse(data["following"]@rplus)
    factor = splu(following)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json")
                       .read_text(encoding="utf-8"))
    scale_row = next(row for row in prior["rows"] if row["flavor"] == "plus")
    scale_case = next(item for item in scale_row["cases"]
                      if item["envelope"] == "all_125_centers")
    scale = complex(scale_case["chi_scale_real"], scale_case["chi_scale_imag"])
    basis = dirac.flavor_bases()["plus"]
    alpha, beta = dirac.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, dirac.OFFSETS, scale)
    # The lift is ill-conditioned but full column rank.  A fixed pseudoinverse
    # is a measurement map only and never acts on the native recurrence.
    readout = np.linalg.pinv(raw, rcond=1e-9)
    packet_previous, packet_current, _packet_following, momentum = dirac.packet(
        0, 1, alpha, beta, gate.DT)
    q_previous = raw@dirac.real_coordinates(packet_previous)
    q_current = raw@dirac.real_coordinates(packet_current)

    frequencies = 2*np.pi*np.fft.fftfreq(dirac.SIDE)
    propagator = np.empty((dirac.SIDE,)*3+(4, 4), dtype=complex)
    for index in np.ndindex((dirac.SIDE,)*3):
        wave = np.asarray([frequencies[value] for value in index])
        symbol = dirac.c19_first_symbol(wave)
        hamiltonian = rank.R*(beta-np.eye(4))
        for component in range(3):
            hamiltonian += symbol[component]*alpha[component]
        propagator[index] = expm(-1j*gate.DT*hamiltonian)

    target = packet_current.copy()
    native_initial_center = dirac.circular_center(packet_current)
    target_initial_center = native_initial_center.copy()
    physical_initial, joint_initial = native_centers(q_current)
    steps = int(np.ceil(10*2*np.pi/rank.R/gate.DT))
    sample_steps = set(np.linspace(0, steps, 21, dtype=int).tolist())
    samples = []
    for step in range(steps+1):
        if step in sample_steps:
            coordinates = readout@q_current
            native_field = unpack_coordinates(coordinates)
            retained = raw@coordinates
            native_center = dirac.circular_center(native_field)
            target_center = dirac.circular_center(target)
            target_raw = raw@dirac.real_coordinates(target)
            physical_center, joint_center = native_centers(q_current)
            target_physical_center, target_joint_center = native_centers(target_raw)
            samples.append({
                "step": step,
                "cycles": step*gate.DT*rank.R/(2*np.pi),
                "native_center": native_center.tolist(),
                "target_center": target_center.tolist(),
                "native_wrapped_displacement": dirac.wrapped_shift(
                    native_initial_center, native_center).tolist(),
                "target_wrapped_displacement": dirac.wrapped_shift(
                    target_initial_center, target_center).tolist(),
                "native_Psi_density_center": physical_center.tolist(),
                "target_Psi_density_center": target_physical_center.tolist(),
                "native_Psi_density_displacement": dirac.wrapped_shift(
                    physical_initial, physical_center).tolist(),
                "target_Psi_density_displacement": dirac.wrapped_shift(
                    physical_initial, target_physical_center).tolist(),
                "native_joint_density_center": joint_center.tolist(),
                "target_joint_density_center": target_joint_center.tolist(),
                "native_joint_density_displacement": dirac.wrapped_shift(
                    joint_initial, joint_center).tolist(),
                "target_joint_density_displacement": dirac.wrapped_shift(
                    joint_initial, target_joint_center).tolist(),
                "readout_residual_fraction": float(
                    np.linalg.norm(q_current-retained)
                    / max(np.linalg.norm(q_current), 1e-30)),
                "readout_norm_ratio": float(
                    np.linalg.norm(native_field)
                    / np.linalg.norm(packet_current)),
            })
        if step == steps:
            break
        q_following = factor.solve(
            -(current@q_current+previous@q_previous))
        q_previous, q_current = q_current, q_following
        fourier = np.fft.fftn(target, axes=(0, 1, 2))
        fourier = np.einsum("...ab,...b->...a", propagator, fourier)
        target = np.fft.ifftn(fourier, axes=(0, 1, 2))
    result = {
        "status": "V4_WAKE_COMMON_VELOCITY_MOTION_DIAGNOSTIC",
        "momentum": momentum.tolist(),
        "samples": samples,
        "final": samples[-1],
        "verdict": "NATIVE_FOUR_COMPONENT_MOTION_EXTRACTED",
        "claim_boundary": (
            "Fixed readout diagnostic only. It does not refit the action or "
            "certify V4-006."),
    }
    path = HERE/OUTPUT
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
