"""Locate the first Psi/chi/component departure from the C19-Dirac packet."""

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
    "LFM_DEPARTURE_OUTPUT", "transverse_frame_first_departure.json")


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def field_power(value: np.ndarray) -> float:
    return float(np.sum(abs(value)**2))


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    rest_psi, _rest_chi = relative.unpack_raw(rest, dirac.SIDE)
    carrier = np.sum(abs(rest_psi)**2, axis=-1) > 1e-12
    shell = np.zeros_like(carrier)
    for offset, _weight in rank.native.DIRECTIONS:
        shell |= np.roll(carrier, shift=offset, axis=(0, 1, 2))
    shell &= ~carrier

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
    pprevious, pcurrent, _pfollowing, momentum = dirac.packet(
        0, 1, alpha, beta, gate.DT)
    qprevious = raw@dirac.real_coordinates(pprevious)
    qcurrent = raw@dirac.real_coordinates(pcurrent)
    target = pcurrent.copy()

    frequencies = 2*np.pi*np.fft.fftfreq(dirac.SIDE)
    propagator = np.empty((dirac.SIDE,)*3+(4, 4), dtype=complex)
    for index in np.ndindex((dirac.SIDE,)*3):
        wave = np.asarray([frequencies[value] for value in index])
        symbol = dirac.c19_first_symbol(wave)
        hamiltonian = rank.R*(beta-np.eye(4))
        for component in range(3):
            hamiltonian += symbol[component]*alpha[component]
        propagator[index] = expm(-1j*gate.DT*hamiltonian)

    rows = []
    first_threshold = None
    for step in range(65):
        qtarget = raw@dirac.real_coordinates(target)
        error = qcurrent-qtarget
        psi_error, chi_error = relative.unpack_raw(error, dirac.SIDE)
        psi_native, chi_native = relative.unpack_raw(qcurrent, dirac.SIDE)
        total_reference = max(np.linalg.norm(qtarget), 1e-30)
        row = {
            "step": step,
            "cycles": step*gate.DT*rank.R/(2*np.pi),
            "relative_total_error": float(np.linalg.norm(error)/total_reference),
            "Psi_error_power_fraction": field_power(psi_error)/max(
                field_power(psi_error)+field_power(chi_error), 1e-300),
            "chi_error_power_fraction": field_power(chi_error)/max(
                field_power(psi_error)+field_power(chi_error), 1e-300),
            "Psi_component_error_power": [field_power(psi_error[..., a])
                                           for a in range(3)],
            "chi_component_error_power": [field_power(chi_error[..., a])
                                           for a in range(3)],
            "Psi_carrier_error_power": field_power(psi_error[carrier]),
            "Psi_shell_error_power": field_power(psi_error[shell]),
            "chi_carrier_error_power": field_power(chi_error[carrier]),
            "chi_shell_error_power": field_power(chi_error[shell]),
            "Psi_native_power": field_power(psi_native),
            "chi_native_power": field_power(chi_native),
        }
        rows.append(row)
        if first_threshold is None and row["relative_total_error"] > 1e-3:
            first_threshold = row
        if step == 64:
            break
        qfollowing = factor.solve(-(current@qcurrent+previous@qprevious))
        qprevious, qcurrent = qcurrent, qfollowing
        fourier = np.fft.fftn(target, axes=(0, 1, 2))
        fourier = np.einsum("...ab,...b->...a", propagator, fourier)
        target = np.fft.ifftn(fourier, axes=(0, 1, 2))
    result = {
        "status": "V4_TRANSVERSE_FRAME_FIRST_DEPARTURE",
        "momentum": momentum.tolist(),
        "threshold": 1e-3,
        "first_threshold_crossing": first_threshold,
        "rows": rows,
        "verdict": "FIRST_NATIVE_DIRAC_DEPARTURE_CLASSIFIED",
        "claim_boundary": (
            "Linear tangent diagnostic; it identifies an inverse response "
            "requirement and never enters the action."),
    }
    path = HERE/OUTPUT
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "first_threshold_crossing": first_threshold,
        "step_64": rows[-1],
    }, indent=2))


if __name__ == "__main__":
    main()
