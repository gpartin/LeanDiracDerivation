"""Decompose the first native/Dirac departure in the live Psi--chi frame."""

from __future__ import annotations

import json
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
CACHE = HERE/"wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz"


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def power(value):
    return float(np.sum(abs(value)**2))


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    rest_psi, _ = relative.unpack_raw(rest, dirac.SIDE)
    carrier = np.sum(abs(rest_psi)**2, axis=-1) > 1e-12
    shell = np.zeros_like(carrier)
    for offset, _weight in rank.native.DIRECTIONS:
        shell |= np.roll(carrier, shift=offset, axis=(0, 1, 2))
    shell &= ~carrier

    theta = float(np.arccos(1-gate.DT**2*rank.R**2/2))
    data = np.load(CACHE)
    previous = sparse(data["previous"]@gate.phase_rotation_matrix(rest.size, -theta))
    current = sparse(data["current"])
    following = sparse(data["following"]@gate.phase_rotation_matrix(rest.size, theta))
    factor = splu(following)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scale_row = next(row for row in prior["rows"] if row["flavor"] == "plus")
    scale_case = next(item for item in scale_row["cases"]
                      if item["envelope"] == "all_125_centers")
    scale = complex(scale_case["chi_scale_real"], scale_case["chi_scale_imag"])
    basis = dirac.flavor_bases()["plus"]
    alpha, beta = dirac.compressed_clifford(basis)
    raw = envelope.raw_columns(basis, psi_frame, chi_frame, sites,
                               dirac.OFFSETS, scale)
    pprevious, pcurrent, _pfollowing, _ = dirac.packet(
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

    for _step in range(2):
        qfollowing = factor.solve(-(current@qcurrent+previous@qprevious))
        qprevious, qcurrent = qcurrent, qfollowing
        fourier = np.fft.fftn(target, axes=(0, 1, 2))
        fourier = np.einsum("...ab,...b->...a", propagator, fourier)
        target = np.fft.ifftn(fourier, axes=(0, 1, 2))

    qtarget = raw@dirac.real_coordinates(target)
    psi, chi = relative.unpack_raw(qcurrent, dirac.SIDE)
    target_psi, target_chi = relative.unpack_raw(qtarget, dirac.SIDE)
    dpsi, dchi = psi-target_psi, chi-target_chi
    rho_chi = np.sum(abs(target_chi)**2, axis=-1)
    coefficient = np.sum(np.conj(target_chi)*dpsi, axis=-1)
    parallel = target_chi*coefficient[..., None]/np.maximum(
        rho_chi[..., None], np.finfo(float).eps*rank.R**2)
    transverse = dpsi-parallel

    ds_linear = (np.sum(np.conj(target_chi)*dpsi, axis=-1)
                 + np.sum(np.conj(dchi)*target_psi, axis=-1))
    dw_linear = (np.cross(target_chi, dpsi)
                 + np.cross(dchi, target_psi))
    s_actual = np.sum(np.conj(chi)*psi, axis=-1)
    s_target = np.sum(np.conj(target_chi)*target_psi, axis=-1)
    w_actual = np.cross(chi, psi)
    w_target = np.cross(target_chi, target_psi)

    result = {
        "status": "V4_LIVE_FRAME_FIRST_DEPARTURE_DECOMPOSITION",
        "step": 2,
        "Psi_error_power": power(dpsi),
        "Psi_parallel_to_chi_power": power(parallel),
        "Psi_transverse_to_chi_power": power(transverse),
        "Psi_parallel_fraction": power(parallel)/max(power(dpsi), 1e-300),
        "Psi_transverse_fraction": power(transverse)/max(power(dpsi), 1e-300),
        "linear_parallel_overlap_error_power": power(ds_linear),
        "linear_oriented_area_error_power": power(dw_linear),
        "nonlinear_parallel_overlap_error_power": power(s_actual-s_target),
        "nonlinear_oriented_area_error_power": power(w_actual-w_target),
        "carrier": {
            "Psi_parallel_power": power(parallel[carrier]),
            "Psi_transverse_power": power(transverse[carrier]),
            "parallel_overlap_error_power": power((s_actual-s_target)[carrier]),
            "oriented_area_error_power": power((w_actual-w_target)[carrier]),
        },
        "receiving_shell": {
            "Psi_parallel_power": power(parallel[shell]),
            "Psi_transverse_power": power(transverse[shell]),
            "parallel_overlap_error_power": power((s_actual-s_target)[shell]),
            "oriented_area_error_power": power((w_actual-w_target)[shell]),
        },
        "verdict": "FIRST_DEPARTURE_RESOLVED_IN_COMPLETE_LIVE_FRAME",
        "claim_boundary": "Diagnostic decomposition only; no term is fitted or installed.",
    }
    path = HERE/"live_frame_departure_decomposition.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
