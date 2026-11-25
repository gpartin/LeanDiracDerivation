"""Physical-site motion of balanced copies under the stable common action."""

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
import wake_common_velocity_motion_diagnostic as motion


HERE = Path(__file__).resolve().parent
CACHE = HERE/"wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz"


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


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
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    frequencies = 2*np.pi*np.fft.fftfreq(dirac.SIDE)
    copies = []
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, dirac.OFFSETS,
            scales[flavor])
        pprevious, pcurrent, _pfollowing, momentum = dirac.packet(
            0, 1, alpha, beta, gate.DT)
        propagator = np.empty((dirac.SIDE,)*3+(4, 4), dtype=complex)
        for index in np.ndindex((dirac.SIDE,)*3):
            wave = np.asarray([frequencies[value] for value in index])
            symbol = dirac.c19_first_symbol(wave)
            hamiltonian = rank.R*(beta-np.eye(4))
            for component in range(3):
                hamiltonian += symbol[component]*alpha[component]
            propagator[index] = expm(-1j*gate.DT*hamiltonian)
        copies.append({
            "flavor": flavor, "raw": raw, "previous": pprevious,
            "current": pcurrent, "target": pcurrent.copy(),
            "propagator": propagator, "momentum": momentum,
        })

    qprevious = np.column_stack([
        item["raw"]@dirac.real_coordinates(item["previous"])
        for item in copies])
    qcurrent = np.column_stack([
        item["raw"]@dirac.real_coordinates(item["current"])
        for item in copies])
    phases = (1, 1j, -1, -1j)
    weights = [np.asarray((1, phase), dtype=complex)/np.sqrt(2)
               for phase in phases]
    initial_centers = [motion.native_centers(qcurrent@value)
                       for value in weights]
    steps = int(np.ceil(10*2*np.pi/rank.R/gate.DT))
    for _step in range(steps):
        qfollowing = factor.solve(-(current@qcurrent+previous@qprevious))
        qprevious, qcurrent = qcurrent, qfollowing
        for item in copies:
            fourier = np.fft.fftn(item["target"], axes=(0, 1, 2))
            fourier = np.einsum(
                "...ab,...b->...a", item["propagator"], fourier)
            item["target"] = np.fft.ifftn(fourier, axes=(0, 1, 2))

    rows = []
    for phase, weight, centers0 in zip(phases, weights, initial_centers):
        native = qcurrent@weight
        target = sum(
            item["raw"]@dirac.real_coordinates(weight[index]*item["target"])
            for index, item in enumerate(copies))
        native_centers = motion.native_centers(native)
        target_centers = motion.native_centers(target)
        rows.append({
            "relative_phase": {"real": float(phase.real),
                               "imag": float(phase.imag)},
            "native_Psi_displacement": dirac.wrapped_shift(
                centers0[0], native_centers[0]).tolist(),
            "target_Psi_displacement": dirac.wrapped_shift(
                centers0[0], target_centers[0]).tolist(),
            "native_joint_displacement": dirac.wrapped_shift(
                centers0[1], native_centers[1]).tolist(),
            "target_joint_displacement": dirac.wrapped_shift(
                centers0[1], target_centers[1]).tolist(),
            "native_norm_ratio": float(
                np.linalg.norm(native)/np.linalg.norm(qprevious@weight)),
        })
    result = {
        "status": "V4_WAKE_COMMON_VELOCITY_TWO_COPY_MOTION",
        "cycles": 10,
        "rows": rows,
        "verdict": "TWO_COPY_PHYSICAL_MOTION_MEASURED",
        "claim_boundary": (
            "Linear tangent and site-density diagnostic. A favorable lateral "
            "cancellation must still pass exact pattern and nonlinear gates."),
    }
    path = HERE/"wake_common_velocity_two_copy_motion.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
