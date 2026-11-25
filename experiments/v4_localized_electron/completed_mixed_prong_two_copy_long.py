"""Long tangent test of balanced two-copy Dirac packets.

The two native Clifford copies are evolved by the same unrestricted tangent
recurrence.  Relative copy phase is preparation data only; it never enters
the action or update.
"""

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


HERE = Path(__file__).resolve().parent
CACHE = HERE / "completed_mixed_prong_tangent_blocks.npz"


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

    copy_data = []
    frequencies = 2*np.pi*np.fft.fftfreq(dirac.SIDE)
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, dirac.OFFSETS,
            scales[flavor])
        packet_previous, packet_initial, _, momentum = dirac.packet(
            0, 1, alpha, beta, gate.DT)
        propagator = np.empty((dirac.SIDE, dirac.SIDE, dirac.SIDE, 4, 4),
                              dtype=complex)
        for index in np.ndindex((dirac.SIDE,)*3):
            wave = np.asarray([frequencies[value] for value in index])
            transport = dirac.c19_first_symbol(wave)
            hamiltonian = rank.R*(beta-np.eye(4))
            for component in range(3):
                hamiltonian += transport[component]*alpha[component]
            propagator[index] = expm(-1j*gate.DT*hamiltonian)
        copy_data.append((flavor, raw, alpha, beta,
                          packet_previous, packet_initial, momentum,
                          propagator))

    steps = int(np.ceil(10*2*np.pi/rank.R/gate.DT))
    sample_steps = set(np.linspace(0, steps, 101, dtype=int).tolist())
    phases = (1, 1j, -1, -1j)
    weights_by_phase = [
        (1/np.sqrt(2), phase/np.sqrt(2)) for phase in phases]
    q_previous_copies = np.column_stack([
        item[1]@dirac.real_coordinates(item[4]) for item in copy_data])
    q_current_copies = np.column_stack([
        item[1]@dirac.real_coordinates(item[5]) for item in copy_data])
    target_fields = [item[5].copy() for item in copy_data]
    samples_by_phase = [[] for _ in phases]
    initial_norms = [np.linalg.norm(q_current_copies@np.asarray(weights))
                     for weights in weights_by_phase]
    lift = envelope.orth(np.column_stack([item[1] for item in copy_data]))
    for step in range(steps+1):
        if step in sample_steps:
            for selected, weights in enumerate(weights_by_phase):
                q_current = q_current_copies@np.asarray(weights)
                target = np.zeros(rest.size)
                target_i = np.zeros(rest.size)
                for weight, item, field in zip(weights, copy_data,
                                               target_fields):
                    raw = item[1]
                    target += raw@dirac.real_coordinates(weight*field)
                    target_i += raw@dirac.real_coordinates(1j*weight*field)
                retained = lift@(lift.T@q_current)
                phase_basis = np.column_stack((target, target_i))
                coefficients, *_ = np.linalg.lstsq(
                    phase_basis, q_current, rcond=1e-12)
                length = np.linalg.norm(coefficients)
                aligned = phase_basis@(coefficients/max(length, 1e-30))
                samples_by_phase[selected].append({
                    "step": step,
                    "cycles": step*gate.DT*rank.R/(2*np.pi),
                    "phase_aligned_shape_error": float(
                        np.linalg.norm(q_current-aligned)
                        / max(np.linalg.norm(target), 1e-30)),
                    "outside_two_copy_lift_fraction": float(
                        np.linalg.norm(q_current-retained)
                        / max(np.linalg.norm(q_current), 1e-30)),
                    "norm_ratio": float(np.linalg.norm(q_current)
                                        / max(initial_norms[selected], 1e-30)),
                })
        if step == steps:
            break
        q_following_copies = factor.solve(
            -(current@q_current_copies+previous@q_previous_copies))
        q_previous_copies, q_current_copies = (
            q_current_copies, q_following_copies)
        for index, item in enumerate(copy_data):
            fourier = np.fft.fftn(target_fields[index], axes=(0, 1, 2))
            fourier = np.einsum("...ab,...b->...a", item[7], fourier)
            target_fields[index] = np.fft.ifftn(
                fourier, axes=(0, 1, 2))
    rows = []
    for phase, samples in zip(phases, samples_by_phase):
        rows.append({
            "relative_phase": {"real": float(phase.real),
                               "imag": float(phase.imag)},
            "maximum_phase_aligned_shape_error": max(
                x["phase_aligned_shape_error"] for x in samples),
            "maximum_outside_two_copy_lift_fraction": max(
                x["outside_two_copy_lift_fraction"] for x in samples),
            "maximum_norm_drift": max(
                abs(x["norm_ratio"]-1) for x in samples),
            "samples": samples,
        })
    result = {
        "status": "V4_COMPLETED_MIXED_PRONG_TWO_COPY_LONG",
        "cycles": 10,
        "rows": rows,
        "best_row": min(rows, key=lambda x: x[
            "maximum_phase_aligned_shape_error"]),
        "verdict": "TWO_COPY_LONG_DIAGNOSTIC_COMPLETE",
        "claim_boundary": (
            "Linear tangent preparation diagnostic only; the relative copy "
            "phase is initial data, not an action term or selection rule."),
    }
    path = HERE/"completed_mixed_prong_two_copy_long.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rows": [{key: value for key, value in row.items()
                  if key != "samples"} for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
