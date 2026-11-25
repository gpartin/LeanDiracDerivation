"""Classify the continuum-persistent one-cycle transport defect."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as box
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope
import localized_stability as native


HERE = Path(__file__).resolve().parent


def sparse(value):
    result = value.copy()
    result[abs(result) < 1e-13] = 0
    return csc_matrix(result)


def pack(psi, chi):
    return box.pack(psi, chi)


def main():
    bare, _ = action.expected_blocks()
    mass0, gyro0, stiffness0 = ablation.blocks()["complete_research_action"]
    mass, gyro, stiffness = map(sparse, (mass0, gyro0, stiffness0))
    dt = native_time.DT/16
    lhs = (mass/dt+gyro/2).tocsc()
    cb = (dt*stiffness-2*mass/dt).tocsc()
    pb = (mass/dt-gyro/2).tocsc()
    factor = splu(lhs)
    steps = int(np.ceil(2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row = next(row for row in prior["rows"] if row["flavor"] == "plus")
    case = next(item for item in row["cases"]
                if item["envelope"] == "all_125_centers")
    chi_scale = complex(case["chi_scale_real"], case["chi_scale_imag"])
    basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, chi_scale)
    previous, initial, _following, _ = gate.packet(0, 1, alpha, beta, dt)
    q_previous = raw@gate.real_coordinates(previous)
    q_current = raw@gate.real_coordinates(initial)
    for _ in range(steps):
        following = factor.solve(-(cb@q_current+pb@q_previous))
        q_previous, q_current = q_current, following
    target = gate.dirac_step(initial, alpha, beta, 1, steps*dt)
    target_raw = raw@gate.real_coordinates(target)
    difference = q_current-target_raw
    dpsi, dchi = native.unpack(difference, 5)
    rest_psi, rest_chi = native.unpack(bare["rest"], 5)
    carrier = native.norm2(rest_psi) > 1e-12
    field_power = native.norm2(dpsi)+native.norm2(dchi)
    total = float(np.sum(field_power))
    psi_components = np.sum(abs(dpsi)**2, axis=(0, 1, 2))
    chi_components = np.sum(abs(dchi)**2, axis=(0, 1, 2))
    symmetry = np.column_stack((
        pack(1j*rest_psi, np.zeros_like(rest_chi)),
        pack(np.zeros_like(rest_psi), 1j*rest_chi),
        native.translation_vectors(rest_psi, rest_chi),
    ))
    qsym = envelope.orth(symmetry)
    symmetry_fraction = float(
        np.linalg.norm(qsym.T@difference)**2
        / max(np.linalg.norm(difference)**2, 1e-30))
    result = {
        "status": "V4_CUBE_INCIDENCE_LONG_TANGENT_RCA",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "dt": dt,
        "one_cycle_steps": steps,
        "relative_defect": float(
            np.linalg.norm(difference)/np.linalg.norm(target_raw)),
        "defect_power_fractions": {
            "Psi": float(np.sum(abs(dpsi)**2)/total),
            "chi": float(np.sum(abs(dchi)**2)/total),
            "carrier_sites": float(np.sum(field_power[carrier])/total),
            "first_component": float(
                (psi_components[0]+chi_components[0])/total),
            "second_component": float(
                (psi_components[1]+chi_components[1])/total),
            "third_component": float(
                (psi_components[2]+chi_components[2])/total),
            "phase_and_translation_symmetry_subspace": symmetry_fraction,
        },
        "verdict": "LONG_TANGENT_DEFECT_CLASSIFIED",
        "claim_boundary": (
            "Channel classification of one failing case. It is not a no-go "
            "for all reciprocal action changes."),
    }
    (HERE/"cube_incidence_long_tangent_rca.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
