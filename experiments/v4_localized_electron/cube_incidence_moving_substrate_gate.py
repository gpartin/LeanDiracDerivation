"""Test whether a freely supplied substrate drift repairs V4-006.

Two distinct questions are kept separate.

1. Is a simple co-moving translation/phase velocity of the exact localized
   rest orbit itself a native solution of the nonlinear action?
2. Can adding such a drift at tangent order change the packet defect?

The second answer must be no for a linear tangent system unless the drift
changes the background Hessian.  We verify that numerically instead of
silently treating a coordinate boost as a new force.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_noncoplanar_box_transfer as native_box
import canonical_transfer_oscillator_spectrum as native_spectrum
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


def advance_factory(dt):
    mass, gyro, stiffness = map(
        sparse, ablation.blocks()["complete_research_action"])
    lhs = (mass/dt+gyro/2).tocsc()
    current = (dt*stiffness-2*mass/dt).tocsc()
    previous = (mass/dt-gyro/2).tocsc()
    factor = splu(lhs)

    def advance(q_previous, q_current):
        return factor.solve(-(current@q_current+previous@q_previous))

    return advance


def main():
    bare, _ = action.expected_blocks()
    rest = bare["rest"]
    psi, chi = native.unpack(rest, 5)
    zero_psi = np.zeros_like(psi)
    zero_chi = np.zeros_like(chi)
    directions = {
        "Psi_phase": native.pack(1j*psi, zero_chi),
        "chi_phase": native.pack(zero_psi, 1j*chi),
    }
    translations = native.translation_vectors(psi, chi)
    for axis, name in enumerate("xyz"):
        directions[f"translation_{name}"] = translations[:, axis]
    directions = {
        name: value/max(np.linalg.norm(value), 1e-30)
        for name, value in directions.items()
    }

    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    rest_layers = (rotation.T@rest, rest, rotation@rest)
    nonlinear_rows = []
    for name, direction in directions.items():
        for speed in (-0.05, -0.01, 0.01, 0.05):
            layers = (
                rest_layers[0]-native_time.DT*speed*(rotation.T@direction),
                rest_layers[1],
                rest_layers[2]+native_time.DT*speed*(rotation@direction),
            )
            tensors = tuple(torch.tensor(
                value, dtype=torch.float64, requires_grad=True)
                for value in layers)
            residual = action.del_residual(*tensors).detach().numpy()
            nonlinear_rows.append({
                "drift_channel": name,
                "signed_speed": speed,
                "DEL_norm": float(np.linalg.norm(residual)),
                "DEL_norm_per_abs_speed": float(
                    np.linalg.norm(residual)/abs(speed)),
            })

    dt = native_time.DT/16
    advance = advance_factory(dt)
    steps = int(np.ceil(10*2*np.pi/native_time.CHI0/dt))
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    row = next(row for row in prior["rows"] if row["flavor"] == "plus")
    case = next(item for item in row["cases"]
                if item["envelope"] == "all_125_centers")
    chi_scale = complex(case["chi_scale_real"], case["chi_scale_imag"])
    basis = gate.flavor_bases()["plus"]
    alpha, beta = gate.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, gate.OFFSETS, chi_scale)
    previous, current, _following, _ = gate.packet(0, 1, alpha, beta, dt)
    packet_previous = raw@gate.real_coordinates(previous)
    packet_current = raw@gate.real_coordinates(current)

    superposition_rows = []
    for name, direction in directions.items():
        # This is a finite tangent velocity, not a new nonlinear background.
        background_previous = -dt*direction
        background_current = np.zeros_like(direction)
        combined_previous = packet_previous+background_previous
        combined_current = packet_current+background_current
        packet_p, packet_c = packet_previous.copy(), packet_current.copy()
        background_p, background_c = (
            background_previous.copy(), background_current.copy())
        combined_p, combined_c = combined_previous, combined_current
        maximum = 0.0
        for _ in range(steps):
            packet_f = advance(packet_p, packet_c)
            background_f = advance(background_p, background_c)
            combined_f = advance(combined_p, combined_c)
            maximum = max(maximum, float(np.linalg.norm(
                (combined_f-background_f)-packet_f)
                / max(np.linalg.norm(packet_f), 1e-30)))
            packet_p, packet_c = packet_c, packet_f
            background_p, background_c = background_c, background_f
            combined_p, combined_c = combined_c, combined_f
        superposition_rows.append({
            "drift_channel": name,
            "maximum_packet_change_after_subtracting_drift": maximum,
        })

    translation_rows = [row for row in nonlinear_rows
                        if row["drift_channel"].startswith("translation")]
    minimum_translation_residual = min(
        row["DEL_norm"] for row in translation_rows)
    maximum_superposition_defect = max(
        row["maximum_packet_change_after_subtracting_drift"]
        for row in superposition_rows)
    result = {
        "status": "V4_CUBE_INCIDENCE_MOVING_SUBSTRATE_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "nonlinear_drift_rows": nonlinear_rows,
        "linear_superposition_rows": superposition_rows,
        "minimum_simple_translation_DEL_norm": minimum_translation_residual,
        "maximum_linear_background_effect_on_packet":
            maximum_superposition_defect,
        "verdict": (
            "SIMPLE_MOVING_SUBSTRATE_DOES_NOT_REPAIR_V4_006"
            if minimum_translation_residual > 1e-8
            and maximum_superposition_defect < 1e-9 else
            "MOVING_SUBSTRATE_REQUIRES_FURTHER_CLASSIFICATION"),
        "root_cause": (
            "A moving coordinate lattice is not a new native field state. "
            "The tested translation and phase velocities are not exact "
            "nonlinear rest-orbit continuations, while a tangent background "
            "superposes independently and therefore cannot change the packet "
            "sector defect. A finite-amplitude self-consistent running field "
            "branch would be a distinct background and must first be solved "
            "from the action."),
        "claim_boundary": (
            "This rejects simple boosted-rest and linear drift repairs. It is "
            "not a no-go for every finite-amplitude running Psi-chi branch."),
    }
    (HERE/"cube_incidence_moving_substrate_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
