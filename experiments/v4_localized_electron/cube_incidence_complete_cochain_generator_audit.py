"""Resolve the current V4 tangent mismatch in the complete cubical basis.

The existing packet gate reports a small but nonzero one-step discrepancy.
This audit replaces the Gaussian packet by exact periodic Fourier modes,
compresses the native two-layer update through the *same* all-center cubical
embedding, and expands the residual in the complete Clifford basis.  Dirac is
an independent comparator only; it is not used in the native force.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, lu_factor, lu_solve

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as packet_gate
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent
SIDE = packet_gate.SIDE


def complex_coordinates(values: np.ndarray) -> np.ndarray:
    pairs = values.reshape(-1, 2)
    return (pairs[:, 0] + 1j*pairs[:, 1]).reshape(SIDE, SIDE, SIDE, 4)


def mode_field(momentum: np.ndarray, component: int) -> np.ndarray:
    coordinates = np.asarray(packet_gate.OFFSETS, dtype=float).reshape(
        SIDE, SIDE, SIDE, 3)
    phase = np.exp(1j*np.einsum("...a,a->...", coordinates, momentum))
    field = np.zeros((SIDE, SIDE, SIDE, 4), dtype=complex)
    field[..., component] = phase
    return field


def mode_amplitude(field: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(packet_gate.OFFSETS, dtype=float).reshape(
        SIDE, SIDE, SIDE, 3)
    phase = np.exp(-1j*np.einsum("...a,a->...", coordinates, momentum))
    return np.mean(field*phase[..., None], axis=(0, 1, 2))


def mode_leakage(field: np.ndarray, momentum: np.ndarray) -> float:
    amplitude = mode_amplitude(field, momentum)
    coordinates = np.asarray(packet_gate.OFFSETS, dtype=float).reshape(
        SIDE, SIDE, SIDE, 3)
    phase = np.exp(1j*np.einsum("...a,a->...", coordinates, momentum))
    retained = phase[..., None]*amplitude
    return float(np.linalg.norm(field-retained)
                 / max(np.linalg.norm(field), 1.0e-30))


def clifford_basis(alpha: list[np.ndarray], beta: np.ndarray):
    names = ("beta", "alpha_x", "alpha_y", "alpha_z")
    generators = (beta, *alpha)
    rows = []
    for selector in itertools.product((0, 1), repeat=4):
        matrix = np.eye(4, dtype=complex)
        label = []
        for use, name, generator in zip(selector, names, generators,
                                        strict=True):
            if use:
                matrix = matrix@generator
                label.append(name)
        rows.append(("identity" if not label else "*".join(label), matrix))
    return rows


def expansion(matrix: np.ndarray, basis) -> tuple[list[dict], float]:
    rows = []
    reconstructed = np.zeros_like(matrix)
    for name, element in basis:
        coefficient = (np.vdot(element, matrix)
                       / max(float(np.vdot(element, element).real), 1.0e-30))
        reconstructed += coefficient*element
        rows.append({
            "channel": name,
            "coefficient_real": float(coefficient.real),
            "coefficient_imag": float(coefficient.imag),
            "magnitude": float(abs(coefficient)),
        })
    rows.sort(key=lambda row: row["magnitude"], reverse=True)
    defect = float(np.linalg.norm(matrix-reconstructed)
                   / max(np.linalg.norm(matrix), 1.0e-30))
    return rows, defect


def main() -> None:
    bare, matrices = action.expected_blocks()
    mass, gyro, stiffness = (matrices[name]
                             for name in ("mass", "gyro", "stiffness"))
    dt = native_time.DT
    factor = lu_factor(mass/dt+gyro/2)
    advance = -lu_solve(factor, dt*stiffness-2*mass/dt)
    memory = -lu_solve(factor, mass/dt-gyro/2)
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    fundamental = 2*np.pi/SIDE
    momenta = {
        "rest": np.asarray((0.0, 0.0, 0.0)),
        "x": np.asarray((fundamental, 0.0, 0.0)),
        "y": np.asarray((0.0, fundamental, 0.0)),
        "z": np.asarray((0.0, 0.0, fundamental)),
        "diagonal": np.asarray((fundamental, fundamental, fundamental)),
    }
    flavor_rows = []
    for flavor, spin_basis in packet_gate.flavor_bases().items():
        alpha, beta = packet_gate.compressed_clifford(spin_basis)
        raw = envelope.raw_columns(
            spin_basis, psi_frame, chi_frame, sites,
            packet_gate.OFFSETS, scales[flavor])
        raw_inverse = np.linalg.pinv(raw, rcond=1.0e-11)
        clifford = clifford_basis(alpha, beta)
        cases = []
        map_records = {}
        for direction, momentum in momenta.items():
            transport = packet_gate.c19_first_symbol(momentum)
            hamiltonian = native_time.CHI0*(beta-np.eye(4))
            for axis in range(3):
                hamiltonian += transport[axis]*alpha[axis]
            target_map = expm(-1j*dt*hamiltonian)
            previous_map = expm(1j*dt*hamiltonian)
            native_map = np.zeros((4, 4), dtype=complex)
            leakages = []
            for component in range(4):
                current = mode_field(momentum, component)
                previous = np.einsum(
                    "ab,...b->...a", previous_map, current)
                q_current = raw@packet_gate.real_coordinates(current)
                q_previous = raw@packet_gate.real_coordinates(previous)
                q_native = advance@q_current+memory@q_previous
                recovered = complex_coordinates(raw_inverse@q_native)
                native_map[:, component] = mode_amplitude(
                    recovered, momentum)
                leakages.append(mode_leakage(recovered, momentum))
            residual = native_map-target_map
            channels, reconstruction_defect = expansion(residual, clifford)
            case = {
                "direction": direction,
                "momentum": momentum.tolist(),
                "c19_first_symbol": transport.tolist(),
                "native_vs_discrete_dirac_map_error": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(target_map), 1.0e-30)),
                "maximum_other_momentum_leakage": float(max(leakages)),
                "clifford_expansion_defect": reconstruction_defect,
                "leading_residual_channels": channels[:8],
            }
            cases.append(case)
            map_records[direction] = (native_map, target_map, case)
        native_rest, target_rest, _ = map_records["rest"]
        for direction, (native_map, target_map, case) in map_records.items():
            if direction == "rest":
                case["transport_increment_error_after_rest_subtraction"] = None
                continue
            native_increment = native_map-native_rest
            target_increment = target_map-target_rest
            increment_residual = native_increment-target_increment
            increment_channels, increment_defect = expansion(
                increment_residual, clifford)
            case["transport_increment_error_after_rest_subtraction"] = float(
                np.linalg.norm(increment_residual)
                / max(np.linalg.norm(target_increment), 1.0e-30))
            case["transport_increment_clifford_expansion_defect"] = (
                increment_defect)
            case["leading_transport_increment_residual_channels"] = (
                increment_channels[:8])
        flavor_rows.append({"flavor": flavor, "cases": cases})

    worst = max(case["native_vs_discrete_dirac_map_error"]
                for row in flavor_rows for case in row["cases"])
    dominant = {}
    for row in flavor_rows:
        for case in row["cases"]:
            name = case["leading_residual_channels"][0]["channel"]
            dominant[name] = dominant.get(name, 0) + 1
    worst_transport_increment = max(
        case["transport_increment_error_after_rest_subtraction"]
        for row in flavor_rows for case in row["cases"]
        if case["direction"] != "rest")
    result = {
        "status": "V4_COMPLETE_COCHAIN_NATIVE_GENERATOR_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "embedding": (
            "complete eight-corner cubical cochains, separated into the two "
            "native rank-four flavor copies and lifted over all 125 centers"),
        "comparator": (
            "independent exact-C19 centered-time Dirac one-step map in the "
            "carrier co-rotating frame"),
        "rows": flavor_rows,
        "worst_native_vs_discrete_dirac_map_error": float(worst),
        "worst_transport_increment_error_after_rest_subtraction": float(
            worst_transport_increment),
        "dominant_channel_counts": dominant,
        "verdict": (
            "CURRENT_ACTION_COMPLETE_COCHAIN_GENERATOR_PASS"
            if worst < 1.0e-3 else
            "CURRENT_ACTION_COMPLETE_COCHAIN_GENERATOR_MISMATCH"),
        "claim_boundary": (
            "Linear one-step Fourier response of the current research action. "
            "No action coefficient is fitted.  This is not nonlinear mobile "
            "carrier closure or V4-006 acceptance."),
    }
    output = HERE/"cube_incidence_complete_cochain_generator_audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
