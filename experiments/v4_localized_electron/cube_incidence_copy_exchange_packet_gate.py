"""Test whether the second cubical copy supplies the C19 edge channel.

C19 has two spatial bond orbits.  Face steps reverse checkerboard parity,
whereas edge steps preserve it.  Since every cubical gamma reverses beta,
the same one-copy staggering cannot be invariant under both orbits.

The complete eight-corner algebra has a geometry-native copy-exchange
involution X which commutes with beta and every gamma and anticommutes with
the historical flavor F.  This script tests the unique orbit-resolved
first-order comparator

  H(k) = chi0 (beta-I)
       + sum_a [d_face_a(k) gamma_a + d_edge_a(k) X gamma_a].

No such term is installed in the native action.  The calculation only asks
whether the current native Psi--chi tangent already follows this overlooked
two-copy readout more closely than the one-copy comparator.
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
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import cubical_edge_shift_channels as cubical
import current_action_localized_envelope_closure as envelope
import localized_stability as c19


HERE = Path(__file__).resolve().parent


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def copy_exchange() -> np.ndarray:
    """Sparse commutant involution selected by cube coordinate ordering."""
    result = np.zeros((8, 8), dtype=complex)
    # X maps (x,y,z) to (1-x,y,1-z).  The signs are the corresponding
    # oriented cubical incidence signs.  This exact sparse representative is
    # one axis of the two-real-dimensional flavor-exchange plane.
    entries = {
        (0, 5): -1j, (5, 0): 1j,
        (1, 4): 1j, (4, 1): -1j,
        (2, 7): 1j, (7, 2): -1j,
        (3, 6): -1j, (6, 3): 1j,
    }
    for index, value in entries.items():
        result[index] = value
    return result


def orbit_symbols(momentum):
    face = np.zeros(3)
    edge = np.zeros(3)
    for offset, weight in c19.DIRECTIONS:
        direction = np.asarray(offset, dtype=float)
        contribution = weight * direction * np.sin(momentum @ direction) / 12
        if np.sum(np.abs(direction)) == 1:
            face += contribution
        else:
            edge += contribution
    return face, edge


def hamiltonian(momentum, exchange_eigenvalue=None):
    identity = np.eye(8, dtype=complex)
    beta = cubical.beta()
    exchange = copy_exchange()
    face, edge = orbit_symbols(momentum)
    result = native_time.CHI0 * (beta - identity)
    for axis in range(3):
        gamma = cubical.gamma(axis)
        result += face[axis] * gamma + edge[axis] * exchange @ gamma
    return result


def dirac_step(field, sign, dt):
    fourier = np.fft.fftn(field, axes=(0, 1, 2))
    result = np.empty_like(fourier)
    frequencies = 2 * np.pi * np.fft.fftfreq(gate.SIDE)
    for index in itertools.product(range(gate.SIDE), repeat=3):
        momentum = np.asarray([frequencies[value] for value in index])
        result[index] = expm(
            -1j * sign * dt * hamiltonian(momentum)) @ fourier[index]
    return np.fft.ifftn(result, axes=(0, 1, 2))


def packet(axis, sign, exchange_sign, dt, width=1.35,
           carrier_wave_number=0.35):
    coordinates = np.asarray(gate.OFFSETS, dtype=float).reshape(
        gate.SIDE, gate.SIDE, gate.SIDE, 3)
    distance2 = np.sum(coordinates ** 2, axis=-1)
    momentum = np.zeros(3)
    if axis == 3:
        momentum[:] = sign * carrier_wave_number / np.sqrt(3)
    else:
        momentum[axis] = sign * carrier_wave_number
    scalar = np.exp(-distance2 / (2 * width ** 2)) * np.exp(
        1j * np.einsum("...a,a->...", coordinates, momentum))
    h0 = hamiltonian(momentum)
    values, vectors = np.linalg.eigh(h0)
    exchange = copy_exchange()
    # Resolve the positive-energy degeneracy by diagonalizing X inside it.
    selected = vectors[:, values > np.max(values) - 1e-8]
    xsmall = selected.conj().T @ exchange @ selected
    xvalues, xvectors = np.linalg.eigh(xsmall)
    index = int(np.argmax(xvalues) if exchange_sign > 0 else np.argmin(xvalues))
    spinor = selected @ xvectors[:, index]
    current = scalar[..., None] * spinor
    return (dirac_step(current, -1, dt), current,
            dirac_step(current, 1, dt), momentum)


def coordinates(field8, basis):
    return np.einsum("ac,...a->...c", basis.conj(), field8)


def native_lift(field8, raw_plus, raw_minus, bases):
    plus = coordinates(field8, bases["plus"])
    minus = coordinates(field8, bases["minus"])
    return (raw_plus @ gate.real_coordinates(plus)
            + raw_minus @ gate.real_coordinates(minus))


def main() -> None:
    exchange = copy_exchange()
    identity = np.eye(8, dtype=complex)
    algebra = {
        "square_defect": float(np.linalg.norm(exchange @ exchange - identity)),
        "hermiticity_defect": float(np.linalg.norm(
            exchange.conj().T - exchange)),
        "flavor_anticommutator_defect": float(np.linalg.norm(
            exchange @ cubical.flavor() + cubical.flavor() @ exchange)),
        "beta_commutator_defect": float(np.linalg.norm(
            exchange @ cubical.beta() - cubical.beta() @ exchange)),
        "gamma_commutator_defects": [float(np.linalg.norm(
            exchange @ cubical.gamma(axis)
            - cubical.gamma(axis) @ exchange)) for axis in range(3)],
    }

    bare, _ = action.expected_blocks()
    mass0, gyro0, stiffness0 = ablation.blocks()["complete_research_action"]
    mass, gyro, stiffness = map(sparse, (mass0, gyro0, stiffness0))
    dt = native_time.DT / 16
    lhs = (mass / dt + gyro / 2).tocsc()
    current_block = (dt * stiffness - 2 * mass / dt).tocsc()
    previous_block = (mass / dt - gyro / 2).tocsc()
    factor = splu(lhs)
    steps = int(np.ceil(2 * np.pi / native_time.CHI0 / dt))

    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    bases = gate.flavor_bases()
    raw = {
        flavor: envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS,
            scales[flavor])
        for flavor, basis in bases.items()
    }

    rows = []
    for exchange_sign in (-1, 1):
        for axis, sign in ((0, -1), (0, 1), (3, 1)):
            previous, current, _following, momentum = packet(
                axis, sign, exchange_sign, dt)
            q_previous = native_lift(
                previous, raw["plus"], raw["minus"], bases)
            q_current = native_lift(
                current, raw["plus"], raw["minus"], bases)
            one_target = native_lift(
                dirac_step(current, 1, dt), raw["plus"], raw["minus"], bases)
            one_native = factor.solve(
                -(current_block @ q_current + previous_block @ q_previous))
            for _ in range(steps):
                following = factor.solve(
                    -(current_block @ q_current + previous_block @ q_previous))
                q_previous, q_current = q_current, following
            target_field = dirac_step(current, 1, steps * dt)
            target = native_lift(
                target_field, raw["plus"], raw["minus"], bases)
            target_i = native_lift(
                1j * target_field, raw["plus"], raw["minus"], bases)
            phase_basis = np.column_stack((target, target_i))
            coefficient, *_ = np.linalg.lstsq(
                phase_basis, q_current, rcond=1e-12)
            aligned = phase_basis @ (coefficient / max(
                np.linalg.norm(coefficient), 1e-300))
            face, edge = orbit_symbols(momentum)
            rows.append({
                "exchange_eigenvalue": exchange_sign,
                "direction": "diagonal" if axis == 3 else "x",
                "sign": sign,
                "momentum": momentum.tolist(),
                "face_symbol": face.tolist(),
                "edge_symbol": edge.tolist(),
                "one_step_native_vs_comparator_relative_error": float(
                    np.linalg.norm(one_native - one_target)
                    / max(np.linalg.norm(one_target), 1e-300)),
                "one_cycle_native_vs_comparator_relative_error": float(
                    np.linalg.norm(q_current - target)
                    / max(np.linalg.norm(target), 1e-300)),
                "one_cycle_phase_aligned_shape_error": float(
                    np.linalg.norm(q_current - aligned)
                    / max(np.linalg.norm(target), 1e-300)),
                "one_cycle_norm_ratio": float(
                    np.linalg.norm(q_current)
                    / max(np.linalg.norm(target), 1e-300)),
            })

    best = min(rows, key=lambda row: row[
        "one_cycle_phase_aligned_shape_error"])
    result = {
        "status": "V4_C19_COPY_EXCHANGE_PACKET_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "copy_exchange_algebra": algebra,
        "comparator": (
            "face orbit uses gamma_a; edge orbit uses X gamma_a; X is a "
            "cube-derived flavor-exchange involution"),
        "rows": rows,
        "best_case": best,
        "verdict": (
            "COPY_EXCHANGE_COMPARATOR_MATERIAL_IMPROVEMENT"
            if best["one_cycle_phase_aligned_shape_error"] < 0.27 else
            "COPY_EXCHANGE_DOES_NOT_REPAIR_LONG_PACKET"),
        "claim_boundary": (
            "Diagnostic one-cycle linear tangent comparison. The modified "
            "comparator has not been derived from the native action and is "
            "not a Dirac or movement pass. No coefficient or equation is "
            "changed."),
    }
    (HERE / "cube_incidence_copy_exchange_packet_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "copy_exchange_algebra": algebra,
        "best_case": best,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
