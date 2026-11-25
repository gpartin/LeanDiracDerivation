"""Lossless phase-space test for the C19 Hodge four-field.

The configuration differential of F=(tau,W) has rank 500 for a 1000-real-
dimensional four-component field.  A second-order native theory must use the
current and previous layers.  This script supplies the previous layer from the
independent discrete C19 Dirac comparator and tests the stacked native map

    spinor -> (delta F_n, (delta F_n-delta F_(n-1))/dt).

The comparator is used only to specify the tested positive-frequency tangent;
it never enters the native action or forces.
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as gate
import cube_incidence_hodge_readout_bridge as readout
import personal_touch_unit_geometric_canonical as canonical


HERE = Path(__file__).resolve().parent
SIDE = 5
COMPLEX_DIMENSION = SIDE**3*4
REAL_DIMENSION = 2*COMPLEX_DIMENSION


def complex_to_real_columns(matrix: np.ndarray):
    return np.stack((matrix.real, matrix.imag), axis=1).reshape(
        2*matrix.shape[0], matrix.shape[1])


def previous_real_operator(copy_name: str):
    basis = gate.flavor_bases()[copy_name]
    alpha, beta = gate.compressed_clifford(basis)
    fields = np.eye(COMPLEX_DIMENSION, dtype=complex).reshape(
        SIDE, SIDE, SIDE, 4, COMPLEX_DIMENSION)
    fourier = np.fft.fftn(fields, axes=(0, 1, 2))
    transformed = np.empty_like(fourier)
    frequencies = 2*np.pi*np.fft.fftfreq(SIDE)
    for index in itertools.product(range(SIDE), repeat=3):
        momentum = np.asarray([frequencies[value] for value in index])
        transport = gate.c19_first_symbol(momentum)
        hamiltonian = native_time.CHI0*(beta-np.eye(4))
        for axis in range(3):
            hamiltonian = hamiltonian+transport[axis]*alpha[axis]
        propagator = expm(1j*native_time.DT*hamiltonian)
        transformed[index] = np.einsum(
            "ab,bn->an", propagator, fourier[index])
    complex_operator = np.fft.ifftn(
        transformed, axes=(0, 1, 2)).reshape(
            COMPLEX_DIMENSION, COMPLEX_DIMENSION)
    real_operator = np.empty((REAL_DIMENSION, REAL_DIMENSION))
    real_operator[:, 0::2] = complex_to_real_columns(complex_operator)
    real_operator[:, 1::2] = complex_to_real_columns(1j*complex_operator)
    return real_operator


def complex_structure(dimension: int):
    result = np.zeros((dimension, dimension))
    result[0::2, 1::2] = -np.eye(dimension//2)
    result[1::2, 0::2] = np.eye(dimension//2)
    return result


def analyze(copy_name: str, rest: np.ndarray, raw: np.ndarray,
            rotation: np.ndarray, theta: float):
    previous_coordinates = previous_real_operator(copy_name)
    previous_raw = raw@previous_coordinates
    current_f = readout.composite_jacobian(rest, raw, activated=False)
    previous_f = readout.composite_jacobian(
        rotation.T@rest, previous_raw, activated=False)
    field_complex_structure = complex_structure(current_f.shape[0])
    # F=(chi.conj(Psi),chi x conj(Psi))/chi0 carries phase exp(-i theta)
    # when Psi carries exp(+i theta).  The n-1 layer therefore carries
    # exp(+i theta) relative to n and is aligned by exp(-i theta).
    previous_aligned = ((np.cos(theta)*np.eye(current_f.shape[0])
                         - np.sin(theta)*field_complex_structure)@previous_f)
    velocity_f = (current_f-previous_aligned)/native_time.DT
    phase = np.vstack((current_f, velocity_f))
    u, singular, _vh = np.linalg.svd(phase, full_matrices=False)
    threshold = 1e-10*max(float(singular[0]), 1e-30)
    rank = int(np.sum(singular > threshold))
    q = u[:, :rank]
    hodge_image = np.vstack((
        readout.hodge_on_columns(current_f),
        readout.hodge_on_columns(velocity_f)))
    leakage = hodge_image-q@(q.T@hodge_image)
    configuration_rank = int(np.linalg.matrix_rank(
        current_f, tol=1e-10*np.linalg.norm(current_f, 2)))
    # Multiplication by i in the interleaved (real,imag) representation.
    positive = (current_f
                + field_complex_structure@velocity_f/native_time.CHI0)
    pu, psingular, _pvh = np.linalg.svd(positive, full_matrices=False)
    pthreshold = 1e-10*max(float(psingular[0]), 1e-30)
    prank = int(np.sum(psingular > pthreshold))
    pq = pu[:, :prank]
    positive_hodge = readout.hodge_on_columns(positive)
    positive_leakage = positive_hodge-pq@(pq.T@positive_hodge)
    return {
        "copy": copy_name,
        "input_real_dimension": REAL_DIMENSION,
        "configuration_rank": configuration_rank,
        "previous_layer_aligned_to_current_carrier_frame": True,
        "phase_space_rank": rank,
        "phase_space_condition_number": float(
            singular[0]/singular[rank-1]) if rank else None,
        "smallest_retained_singular": float(singular[rank-1]) if rank else 0,
        "hodge_phase_space_cross_block": float(
            np.linalg.norm(leakage)/max(np.linalg.norm(hodge_image), 1e-30)),
        "positive_frequency_rank": prank,
        "positive_frequency_condition_number": float(
            psingular[0]/psingular[prank-1]) if prank else None,
        "positive_frequency_smallest_retained_singular": float(
            psingular[prank-1]) if prank else 0,
        "positive_frequency_hodge_cross_block": float(
            np.linalg.norm(positive_leakage)
            / max(np.linalg.norm(positive_hodge), 1e-30)),
        "basis": q,
    }


def main() -> None:
    rest = np.load(canonical.CACHE)["rest"]
    theta = float(np.arccos(
        1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    historical = np.load(HERE/"cube_incidence_dirac_envelope_gate.npz")
    rows = {name: analyze(name, rest, historical[name], rotation, theta)
            for name in ("plus", "minus")}
    overlap = np.linalg.svd(
        rows["plus"]["basis"].T@rows["minus"]["basis"],
        compute_uv=False)
    for row in rows.values():
        row.pop("basis")
    lossless = all(row["phase_space_rank"] == REAL_DIMENSION
                   for row in rows.values())
    closed = all(row["hodge_phase_space_cross_block"] < 1e-8
                 for row in rows.values())
    positive_lossless = all(row["positive_frequency_rank"] == REAL_DIMENSION
                            for row in rows.values())
    positive_closed = all(row["positive_frequency_hodge_cross_block"] < 1e-8
                          for row in rows.values())
    result = {
        "status": "V4_C19_HODGE_PHASE_SPACE_BRIDGE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "previous_layer_source": (
            "independent centered-time C19 Dirac comparator; measurement "
            "test only, never a native force or update"),
        "rows": list(rows.values()),
        "copy_phase_space_overlap": {
            "count": int(overlap.size),
            "minimum_singular": float(np.min(overlap)),
            "maximum_singular": float(np.max(overlap)),
            "count_above_1_minus_1e_minus_10": int(np.sum(
                overlap > 1-1e-10)),
        },
        "both_copies_lossless": lossless,
        "both_copies_hodge_closed": closed,
        "both_positive_frequency_readouts_lossless": positive_lossless,
        "both_positive_frequency_readouts_hodge_closed": positive_closed,
        "verdict": (
            "HODGE_POSITIVE_FREQUENCY_READOUT_LOSSLESS_AND_CLOSED"
            if positive_lossless and positive_closed else
            "HODGE_PHASE_SPACE_READOUT_REMAINS_INCOMPLETE"),
        "claim_boundary": (
            "Linear positive-frequency tangent around one exact rest orbit. "
            "The previous layer is comparator-specified, so a passing rank "
            "would still require an independently native layer relation, "
            "mass/current mapping, and nonlinear continuation."),
    }
    (HERE/"cube_incidence_hodge_phase_space_bridge.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
