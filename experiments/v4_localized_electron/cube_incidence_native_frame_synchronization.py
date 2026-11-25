"""Synchronize the four carrier directions using every native C19 overlap.

Single-bond polar factors are underdetermined when a neighbor overlap is rank
deficient.  This audit builds the full Hermitian connection adjacency on the
periodic center lattice, takes its four leading global modes, and obtains a
local U(4) frame from their blockwise polar factors.  The construction uses
only the native carrier, kinetic metric, and C19 geometry.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, polar
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_native_frame_connection as connection
import cube_incidence_nonlinear_action as action
import cube_incidence_spin_diagonal_packet_gate as spin_gate
import current_action_localized_envelope_closure as envelope
import localized_stability as native


HERE = Path(__file__).resolve().parent
SIDE = 5


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def flat(index):
    return (index[0] * SIDE + index[1]) * SIDE + index[2]


def synchronized_frames(basis, psi_frame, chi_frame, sites, chi_scale):
    center = connection.mode_matrix(
        basis, psi_frame, chi_frame, sites, (0, 0, 0), chi_scale)
    metric = center.conj().T @ center
    whiten = connection.inverse_sqrt(metric)
    overlaps = {}
    for offset, weight in native.DIRECTIONS:
        neighbor = connection.mode_matrix(
            basis, psi_frame, chi_frame, sites, offset, chi_scale)
        overlaps[offset] = whiten @ (center.conj().T @ neighbor) @ whiten

    size = SIDE ** 3 * 4
    adjacency = np.zeros((size, size), dtype=complex)
    for index in itertools.product(range(SIDE), repeat=3):
        first = flat(index)
        for offset, weight in native.DIRECTIONS:
            target = tuple((index[axis] + offset[axis]) % SIDE
                           for axis in range(3))
            second = flat(target)
            adjacency[first*4:(first+1)*4, second*4:(second+1)*4] += (
                weight * overlaps[offset])
    adjacency = (adjacency + adjacency.conj().T) / 2
    values, vectors = eigh(adjacency, subset_by_index=(size-4, size-1))
    frames = np.empty((SIDE, SIDE, SIDE, 4, 4), dtype=complex)
    local_singular = []
    local_unitarity = []
    for index in itertools.product(range(SIDE), repeat=3):
        site = flat(index)
        block = vectors[site*4:(site+1)*4, :]
        unitary, positive = polar(block)
        frames[index] = unitary
        local_singular.extend(np.linalg.svd(block, compute_uv=False).tolist())
        local_unitarity.append(np.linalg.norm(
            unitary.conj().T @ unitary - np.eye(4)))
    return frames, {
        "leading_connection_eigenvalues": [float(value) for value in values],
        "minimum_local_block_singular": float(min(local_singular)),
        "maximum_local_block_singular": float(max(local_singular)),
        "maximum_local_frame_unitarity_defect": float(max(local_unitarity)),
    }


def main() -> None:
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

    rows = []
    frame_diagnostics = {}
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS,
            scales[flavor])
        frames, diagnostics = synchronized_frames(
            basis, psi_frame, chi_frame, sites, scales[flavor])
        frame_diagnostics[flavor] = diagnostics
        for axis, sign in ((0, -1), (0, 1), (3, 1)):
            row = spin_gate.run_case(
                raw, alpha, beta, frames, axis, sign, dt, steps,
                factor, current_block, previous_block)
            row["flavor"] = flavor
            rows.append(row)

    best = min(rows, key=lambda row: row[
        "one_cycle_phase_aligned_shape_error"])
    result = {
        "status": "V4_NATIVE_FRAME_SYNCHRONIZATION_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "frame_derivation": (
            "top four modes of the native kinetic-overlap C19 connection "
            "adjacency, blockwise polar synchronized"),
        "frame_diagnostics": frame_diagnostics,
        "rows": rows,
        "best_case": best,
        "verdict": (
            "NATIVE_SYNCHRONIZED_FRAME_MATERIAL_IMPROVEMENT"
            if best["one_cycle_phase_aligned_shape_error"] < 0.217 else
            "NATIVE_SYNCHRONIZED_FRAME_DOES_NOT_CLOSE_LONG_PACKET"),
        "claim_boundary": (
            "Native-overlap preparation/readout test only. No action or "
            "coefficient is changed; a favorable result would still require "
            "nonlinear motion, current, refinement, and Lean proof."),
    }
    (HERE / "cube_incidence_native_frame_synchronization.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "frame_diagnostics": frame_diagnostics,
        "best_case": best,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
