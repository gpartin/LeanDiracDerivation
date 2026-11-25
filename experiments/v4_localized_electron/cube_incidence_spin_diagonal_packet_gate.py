"""Test the geometry-derived staggered spin frame for native packet motion.

The previous packet used the same corner-frame basis at every envelope
center.  Staggered lattice Dirac theory instead uses a local cubical frame

    U(r) = alpha_x^r_x alpha_y^r_y alpha_z^r_z,

with exponents reduced modulo two.  This is a change of readout/preparation,
not a force: an independently evolved Dirac field phi is lifted as
z(r)=U(r)phi(r) at each committed layer.  It tests whether the native
overlapping-cube synthesis expects this geometry-derived orientation.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def local_frames(alpha, order):
    frames = np.empty((gate.SIDE, gate.SIDE, gate.SIDE, 4, 4), dtype=complex)
    for index in itertools.product(range(gate.SIDE), repeat=3):
        coordinate = np.asarray(index) - gate.SIDE // 2
        frame = np.eye(4, dtype=complex)
        for axis in order:
            if int(coordinate[axis]) % 2:
                frame = frame @ alpha[axis]
        frames[index] = frame
    return frames


def apply_frames(field, frames):
    return np.einsum("...ab,...b->...a", frames, field)


def run_case(raw, alpha, beta, frames, axis, sign, dt, steps,
             factor, current_block, previous_block):
    previous_phi, current_phi, _following_phi, momentum = gate.packet(
        axis, sign, alpha, beta, dt)
    previous = apply_frames(previous_phi, frames)
    current = apply_frames(current_phi, frames)
    following = apply_frames(
        gate.dirac_step(current_phi, alpha, beta, 1, dt), frames)
    q_previous = raw @ gate.real_coordinates(previous)
    q_current = raw @ gate.real_coordinates(current)
    q_target_one = raw @ gate.real_coordinates(following)
    q_native_one = factor.solve(
        -(current_block @ q_current + previous_block @ q_previous))
    for _ in range(steps):
        q_following = factor.solve(
            -(current_block @ q_current + previous_block @ q_previous))
        q_previous, q_current = q_current, q_following
    target_phi = gate.dirac_step(
        current_phi, alpha, beta, 1, steps * dt)
    target_field = apply_frames(target_phi, frames)
    target = raw @ gate.real_coordinates(target_field)
    target_i = raw @ gate.real_coordinates(1j * target_field)
    phase_basis = np.column_stack((target, target_i))
    coefficients, *_ = np.linalg.lstsq(
        phase_basis, q_current, rcond=1e-12)
    aligned = phase_basis @ (coefficients / max(
        np.linalg.norm(coefficients), 1e-300))
    return {
        "direction": "diagonal" if axis == 3 else "x",
        "sign": sign,
        "momentum": momentum.tolist(),
        "one_step_native_vs_dirac_relative_error": float(
            np.linalg.norm(q_native_one - q_target_one)
            / max(np.linalg.norm(q_target_one), 1e-300)),
        "one_cycle_native_vs_dirac_relative_error": float(
            np.linalg.norm(q_current - target)
            / max(np.linalg.norm(target), 1e-300)),
        "one_cycle_phase_aligned_shape_error": float(
            np.linalg.norm(q_current - aligned)
            / max(np.linalg.norm(target), 1e-300)),
        "one_cycle_norm_ratio": float(
            np.linalg.norm(q_current) / max(np.linalg.norm(target), 1e-300)),
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
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS,
            scales[flavor])
        for order in itertools.permutations(range(3)):
            frames = local_frames(alpha, order)
            for axis, sign in ((0, -1), (0, 1), (3, 1)):
                row = run_case(
                    raw, alpha, beta, frames, axis, sign, dt, steps,
                    factor, current_block, previous_block)
                row["flavor"] = flavor
                row["axis_order"] = list(order)
                rows.append(row)

    best = min(rows, key=lambda row: row[
        "one_cycle_phase_aligned_shape_error"])
    result = {
        "status": "V4_CUBICAL_SPIN_DIAGONAL_PACKET_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "preparation": (
            "same four-component Dirac packet expressed in the local cubical "
            "frame U(r)=product alpha_a^r_a before the unchanged native lift"),
        "rows": rows,
        "best_case": best,
        "verdict": (
            "SPIN_DIAGONAL_FRAME_MATERIAL_IMPROVEMENT"
            if best["one_cycle_phase_aligned_shape_error"] < 0.27 else
            "SPIN_DIAGONAL_FRAME_DOES_NOT_REPAIR_LONG_PACKET"),
        "claim_boundary": (
            "One-cycle linear tangent screen. The local frame is fixed by "
            "cube geometry and is never applied during native evolution. A "
            "pass would still require nonlinear motion, current, refinement, "
            "and Lean proof."),
    }
    (HERE / "cube_incidence_spin_diagonal_packet_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "best_case": best,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
