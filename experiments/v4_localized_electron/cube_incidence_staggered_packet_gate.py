"""Test the cubically staggered four-component packet under the native action.

The complete cube algebra supplies beta as corner-degree parity and alpha_a
anticommutes with beta.  Consequently the discrete Dirac operator preserves
the subspace in which beta-even amplitudes occupy even envelope centers and
beta-odd amplitudes occupy odd envelope centers (or the exchanged choice).

This script tests whether that geometry-derived staggering is also the native
Psi--chi carrier layout missed by the previous all-components-at-every-center
packet.  Dirac is an independent comparator only.  No action coefficient,
force, projection during evolution, or receiver state is changed.
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


def stagger(field: np.ndarray, beta: np.ndarray, phase: int) -> np.ndarray:
    """Apply the native center-parity/beta compatibility projector."""
    identity = np.eye(4, dtype=complex)
    p_plus = (identity + beta) / 2
    p_minus = (identity - beta) / 2
    result = np.zeros_like(field)
    for index in itertools.product(range(gate.SIDE), repeat=3):
        # gate.OFFSETS are ordered as a SIDE^3 array from -2 through 2.
        coordinate = np.asarray(index) - gate.SIDE // 2
        even = (-1) ** int(np.sum(coordinate))
        projector = p_plus if phase * even == 1 else p_minus
        result[index] = projector @ field[index]
    norm = np.linalg.norm(result)
    return result / max(norm, 1e-300) * np.linalg.norm(field)


def stagger_defect(field: np.ndarray, beta: np.ndarray, phase: int) -> float:
    return float(np.linalg.norm(field - stagger(field, beta, phase))
                 / max(np.linalg.norm(field), 1e-300))


def run_case(raw, alpha, beta, phase, axis, sign, steps, dt,
             factor, current_block, previous_block):
    previous0, current0, _following0, momentum = gate.packet(
        axis, sign, alpha, beta, dt)
    current = stagger(current0, beta, phase)
    # Generate the previous layer from the independent discrete Dirac law
    # after staggering.  The exact comparator preserves the staggered space.
    previous = gate.dirac_step(current, alpha, beta, -1, dt)
    q_previous = raw @ gate.real_coordinates(previous)
    q_current = raw @ gate.real_coordinates(current)

    one_target = raw @ gate.real_coordinates(
        gate.dirac_step(current, alpha, beta, 1, dt))
    one_native = factor.solve(
        -(current_block @ q_current + previous_block @ q_previous))
    one_error = float(np.linalg.norm(one_native - one_target)
                      / max(np.linalg.norm(one_target), 1e-300))

    for _ in range(steps):
        following = factor.solve(
            -(current_block @ q_current + previous_block @ q_previous))
        q_previous, q_current = q_current, following
    target_field = gate.dirac_step(current, alpha, beta, 1, steps * dt)
    target = raw @ gate.real_coordinates(target_field)
    target_i = raw @ gate.real_coordinates(1j * target_field)
    phase_basis = np.column_stack((target, target_i))
    coefficients, *_ = np.linalg.lstsq(
        phase_basis, q_current, rcond=1e-12)
    coefficient_norm = np.linalg.norm(coefficients)
    aligned = phase_basis @ (coefficients / max(coefficient_norm, 1e-300))
    return {
        "stagger_phase": phase,
        "direction": "diagonal" if axis == 3 else "xyz"[axis],
        "sign": sign,
        "momentum": momentum.tolist(),
        "initial_stagger_defect": stagger_defect(current, beta, phase),
        "dirac_previous_stagger_defect": stagger_defect(
            previous, beta, phase),
        "one_step_native_vs_dirac_relative_error": one_error,
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
        for phase in (-1, 1):
            for axis, sign in ((0, -1), (0, 1), (3, 1)):
                row = run_case(raw, alpha, beta, phase, axis, sign,
                               steps, dt, factor, current_block,
                               previous_block)
                row["flavor"] = flavor
                rows.append(row)

    best = min(rows, key=lambda row: row[
        "one_cycle_phase_aligned_shape_error"])
    worst_stagger = max(
        max(row["initial_stagger_defect"],
            row["dirac_previous_stagger_defect"])
        for row in rows)
    result = {
        "status": "V4_CUBICAL_STAGGERED_PACKET_GATE",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "preparation": (
            "beta-even and beta-odd four-component amplitudes on alternating "
            "envelope centers, lifted through the unchanged native Psi-chi "
            "carrier synthesis"),
        "dt": dt,
        "one_cycle_steps": steps,
        "rows": rows,
        "maximum_exact_dirac_stagger_invariance_defect": worst_stagger,
        "best_case": best,
        "verdict": (
            "STAGGERED_PREPARATION_MATERIAL_IMPROVEMENT"
            if best["one_cycle_phase_aligned_shape_error"] < 0.315 else
            "STAGGERED_PREPARATION_DOES_NOT_REPAIR_LONG_PACKET"),
        "claim_boundary": (
            "One-cycle linear native tangent test. A preparation improvement "
            "would not admit an action or establish nonlinear motion, current "
            "closure, box convergence, or V4-006."),
    }
    (HERE / "cube_incidence_staggered_packet_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "maximum_exact_dirac_stagger_invariance_defect": worst_stagger,
        "best_case": best,
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
