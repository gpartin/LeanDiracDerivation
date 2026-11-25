"""Long sparse native-tangent propagation of the extended packet."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as action
import cube_incidence_memory_channel_ablation as memory_ablation
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def sparse(value, tolerance=1e-13):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=100.0)
    parser.add_argument("--dt-divisor", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--history", choices=("dirac", "native-first-step"),
                        default="dirac")
    parser.add_argument(
        "--action-ablation",
        choices=("complete_research_action",
                 "base_without_density_or_cube_connection",
                 "base_plus_density_connection",
                 "base_plus_cube_incidence_connection"),
        default="complete_research_action")
    parser.add_argument("--output", default="cube_incidence_extended_packet_long_tangent.json")
    args = parser.parse_args()
    bare, _matrices = action.expected_blocks()
    selected = memory_ablation.blocks()[args.action_ablation]
    mass, gyro, stiffness = (sparse(value) for value in selected)
    dt = native_time.DT/args.dt_divisor
    lhs = (mass/dt+gyro/2).tocsc()
    current_block = (dt*stiffness-2*mass/dt).tocsc()
    previous_block = (mass/dt-gyro/2).tocsc()
    factor = splu(lhs)
    previous_factor = splu(previous_block)
    steps = int(np.ceil(args.cycles*2*np.pi/native_time.CHI0/dt))
    sample_steps = set(np.linspace(0, steps, 101, dtype=int).tolist())
    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads((HERE/"cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    specifications = list(itertools.product(
        gate.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    rows = []
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        previous, initial, _following, momentum = gate.packet(
            axis, sign, alpha, beta, dt)
        q_current = raw@gate.real_coordinates(initial)
        if args.history == "native-first-step":
            target_one = gate.dirac_step(initial, alpha, beta, 1, dt)
            q_target_one = raw@gate.real_coordinates(target_one)
            q_previous = previous_factor.solve(
                -(lhs@q_target_one+current_block@q_current))
        else:
            q_previous = raw@gate.real_coordinates(previous)
        samples = []
        for step in range(steps+1):
            if step in sample_steps:
                target = gate.dirac_step(initial, alpha, beta, 1, step*dt)
                target_raw = raw@gate.real_coordinates(target)
                # Permit one common readout phase only as a separate identity
                # diagnostic; the unaligned error remains the Dirac time test.
                target_i = raw@gate.real_coordinates(1j*target)
                phase_basis = np.column_stack((target_raw, target_i))
                coefficients, *_ = np.linalg.lstsq(
                    phase_basis, q_current, rcond=1e-12)
                norm = np.linalg.norm(coefficients)
                aligned = phase_basis@(coefficients/max(norm, 1e-30))
                samples.append({
                    "step": step,
                    "time": step*dt,
                    "cycles": step*dt*native_time.CHI0/(2*np.pi),
                    "native_vs_dirac_relative_error": float(
                        np.linalg.norm(q_current-target_raw)
                        / np.linalg.norm(target_raw)),
                    "phase_aligned_shape_error": float(
                        np.linalg.norm(q_current-aligned)
                        / np.linalg.norm(target_raw)),
                    "norm_ratio": float(
                        np.linalg.norm(q_current)/np.linalg.norm(target_raw)),
                    "common_phase": float(np.arctan2(
                        coefficients[1], coefficients[0])),
                })
            if step == steps:
                break
            following = factor.solve(
                -(current_block@q_current+previous_block@q_previous))
            q_previous, q_current = q_current, following
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "samples": samples,
            "maximum_native_vs_dirac_relative_error": max(
                sample["native_vs_dirac_relative_error"] for sample in samples),
            "maximum_phase_aligned_shape_error": max(
                sample["phase_aligned_shape_error"] for sample in samples),
            "maximum_norm_drift": max(
                abs(sample["norm_ratio"]-1) for sample in samples),
        })
    worst_shape = max(row["maximum_phase_aligned_shape_error"] for row in rows)
    worst_norm = max(row["maximum_norm_drift"] for row in rows)
    result = {
        "status": "V4_CUBE_INCIDENCE_EXTENDED_PACKET_LONG_TANGENT",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "dt": dt,
        "dt_divisor": args.dt_divisor,
        "requested_cycles": args.cycles,
        "history_preparation": args.history,
        "action_ablation": args.action_ablation,
        "steps": steps,
        "case_count": len(rows),
        "rows": rows,
        "maximum_phase_aligned_shape_error": worst_shape,
        "maximum_norm_drift": worst_norm,
        "verdict": ("LONG_TANGENT_PACKET_PASS"
                    if worst_shape < 0.05 and worst_norm < 0.05
                    else "LONG_TANGENT_PACKET_OPEN"),
        "claim_boundary": (
            "Sparse linearized native propagation. It does not substitute "
            "for unrestricted nonlinear movement or close V4-006."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
