"""Long tangent propagation for the geometry-weighted mixed-prong action."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import exact_translation_precursor_linear_gate as precursor
import mixed_prong_action as action


HERE = Path(__file__).resolve().parent
DT = native_time.DT
CACHE = HERE/"mixed_prong_tangent_blocks.npz"


def phase_rotation_matrix(size: int, phase: float) -> np.ndarray:
    result = np.eye(size)
    cosine, sine = np.cos(phase), np.sin(phase)
    groups = size // 12
    for group in range(groups):
        base = group*12
        for component in range(3):
            real = base+component
            imag = base+3+component
            result[real, real] = cosine
            result[real, imag] = -sine
            result[imag, real] = sine
            result[imag, imag] = cosine
    return result


def tangent_blocks(rest_layers):
    if CACHE.exists():
        data = np.load(CACHE)
        return tuple(data[name] for name in ("previous", "current", "following"))
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in rest_layers)
    blocks = []
    for selected in range(3):
        def function(value, index=selected):
            trial = list(tensors)
            trial[index] = value
            return action.del_residual(*trial, DT)
        block = torch.func.jacrev(function)(tensors[selected])
        blocks.append(block.detach().numpy())
        print(f"built tangent block {selected+1}/3", flush=True)
    np.savez_compressed(
        CACHE, previous=blocks[0], current=blocks[1], following=blocks[2])
    return tuple(blocks)


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=10.0)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--output", default="mixed_prong_dirac_long_tangent.json")
    args = parser.parse_args()
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rotate_minus = phase_rotation_matrix(rest.size, -theta)
    rotate_plus = phase_rotation_matrix(rest.size, theta)
    rest_layers = (rotate_minus@rest, rest, rotate_plus@rest)
    previous_block, current_block, following_block = tangent_blocks(rest_layers)
    lhs = sparse(following_block@rotate_plus)
    current_operator = sparse(current_block)
    previous_operator = sparse(previous_block@rotate_minus)
    factor = splu(lhs)
    steps = int(np.ceil(args.cycles*2*np.pi/rank.R/DT))
    sample_steps = set(np.linspace(0, steps, 101, dtype=int).tolist())

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    rows = []
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites,
            dirac.OFFSETS, scales[flavor])
        qbasis = envelope.orth(raw)
        packet_previous, packet_initial, _, momentum = dirac.packet(
            axis, sign, alpha, beta, DT)
        q_previous = raw@dirac.real_coordinates(packet_previous)
        q_current = raw@dirac.real_coordinates(packet_initial)
        samples = []
        for step in range(steps+1):
            if step in sample_steps:
                target = dirac.dirac_step(
                    packet_initial, alpha, beta, 1, step*DT)
                target_raw = raw@dirac.real_coordinates(target)
                retained = qbasis@(qbasis.T@q_current)
                target_i = raw@dirac.real_coordinates(1j*target)
                phase_basis = np.column_stack((target_raw, target_i))
                coefficients, *_ = np.linalg.lstsq(
                    phase_basis, q_current, rcond=1e-12)
                length = np.linalg.norm(coefficients)
                aligned = phase_basis@(coefficients/max(length, 1e-30))
                samples.append({
                    "step": step,
                    "cycles": step*DT*rank.R/(2*np.pi),
                    "native_vs_dirac_relative_error": float(
                        np.linalg.norm(q_current-target_raw)
                        / np.linalg.norm(target_raw)),
                    "phase_aligned_shape_error": float(
                        np.linalg.norm(q_current-aligned)
                        / np.linalg.norm(target_raw)),
                    "outside_packet_lift_fraction": float(
                        np.linalg.norm(q_current-retained)
                        / max(np.linalg.norm(q_current), 1e-30)),
                    "norm_ratio": float(
                        np.linalg.norm(q_current)/np.linalg.norm(target_raw)),
                })
            if step == steps:
                break
            q_following = factor.solve(
                -(current_operator@q_current+previous_operator@q_previous))
            q_previous, q_current = q_current, q_following
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "samples": samples,
            "maximum_phase_aligned_shape_error": max(
                sample["phase_aligned_shape_error"] for sample in samples),
            "maximum_outside_packet_lift_fraction": max(
                sample["outside_packet_lift_fraction"] for sample in samples),
            "maximum_norm_drift": max(
                abs(sample["norm_ratio"]-1) for sample in samples),
        })
    worst_shape = max(row["maximum_phase_aligned_shape_error"] for row in rows)
    worst_leakage = max(row["maximum_outside_packet_lift_fraction"] for row in rows)
    worst_norm = max(row["maximum_norm_drift"] for row in rows)
    result = {
        "status": "V4_MIXED_PRONG_DIRAC_LONG_TANGENT",
        "cycles": args.cycles,
        "steps": steps,
        "case_count": len(rows),
        "sparse_nonzeros": {
            "following": int(lhs.nnz),
            "current": int(current_operator.nnz),
            "previous": int(previous_operator.nnz),
        },
        "maximum_phase_aligned_shape_error": worst_shape,
        "maximum_outside_packet_lift_fraction": worst_leakage,
        "maximum_norm_drift": worst_norm,
        "rows": rows,
        "verdict": (
            "MIXED_PRONG_LONG_TANGENT_PASS"
            if worst_shape < 0.05 and worst_leakage < 0.02
            and worst_norm < 0.05 else
            "MIXED_PRONG_LONG_TANGENT_OPEN"),
        "claim_boundary": (
            "Linearized native evolution about the localized rest orbit. "
            "Nonlinear 100-cycle motion and two full translations remain."),
    }
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
