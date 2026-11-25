"""Diagnostic scale scan for the live-link action contribution.

The candidate action is affine in its declared scale.  This script combines
the independently cached incumbent blocks and unit-scale live-link blocks,
then propagates the unchanged Dirac comparator.  The scan is diagnostic: a
best-fit value is not admitted as coefficient provenance.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as dirac
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import edge_snap_action_rank as rank
import live_clifford_tangent_gate as gate


HERE = Path(__file__).resolve().parent
UNIT = HERE / "live_clifford_tangent_blocks_psi_frame_phase.npz"
INCUMBENT = HERE / "wake_stabilized_completed_tangent_blocks.npz"


def blocks(path: Path):
    data = np.load(path)
    return tuple(data[name] for name in ("previous", "current", "following"))


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def propagate(scale, cycles, specifications, rest, rest_layers,
              rotate_minus, rotate_plus, psi_frame, chi_frame, sites,
              incumbent, addition, frequency_branch, dirac_time):
    previous_block, current_block, following_block = tuple(
        incumbent[index]+scale*addition[index] for index in range(3))
    factor = splu(sparse(following_block@rotate_plus))
    current_operator = sparse(current_block)
    previous_operator = sparse(previous_block@rotate_minus)
    steps = int(np.ceil(cycles*2*np.pi/rank.R/native_time.DT))
    rows = []
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = dirac.compressed_clifford(basis)
        raw = gate.psi_columns(basis, psi_frame, chi_frame, sites)
        qbasis = envelope.orth(raw)
        packet_previous, packet_initial, _, _momentum = dirac.packet(
            axis, sign, alpha, beta, native_time.DT,
            centered_time=dirac_time == "centered",
            frequency_branch=frequency_branch)
        q_previous = raw@dirac.real_coordinates(packet_previous)
        q_current = raw@dirac.real_coordinates(packet_initial)
        target = packet_initial.copy()
        for _step in range(steps):
            q_following = factor.solve(
                -(current_operator@q_current+previous_operator@q_previous))
            q_previous, q_current = q_current, q_following
            if dirac_time == "centered":
                target = dirac.centered_dirac_step(
                    target, alpha, beta, 1, native_time.DT,
                    frequency_branch)
        if dirac_time == "continuous":
            target = dirac.dirac_step(
                packet_initial, alpha, beta, 1, steps*native_time.DT,
                frequency_branch)
        target_raw = raw@dirac.real_coordinates(target)
        target_i = raw@dirac.real_coordinates(1j*target)
        phase_basis = np.column_stack((target_raw, target_i))
        coefficients, *_ = np.linalg.lstsq(
            phase_basis, q_current, rcond=1e-12)
        length = np.linalg.norm(coefficients)
        aligned = phase_basis@(coefficients/max(length, 1e-30))
        retained = qbasis@(qbasis.T@q_current)
        psi_mask = np.zeros(rest.size, dtype=bool)
        for site_index in range(rest.size//12):
            psi_mask[site_index*12:site_index*12+6] = True
        rows.append({
            "flavor": flavor,
            "axis": axis,
            "sign": sign,
            "phase_aligned_shape_error": float(
                np.linalg.norm(q_current-aligned)/np.linalg.norm(target_raw)),
            "psi_cross_block": float(
                np.linalg.norm((q_current-retained)[psi_mask])
                / max(np.linalg.norm(q_current[psi_mask]), 1e-30)),
            "norm_drift": float(
                abs(np.linalg.norm(q_current)/np.linalg.norm(target_raw)-1)),
        })
    return {
        "scale": float(scale),
        "maximum_shape_error": max(row["phase_aligned_shape_error"] for row in rows),
        "maximum_psi_cross_block": max(row["psi_cross_block"] for row in rows),
        "maximum_norm_drift": max(row["norm_drift"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=0.1)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--output", default="live_clifford_scale_scan.json")
    parser.add_argument("--unit", type=Path, default=UNIT)
    parser.add_argument("--unit-scale", type=float, default=1.0)
    parser.add_argument("--base", type=Path, default=INCUMBENT)
    parser.add_argument("--base-scale", type=float, default=0.0)
    parser.add_argument("--frequency-branch",
                        choices=("positive", "negative"),
                        default="positive")
    parser.add_argument("--dirac-time", choices=("continuous", "centered"),
                        default="continuous")
    parser.add_argument("--scales", nargs="*", type=float)
    args = parser.parse_args()
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*rank.R**2/2))
    rotate_minus = gate.phase_rotation_matrix(rest.size, -theta)
    rotate_plus = gate.phase_rotation_matrix(rest.size, theta)
    rest_layers = (rotate_minus@rest, rest, rotate_plus@rest)
    base = blocks(args.base)
    unit = blocks(args.unit)
    interval = args.unit_scale-args.base_scale
    if interval == 0:
        raise ValueError("--unit-scale and --base-scale must differ")
    addition = tuple((unit[index]-base[index])/interval
                     for index in range(3))
    incumbent = tuple(base[index]-args.base_scale*addition[index]
                      for index in range(3))
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    scales = (tuple(args.scales) if args.scales else
              (-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4))
    rows = []
    for scale in scales:
        print(f"scale {scale}", flush=True)
        rows.append(propagate(
            scale, args.cycles, specifications, rest, rest_layers,
            rotate_minus, rotate_plus, psi_frame, chi_frame, sites,
            incumbent, addition, args.frequency_branch, args.dirac_time))
    best = min(rows, key=lambda row: row["maximum_shape_error"])
    zero = min(rows, key=lambda row: abs(row["scale"]))
    result = {
        "status": "V4_LIVE_CLIFFORD_SCALE_SCAN",
        "cycles": args.cycles,
        "case_count": len(specifications),
        "base_cache": str(args.base),
        "base_scale": args.base_scale,
        "unit_cache": str(args.unit),
        "unit_scale": args.unit_scale,
        "frequency_branch": args.frequency_branch,
        "dirac_time_comparator": args.dirac_time,
        "rows": rows,
        "diagnostic_best_scale": best["scale"],
        "diagnostic_best_maximum_shape_error": best["maximum_shape_error"],
        "verdict": (
            "GEOMETRIC_NORMALIZATION_CANDIDATE_FOUND"
            if best["maximum_shape_error"] < zero["maximum_shape_error"]
            else "LIVE_LINK_SCALE_DOES_NOT_IMPROVE_DIRAC_RATE"),
        "claim_boundary": (
            "The scan diagnoses normalization only. A fitted scale is not "
            "coefficient provenance and cannot be admitted."),
    }
    (HERE/args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic_best_scale": result["diagnostic_best_scale"],
        "diagnostic_best_maximum_shape_error": result["diagnostic_best_maximum_shape_error"],
        "incumbent_maximum_shape_error": zero["maximum_shape_error"],
        "verdict": result["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
