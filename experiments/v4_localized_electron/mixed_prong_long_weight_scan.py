"""Long tangent stability scan across the common/relative action family."""

from __future__ import annotations

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
import exact_translation_multichannel_prong_gate as common
import exact_translation_precursor_linear_gate as precursor
import mixed_prong_dirac_long_tangent as long_gate


HERE = Path(__file__).resolve().parent
DT = native_time.DT
COMMON_CACHE = HERE/"common_prong_tangent_blocks.npz"
ETAS = (0.0, 0.25, 1/3, 0.5, 0.75, 1.0)


def common_del(previous, current, following):
    return common.del_three(previous, current, following, DT, "both")


def common_blocks(rest_layers):
    if COMMON_CACHE.exists():
        data = np.load(COMMON_CACHE)
        return tuple(data[name] for name in ("previous", "current", "following"))
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in rest_layers)
    blocks = []
    for selected in range(3):
        def function(value, index=selected):
            trial = list(tensors)
            trial[index] = value
            return common_del(*trial)
        block = torch.func.jacrev(function)(tensors[selected])
        blocks.append(block.detach().numpy())
        print(f"built common block {selected+1}/3", flush=True)
    np.savez_compressed(
        COMMON_CACHE, previous=blocks[0], current=blocks[1],
        following=blocks[2])
    return tuple(blocks)


def sparse(value, tolerance=1e-11):
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def main() -> None:
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rminus = long_gate.phase_rotation_matrix(rest.size, -theta)
    rplus = long_gate.phase_rotation_matrix(rest.size, theta)
    rest_layers = (rminus@rest, rest, rplus@rest)
    common_values = common_blocks(rest_layers)
    mixed_data = np.load(long_gate.CACHE)
    mixed_values = tuple(mixed_data[name] for name in
                         ("previous", "current", "following"))
    relative_values = tuple(
        (mixed-0.25*base)/0.75
        for mixed, base in zip(mixed_values, common_values))

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    prior = json.loads(
        (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
            encoding="utf-8"))
    flavor, basis = next(iter(dirac.flavor_bases().items()))
    case = next(item for item in prior["rows"] if item["flavor"] == flavor)
    scale_case = next(item for item in case["cases"]
                      if item["envelope"] == "all_125_centers")
    chi_scale = complex(
        scale_case["chi_scale_real"], scale_case["chi_scale_imag"])
    alpha, beta = dirac.compressed_clifford(basis)
    raw = envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, dirac.OFFSETS, chi_scale)
    qbasis = envelope.orth(raw)
    packet_previous, packet_initial, _, momentum = dirac.packet(
        0, -1, alpha, beta, DT)
    initial_previous = raw@dirac.real_coordinates(packet_previous)
    initial_current = raw@dirac.real_coordinates(packet_initial)
    cycles = 10.0
    steps = int(np.ceil(cycles*2*np.pi/rank.R/DT))

    rows = []
    for eta in ETAS:
        blocks = tuple((1-eta)*base+eta*relative
                       for base, relative in zip(
                           common_values, relative_values))
        lhs = sparse(blocks[2]@rplus)
        current_operator = sparse(blocks[1])
        previous_operator = sparse(blocks[0]@rminus)
        factor = splu(lhs)
        previous = initial_previous.copy()
        current = initial_current.copy()
        maximum_norm = 1.0
        maximum_leakage = 0.0
        for _step in range(steps):
            following = factor.solve(
                -(current_operator@current+previous_operator@previous))
            previous, current = current, following
            norm_ratio = np.linalg.norm(current)/np.linalg.norm(initial_current)
            retained = qbasis@(qbasis.T@current)
            leakage = np.linalg.norm(current-retained)/max(
                np.linalg.norm(current), 1e-30)
            maximum_norm = max(maximum_norm, norm_ratio)
            maximum_leakage = max(maximum_leakage, leakage)
            if not np.isfinite(norm_ratio) or norm_ratio > 1e8:
                break
        target = dirac.dirac_step(
            packet_initial, alpha, beta, 1, min(steps, _step+1)*DT)
        target_raw = raw@dirac.real_coordinates(target)
        rows.append({
            "relative_prong_weight": eta,
            "common_weight": 1-eta,
            "completed_steps": _step+1,
            "maximum_norm_ratio": float(maximum_norm),
            "maximum_outside_packet_lift_fraction": float(maximum_leakage),
            "final_native_vs_dirac_relative_error": float(
                np.linalg.norm(current-target_raw)/np.linalg.norm(target_raw)),
            "stable_control": bool(maximum_norm < 1.05
                                   and maximum_leakage < 0.02),
        })
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "status": "V4_MIXED_PRONG_LONG_WEIGHT_SCAN",
        "cycles": cycles,
        "case": {"flavor": flavor, "direction": "x", "sign": -1,
                 "momentum": momentum.tolist()},
        "rows": rows,
        "verdict": (
            "STABLE_GEOMETRIC_WEIGHT_EXISTS"
            if any(row["stable_control"] for row in rows)
            else "MIXED_PRONG_WEIGHT_FAMILY_LONG_NO_PASS"),
        "claim_boundary": (
            "One tangent packet stability scan. A stable weight would still "
            "need independent provenance, Dirac accuracy, all cases, current "
            "and nonlinear endurance."),
    }
    path = HERE/"mixed_prong_long_weight_scan.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
