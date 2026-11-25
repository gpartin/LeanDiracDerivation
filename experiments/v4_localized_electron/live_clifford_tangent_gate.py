"""Propagate both four-component copies under the live-link tangent map.

The carrier lift uses canonical cubical amplitudes on the native Psi frame.
Chi is an independent substrate coordinate at input rather than being fitted
into the four spinor coordinates.  Its induced response remains in the full
native recurrence and can therefore feed back on later updates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
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
import live_clifford_bond_action as action


HERE = Path(__file__).resolve().parent
DT = native_time.DT
CACHE = HERE / "live_clifford_tangent_blocks_discrete_unit_link.npz"


def phase_rotation_matrix(size: int, phase: float) -> np.ndarray:
    result = np.eye(size)
    cosine, sine = np.cos(phase), np.sin(phase)
    for group in range(size // 12):
        base = group*12
        for component in range(3):
            real, imag = base+component, base+3+component
            result[real, real] = cosine
            result[real, imag] = -sine
            result[imag, real] = sine
            result[imag, imag] = cosine
    return result


def rotate_native_phase(state: np.ndarray, phase: float) -> np.ndarray:
    """Apply one common complex phase without constructing a dense matrix."""
    result = state.copy().reshape(-1, 12)
    cosine, sine = np.cos(phase), np.sin(phase)
    for base in (0, 6):
        real = result[:, base:base+3].copy()
        imag = result[:, base+3:base+6].copy()
        result[:, base:base+3] = cosine*real-sine*imag
        result[:, base+3:base+6] = sine*real+cosine*imag
    return result.reshape(-1)


def tangent_blocks(rest_layers, rebuild=False, dt=DT):
    if CACHE.exists() and not rebuild:
        data = np.load(CACHE)
        return tuple(data[name] for name in ("previous", "current", "following"))
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in rest_layers)
    blocks = []
    for selected in range(3):
        def function(value, index=selected):
            trial = list(tensors)
            trial[index] = value
            return action.del_residual(*trial, dt)
        block = torch.func.jacrev(function)(tensors[selected])
        blocks.append(block.detach().numpy())
        print(f"built live-link tangent block {selected+1}/3", flush=True)
    np.savez_compressed(
        CACHE, previous=blocks[0], current=blocks[1], following=blocks[2])
    return tuple(blocks)


def sparse(value, tolerance=None):
    if tolerance is None:
        tolerance = float(os.environ.get("LFM_LIVE_TANGENT_SPARSE_TOL", "1e-11"))
    result = value.copy()
    result[abs(result) < tolerance] = 0
    return csc_matrix(result)


def psi_columns(basis, psi_frame, chi_frame, sites):
    return envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, dirac.OFFSETS, 0j)


def packet_columns(basis, psi_frame, chi_frame, sites, chi_scale):
    return envelope.raw_columns(
        basis, psi_frame, chi_frame, sites, dirac.OFFSETS, chi_scale)


def main() -> None:
    global CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--chi-lift", choices=("zero", "prior"),
                        default="zero")
    parser.add_argument("--output", default="live_clifford_long_tangent.json")
    parser.add_argument("--frequency-branch",
                        choices=("positive", "negative"),
                        default="positive")
    parser.add_argument("--dirac-time", choices=("continuous", "centered"),
                        default="continuous")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--transport-scale", type=float, default=1.0)
    parser.add_argument("--wave-number", type=float, default=0.35)
    parser.add_argument("--packet-width", type=float, default=1.35)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument(
        "--cache", type=Path,
        help="Explicit audited tangent cache; useful for composed scans.")
    args = parser.parse_args()
    dt = args.dt
    scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
    plaquette_tag = format(
        action.PLAQUETTE_SCALE, ".12g").replace(".", "p").replace("-", "m")
    spacetime_tag = format(
        action.SPACETIME_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    symplectic_tag = format(
        action.SYMPLECTIC_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_tag = format(
        action.PARALLEL_OVERLAP_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_face_tag = format(
        action.PARALLEL_FACE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_matched_tag = format(
        action.PARALLEL_MATCHED_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    overlap_potential_tag = format(
        action.PARALLEL_MATCHED_POTENTIAL_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    triangle_tag = format(
        action.TRIANGLE_DELTA_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    material_area_tag = format(
        action.MATERIAL_AREA_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    dt_tag = format(dt, ".12g").replace(".", "p").replace("-", "m")
    CACHE = (HERE
             / f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}_plaquette_{plaquette_tag}_spacetime_{spacetime_tag}_symplectic_{symplectic_tag}_overlap_{overlap_tag}_overlapface_{overlap_face_tag}_overlapmatched_{overlap_matched_tag}_overlappotential_{overlap_potential_tag}_triangle_{triangle_tag}_materialarea_{material_area_tag}_dt_{dt_tag}.npz")
    if args.cache is not None:
        CACHE = args.cache if args.cache.is_absolute() else HERE/args.cache
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-dt**2*rank.R**2/2))
    rotate_minus = phase_rotation_matrix(rest.size, -theta)
    rotate_plus = phase_rotation_matrix(rest.size, theta)
    rest_layers = (rotate_minus@rest, rest, rotate_plus@rest)
    previous_block, current_block, following_block = tangent_blocks(
        rest_layers, rebuild=args.rebuild_cache, dt=dt)
    lhs = sparse(following_block@rotate_plus)
    current_operator = sparse(current_block)
    previous_operator = sparse(previous_block@rotate_minus)
    factor = splu(lhs)
    steps = int(np.ceil(args.cycles*2*np.pi/rank.R/dt))
    sample_steps = set(np.linspace(0, steps, min(101, steps+1), dtype=int))

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    chi_scales = {name: 0j for name in dirac.flavor_bases()}
    if args.chi_lift == "prior":
        prior = json.loads(
            (HERE/"cube_incidence_dirac_envelope_gate.json").read_text(
                encoding="utf-8"))
        for row in prior["rows"]:
            case = next(item for item in row["cases"]
                        if item["envelope"] == "all_125_centers")
            chi_scales[row["flavor"]] = complex(
                case["chi_scale_real"], case["chi_scale_imag"])
    rows = []
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[
            args.case_offset:args.case_offset+args.max_cases]
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = dirac.compressed_clifford(basis)
        raw = packet_columns(
            basis, psi_frame, chi_frame, sites, chi_scales[flavor])
        qbasis = envelope.orth(raw)
        packet_previous, packet_initial, _, momentum = dirac.packet(
            axis, sign, alpha, beta, dt,
            width=args.packet_width,
            carrier_wave_number=args.wave_number,
            centered_time=args.dirac_time == "centered",
            frequency_branch=args.frequency_branch,
            transport_scale=args.transport_scale)
        q_previous = raw@dirac.real_coordinates(packet_previous)
        q_current = raw@dirac.real_coordinates(packet_initial)
        dirac_current = packet_initial.copy()
        samples = []
        for step in range(steps+1):
            q_following = factor.solve(
                -(current_operator@q_current+previous_operator@q_previous))
            if step in sample_steps:
                if args.dirac_time == "centered":
                    target = dirac_current
                else:
                    target = dirac.dirac_step(
                        packet_initial, alpha, beta, 1, step*dt,
                        args.frequency_branch, args.transport_scale)
                target_raw = raw@dirac.real_coordinates(target)
                retained = qbasis@(qbasis.T@q_current)
                target_i = raw@dirac.real_coordinates(1j*target)
                phase_basis = np.column_stack((target_raw, target_i))
                coefficients, *_ = np.linalg.lstsq(
                    phase_basis, q_current, rcond=1e-12)
                length = np.linalg.norm(coefficients)
                aligned = phase_basis@(coefficients/max(length, 1e-30))
                # The native equations are second order.  Their lossless
                # first-order amplitude lives in the current layer together
                # with the centered velocity from the adjacent layers.  The
                # old configuration-only metric can therefore report false
                # breathing when amplitude moves into conjugate momentum.
                target_previous = dirac.centered_dirac_step(
                    target, alpha, beta, -1, dt,
                    args.frequency_branch, args.transport_scale)
                target_following = dirac.centered_dirac_step(
                    target, alpha, beta, 1, dt,
                    args.frequency_branch, args.transport_scale)
                target_previous_raw = raw@dirac.real_coordinates(
                    target_previous)
                target_following_raw = raw@dirac.real_coordinates(
                    target_following)
                native_velocity = (
                    rotate_native_phase(q_following, theta)
                    - rotate_native_phase(q_previous, -theta))/(2*dt)
                target_velocity = (
                    rotate_native_phase(target_following_raw, theta)
                    - rotate_native_phase(target_previous_raw, -theta))/(2*dt)
                frequency_sign = (
                    1.0 if args.frequency_branch == "positive" else -1.0)
                native_amplitude = 0.5*(
                    q_current + frequency_sign*rotate_native_phase(
                        native_velocity, np.pi/2)/rank.R)
                target_amplitude = 0.5*(
                    target_raw + frequency_sign*rotate_native_phase(
                        target_velocity, np.pi/2)/rank.R)
                target_amplitude_i = rotate_native_phase(
                    target_amplitude, np.pi/2)
                amplitude_basis = np.column_stack(
                    (target_amplitude, target_amplitude_i))
                amplitude_coefficients, *_ = np.linalg.lstsq(
                    amplitude_basis, native_amplitude, rcond=1e-12)
                amplitude_length = np.linalg.norm(amplitude_coefficients)
                amplitude_aligned = amplitude_basis@(
                    amplitude_coefficients/max(amplitude_length, 1e-30))
                retained_amplitude = qbasis@(
                    qbasis.T@native_amplitude)
                psi_mask = np.zeros(rest.size, dtype=bool)
                for site_index in range(rest.size//12):
                    psi_mask[site_index*12:site_index*12+6] = True
                samples.append({
                    "step": step,
                    "cycles": step*dt*rank.R/(2*np.pi),
                    "native_vs_dirac_relative_error": float(
                        np.linalg.norm(q_current-target_raw)
                        / np.linalg.norm(target_raw)),
                    "phase_aligned_shape_error": float(
                        np.linalg.norm(q_current-aligned)
                        / np.linalg.norm(target_raw)),
                    "psi_outside_four_component_fraction": float(
                        np.linalg.norm((q_current-retained)[psi_mask])
                        / max(np.linalg.norm(q_current[psi_mask]), 1e-30)),
                    "induced_chi_fraction": float(
                        np.linalg.norm(q_current[~psi_mask])
                        / max(np.linalg.norm(q_current), 1e-30)),
                    "norm_ratio": float(
                        np.linalg.norm(q_current)/np.linalg.norm(target_raw)),
                    "phase_space_shape_error": float(
                        np.linalg.norm(native_amplitude-amplitude_aligned)
                        / max(np.linalg.norm(target_amplitude), 1e-30)),
                    "phase_space_cross_block": float(
                        np.linalg.norm(native_amplitude-retained_amplitude)
                        / max(np.linalg.norm(native_amplitude), 1e-30)),
                    "phase_space_norm_ratio": float(
                        np.linalg.norm(native_amplitude)
                        / max(np.linalg.norm(target_amplitude), 1e-30)),
                })
            if step == steps:
                break
            q_previous, q_current = q_current, q_following
            if args.dirac_time == "centered":
                dirac_current = dirac.centered_dirac_step(
                    dirac_current, alpha, beta, 1, dt,
                    args.frequency_branch, args.transport_scale)
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "samples": samples,
            "maximum_phase_aligned_shape_error": max(
                sample["phase_aligned_shape_error"] for sample in samples),
            "maximum_psi_cross_block": max(
                sample["psi_outside_four_component_fraction"] for sample in samples),
            "maximum_induced_chi_fraction": max(
                sample["induced_chi_fraction"] for sample in samples),
            "maximum_norm_drift": max(
                abs(sample["norm_ratio"]-1) for sample in samples),
            "maximum_phase_space_shape_error": max(
                sample["phase_space_shape_error"] for sample in samples),
            "maximum_phase_space_cross_block": max(
                sample["phase_space_cross_block"] for sample in samples),
            "maximum_phase_space_norm_drift": max(
                abs(sample["phase_space_norm_ratio"]-1)
                for sample in samples),
        })
    worst_shape = max(row["maximum_phase_aligned_shape_error"] for row in rows)
    worst_leakage = max(row["maximum_psi_cross_block"] for row in rows)
    worst_norm = max(row["maximum_norm_drift"] for row in rows)
    worst_phase_shape = max(
        row["maximum_phase_space_shape_error"] for row in rows)
    worst_phase_leakage = max(
        row["maximum_phase_space_cross_block"] for row in rows)
    worst_phase_norm = max(
        row["maximum_phase_space_norm_drift"] for row in rows)
    result = {
        "status": "V4_LIVE_CLIFFORD_LONG_TANGENT",
        "link_source": getattr(
            action, "LINK_SOURCE",
            getattr(getattr(action, "incumbent", None), "LINK_SOURCE", None)),
        "action_scale": getattr(
            action, "SCALE",
            getattr(getattr(action, "incumbent", None), "SCALE", None)),
        "chi_payload": getattr(
            action, "CHI_PAYLOAD",
            getattr(getattr(action, "incumbent", None), "CHI_PAYLOAD", None)),
        "relative_grade_scale": getattr(action, "GRADE_SCALE", None),
        "parallel_overlap_scale": getattr(
            action, "PARALLEL_OVERLAP_SCALE", None),
        "parallel_face_scale": getattr(action, "PARALLEL_FACE_SCALE", None),
        "parallel_matched_scale": getattr(
            action, "PARALLEL_MATCHED_SCALE", None),
        "parallel_matched_potential_scale": getattr(
            action, "PARALLEL_MATCHED_POTENTIAL_SCALE", None),
        "triangle_delta_scale": getattr(action, "TRIANGLE_DELTA_SCALE", None),
        "material_area_scale": getattr(action, "MATERIAL_AREA_SCALE", None),
        "chi_lift": args.chi_lift,
        "frequency_branch": args.frequency_branch,
        "dirac_time_comparator": args.dirac_time,
        "tangent_cache": CACHE.name,
        "tangent_cache_rebuilt": args.rebuild_cache,
        "dirac_transport_scale": args.transport_scale,
        "dirac_comparator_mass": dirac.COMPARATOR_MASS,
        "carrier_wave_number": args.wave_number,
        "packet_width": args.packet_width,
        "dt": dt,
        "chi_scales": {
            key: [float(value.real), float(value.imag)]
            for key, value in chi_scales.items()
        },
        "cycles": args.cycles,
        "steps": steps,
        "case_count": len(rows),
        "maximum_phase_aligned_shape_error": worst_shape,
        "maximum_psi_cross_block": worst_leakage,
        "maximum_norm_drift": worst_norm,
        "maximum_phase_space_shape_error": worst_phase_shape,
        "maximum_phase_space_cross_block": worst_phase_leakage,
        "maximum_phase_space_norm_drift": worst_phase_norm,
        "rows": rows,
        "verdict": (
            "LIVE_CLIFFORD_LONG_TANGENT_PASS"
            if worst_phase_shape < 0.05 and worst_phase_leakage < 0.02
            and worst_phase_norm < 0.05 else
            "LIVE_CLIFFORD_LONG_TANGENT_OPEN"),
        "claim_boundary": (
            "Full two-field linear recurrence about the localized rest orbit. "
            "The pass verdict uses the fixed lossless phase-space amplitude; "
            "configuration-only metrics remain as breathing diagnostics. "
            "Nonlinear translation, current, refinement, and 100 cycles remain."),
    }
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
