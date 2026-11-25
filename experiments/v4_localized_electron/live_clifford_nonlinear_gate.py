"""Four unrestricted nonlinear updates for the live-link action candidate.

The Dirac packet supplies only the two initial perturbation layers and the
post-update comparator.  The native candidate action alone supplies every
force and every subsequent layer through its implicit DEL equation.
"""

from __future__ import annotations

import itertools
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
import torch

import canonical_noncoplanar_bifurcation_audit as native_time
import canonical_transfer_oscillator_spectrum as native_spectrum
import cube_incidence_extended_packet_gate as gate
import cube_incidence_nonlinear_action as prior_action
import current_action_localized_envelope_closure as envelope
import live_clifford_bond_action as action
import live_clifford_tangent_gate as tangent
import live_face_compact_gradient_bridge as gradient_bridge


HERE = Path(__file__).resolve().parent
UPDATES = 4
AMPLITUDE = 1e-4


def residual(previous: np.ndarray, current: np.ndarray,
             following: np.ndarray) -> np.ndarray:
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))
    return action.del_residual(*tensors, native_time.DT).detach().numpy()


def relative_residual(previous: np.ndarray, current: np.ndarray,
                      following: np.ndarray,
                      base_residual: np.ndarray | None) -> np.ndarray:
    value = residual(previous, current, following)
    return value if base_residual is None else value-base_residual


def solve_following(previous, current, guess, factor,
                    tolerance=1e-10, maximum_iterations=48,
                    base_residual=None):
    following = guess.copy()
    history = []
    active_factor = factor
    refreshes = 0
    for _iteration in range(maximum_iterations):
        value = relative_residual(
            previous, current, following, base_residual)
        rms = float(np.linalg.norm(value)/np.sqrt(value.size))
        history.append(rms)
        if rms < tolerance:
            return following, history, True, active_factor
        correction = lu_solve(active_factor, -value)
        accepted = False
        for power in range(10):
            trial = following+0.5**power*correction
            trial_rms = float(np.linalg.norm(
                relative_residual(previous, current, trial, base_residual))
                /np.sqrt(value.size))
            if trial_rms < rms:
                following = trial
                accepted = True
                if trial_rms/max(rms, 1e-300) > 0.7 and refreshes < 2:
                    p = torch.tensor(previous, dtype=torch.float64)
                    c = torch.tensor(current, dtype=torch.float64)
                    f = torch.tensor(following, dtype=torch.float64)
                    jacobian = torch.func.jacrev(
                        lambda value: action.del_residual(
                            p, c, value, native_time.DT))(f).detach().numpy()
                    active_factor = lu_factor(jacobian)
                    refreshes += 1
                break
        if not accepted:
            if refreshes >= 2:
                break
            # The action is nonlinear in the live chi geometry.  A fixed
            # rest-orbit Jacobian is only a preconditioner; refresh the exact
            # following-layer Jacobian when its Newton direction ceases to
            # decrease the DEL residual.  This changes the solver, not the
            # action or state.
            p = torch.tensor(previous, dtype=torch.float64)
            c = torch.tensor(current, dtype=torch.float64)
            f = torch.tensor(following, dtype=torch.float64)
            jacobian = torch.func.jacrev(
                lambda value: action.del_residual(
                    p, c, value, native_time.DT))(f).detach().numpy()
            active_factor = lu_factor(jacobian)
            refreshes += 1
    return following, history, False, active_factor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--chi-lift", choices=("zero", "prior"),
                        default="zero")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--solver-tolerance", type=float, default=1e-10)
    parser.add_argument("--raw-residual", action="store_true",
                        help="do not subtract the floating rest-orbit residual")
    parser.add_argument("--wave-number", type=float, default=0.35)
    parser.add_argument("--packet-width", type=float, default=1.35)
    parser.add_argument("--readout", choices=("local", "face-gradient"),
                        default="local")
    parser.add_argument("--dirac-time", choices=("continuous", "centered"),
                        default="continuous")
    parser.add_argument("--frequency-branch",
                        choices=("positive", "negative"),
                        default="positive")
    parser.add_argument("--save-states",
                        help="optional NPZ path for the first trajectory")
    args = parser.parse_args()
    scale_tag = format(action.SCALE, ".12g").replace(".", "p").replace("-", "m")
    plaquette_tag = format(
        action.PLAQUETTE_SCALE, ".12g").replace(".", "p").replace("-", "m")
    spacetime_tag = format(
        action.SPACETIME_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    symplectic_tag = format(
        action.SYMPLECTIC_PLAQUETTE_SCALE,
        ".12g").replace(".", "p").replace("-", "m")
    tangent.CACHE = (HERE
                     / f"live_clifford_tangent_blocks_{action.LINK_SOURCE}_{scale_tag}_plaquette_{plaquette_tag}_spacetime_{spacetime_tag}_symplectic_{symplectic_tag}.npz")
    bare, _matrices = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-native_time.DT**2*native_time.CHI0**2/2))
    rotation = native_spectrum.rotation_matrix(rest.size, theta)
    rest_layers = (rotation.T@rest, rest, rotation@rest)
    previous_block, current_block, following_block = tangent.tangent_blocks(
        rest_layers)
    factor = lu_factor(following_block)
    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    chi_scales = {name: 0j for name in gate.flavor_bases()}
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
    saved_states = None
    specifications = list(itertools.product(
        gate.flavor_bases().items(), (0, 3), (-1, 1)))[
            args.case_offset:args.case_offset+args.max_cases]
    for (flavor, basis), axis, sign in specifications:
        # A refreshed Newton factor is local to a trajectory.  Reusing one
        # packet's nonlinear Jacobian for the other Clifford copy can create
        # artificial slow convergence even though the action is unchanged.
        solver_factor = factor
        alpha, beta = gate.compressed_clifford(basis)
        if args.readout == "face-gradient":
            parts = gradient_bridge.components(
                basis, psi_frame, chi_frame, sites)
            laplacian = gradient_bridge.center_face_laplacian()
            # Native kinetic normalization fixes the magnitude.  The
            # positive-frequency companion quadrature fixes the common
            # (-1+i) phase.  The same expression is used for both exhaustive
            # cubical copies and for every direction; it is never refitted
            # during a trajectory.
            coefficient = 1/np.sqrt(native_time.B)
            values = np.asarray((-coefficient, coefficient,
                                 -coefficient, coefficient))
            raw = (parts[0]
                   + values[0]*parts[1]+values[1]*parts[2]
                   + values[2]*(parts[1]@laplacian)
                   + values[3]*(parts[2]@laplacian))
        else:
            raw = envelope.raw_columns(
                basis, psi_frame, chi_frame, sites, gate.OFFSETS,
                chi_scales[flavor])
        qbasis = envelope.orth(raw)
        packet_previous, packet_current, _packet_following, momentum = (
            gate.packet(axis, sign, alpha, beta, native_time.DT,
                        width=args.packet_width,
                        carrier_wave_number=args.wave_number,
                        centered_time=args.dirac_time == "centered",
                        frequency_branch=args.frequency_branch))
        q_previous = raw@gate.real_coordinates(packet_previous)
        q_current = raw@gate.real_coordinates(packet_current)
        scale = args.amplitude*np.linalg.norm(rest)/np.linalg.norm(q_current)
        q_previous *= scale
        q_current *= scale
        previous = rotation.T@rest+rotation.T@q_previous
        current = rest+q_current
        trajectory = [previous.copy(), current.copy()]
        rest_layer = rest.copy()
        dirac_current = packet_current.copy()
        samples = []
        converged = True
        for step in range(1, args.updates+1):
            dirac_stepper = (gate.centered_dirac_step
                             if args.dirac_time == "centered"
                             else gate.dirac_step)
            dirac_current = dirac_stepper(
                dirac_current, alpha, beta, 1, native_time.DT,
                args.frequency_branch)
            rest_next = rotation@rest_layer
            rest_previous = rotation.T@rest_layer
            base_residual = (None if args.raw_residual else residual(
                rest_previous, rest_layer, rest_next))
            center_power = step-1
            unrotate = np.linalg.matrix_power(rotation.T, center_power)
            rerotate = np.linalg.matrix_power(rotation, center_power)
            aligned_previous = unrotate@previous
            aligned_current = unrotate@current
            delta_previous = aligned_previous-rotation.T@rest
            delta_current = aligned_current-rest
            delta_following = lu_solve(
                factor, -(previous_block@delta_previous
                          + current_block@delta_current))
            tangent_guess = rerotate@(rotation@rest+delta_following)
            following, history, passed, solver_factor = solve_following(
                previous, current, tangent_guess, solver_factor,
                tolerance=args.solver_tolerance,
                base_residual=base_residual)
            if not passed:
                converged = False
                samples.append({
                    "step": step,
                    "solver_converged": False,
                    "solver_residual_RMS": history[-1],
                    "solver_residual_history_RMS": history,
                })
                break
            perturbation = following-rest_next
            tangent_perturbation = tangent_guess-rest_next
            corotating = perturbation.copy()
            for _ in range(step):
                corotating = rotation.T@corotating
            recovered, *_ = np.linalg.lstsq(raw, corotating, rcond=1e-10)
            reconstructed = raw@recovered
            target = scale*raw@gate.real_coordinates(dirac_current)
            target_i = scale*raw@gate.real_coordinates(1j*dirac_current)
            phase_plane = np.column_stack((target, target_i))
            phase_coefficients, *_ = np.linalg.lstsq(
                phase_plane, corotating, rcond=1e-12)
            phase_fit = phase_plane@phase_coefficients
            target_norm = max(np.linalg.norm(target), 1e-30)
            samples.append({
                    "step": step,
                    "solver_converged": True,
                    "newton_iterations": len(history)-1,
                    "solver_residual_RMS": history[-1],
                    "native_vs_dirac_relative_error": float(
                        np.linalg.norm(corotating-target)/np.linalg.norm(target)),
                    "outside_four_component_fraction": float(
                        np.linalg.norm(corotating-reconstructed)
                        / max(np.linalg.norm(corotating), 1e-30)),
                    "within_four_component_relative_error": float(
                        np.linalg.norm(reconstructed-target)/target_norm),
                    "phase_amplitude_aligned_shape_error": float(
                        np.linalg.norm(corotating-phase_fit)/target_norm),
                    "phase_plane_coefficients": [
                        float(value) for value in phase_coefficients],
                    "native_norm_ratio": float(
                        np.linalg.norm(corotating)/np.linalg.norm(target)),
                    "nonlinear_vs_local_tangent_relative_error": float(
                        np.linalg.norm(perturbation-tangent_perturbation)
                        / max(np.linalg.norm(perturbation), 1e-30)),
            })
            previous, current = current, following
            trajectory.append(current.copy())
            rest_layer = rest_next
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "converged": converged,
            "samples": samples,
        })
        print(f"completed {flavor} axis={axis} sign={sign}: "
              f"converged={converged}", flush=True)
        if saved_states is None:
            saved_states = np.stack(trajectory)
    physical = [sample for row in rows for sample in row["samples"]
                if sample.get("solver_converged")]
    complete = all(row["converged"] and len(row["samples"]) == args.updates
                   for row in rows)
    maximum_error = max(
        (row["native_vs_dirac_relative_error"] for row in physical),
        default=None)
    maximum_leakage = max(
        (row["outside_four_component_fraction"] for row in physical),
        default=None)
    result = {
        "status": "V4_LIVE_CLIFFORD_NONLINEAR_GATE",
        "link_source": action.LINK_SOURCE,
        "action_scale": action.SCALE,
        "plaquette_scale": action.PLAQUETTE_SCALE,
        "spacetime_plaquette_scale": action.SPACETIME_PLAQUETTE_SCALE,
        "symplectic_plaquette_scale": action.SYMPLECTIC_PLAQUETTE_SCALE,
        "amplitude_fraction_of_rest_norm": args.amplitude,
        "chi_lift": args.chi_lift,
        "chi_scales": {
            key: [float(value.real), float(value.imag)]
            for key, value in chi_scales.items()
        },
        "requested_updates": args.updates,
        "carrier_wave_number": args.wave_number,
        "packet_width": args.packet_width,
        "readout": args.readout,
        "dirac_time_comparator": args.dirac_time,
        "frequency_branch": args.frequency_branch,
        "face_gradient_coefficient": (
            [-float(1/np.sqrt(native_time.B)),
             float(1/np.sqrt(native_time.B))]
            if args.readout == "face-gradient" else None),
        "solver_tolerance_RMS": args.solver_tolerance,
        "rest_roundoff_subtracted": not args.raw_residual,
        "all_solvers_converged": complete,
        "maximum_native_vs_dirac_relative_error": maximum_error,
        "maximum_outside_four_component_fraction": maximum_leakage,
        "rows": rows,
        "verdict": (
            "LIVE_CLIFFORD_NONLINEAR_SHORT_RUN_PASS"
            if complete and maximum_error is not None and maximum_error < 0.03
            and maximum_leakage is not None and maximum_leakage < 0.02
            else "LIVE_CLIFFORD_NONLINEAR_SHORT_RUN_OPEN"),
        "claim_boundary": (
            f"{args.updates} unrestricted nonlinear small-amplitude updates. This is "
            "not a full-amplitude carrier handoff, long run, or V4-006 pass."),
    }
    suffix = action.LINK_SOURCE.replace(".", "p")
    output = (HERE/args.output if args.output else
              HERE/f"live_clifford_nonlinear_gate_{suffix}.json")
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.save_states and saved_states is not None:
        np.savez_compressed(HERE/args.save_states, states=saved_states,
                            dt=native_time.DT,
                            rest=rest, rotation=rotation)
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
