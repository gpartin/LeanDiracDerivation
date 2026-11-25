"""Unrestricted four-update Dirac packet gate for the mixed-prong action."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres
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
UPDATES = 4
AMPLITUDE = 1e-4


def phase_rotate(raw: np.ndarray, phase: float) -> np.ndarray:
    value = torch.tensor(raw, dtype=torch.float64)
    return precursor.translate(value, 0, 0, phase, 0.0).detach().numpy()


def residual(previous, current, following):
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    return action.del_residual(*layers, DT).detach().numpy()


def solve(previous, current, guess, factor, frame_phase,
          tolerance=1e-10, maximum_iterations=30):
    following = guess.copy()
    history = []
    for _ in range(maximum_iterations):
        value = residual(previous, current, following)
        norm = float(np.linalg.norm(value)/np.sqrt(value.size))
        history.append(norm)
        if norm < tolerance:
            return following, history, True
        corotating_value = phase_rotate(value, -frame_phase)
        corotating_step = lu_solve(factor, -corotating_value)
        step = phase_rotate(corotating_step, frame_phase)
        accepted = False
        for power in range(9):
            trial = following + 0.5**power * step
            trial_norm = float(np.linalg.norm(
                residual(previous, current, trial))/np.sqrt(value.size))
            if trial_norm < norm:
                following = trial
                accepted = True
                break
        if accepted:
            continue
        # A fixed rest Jacobian is only a preconditioner. If it stalls, solve
        # the exact Newton equation by native Jacobian-vector products. This
        # changes no action term or target and avoids a dense 1500^2 rebuild.
        following_tensor = torch.tensor(following, dtype=torch.float64)
        previous_tensor = torch.tensor(previous, dtype=torch.float64)
        current_tensor = torch.tensor(current, dtype=torch.float64)

        def matvec(vector):
            tangent = torch.tensor(vector.copy(), dtype=torch.float64)
            _base, image = torch.func.jvp(
                lambda next_state: action.del_residual(
                    previous_tensor, current_tensor, next_state, DT),
                (following_tensor,), (tangent,))
            return image.detach().numpy()

        def precondition(vector):
            local = phase_rotate(vector, -frame_phase)
            return phase_rotate(lu_solve(factor, local), frame_phase)

        operator = LinearOperator(
            (value.size, value.size), matvec=matvec, dtype=np.float64)
        preconditioner = LinearOperator(
            (value.size, value.size), matvec=precondition,
            dtype=np.float64)
        exact_step, info = gmres(
            operator, -value, M=preconditioner, rtol=1e-8,
            atol=0.0, restart=12, maxiter=4)
        if info != 0 and not np.all(np.isfinite(exact_step)):
            return following, history, False
        for power in range(12):
            trial = following + 0.5**power * exact_step
            trial_norm = float(np.linalg.norm(
                residual(previous, current, trial))/np.sqrt(value.size))
            if trial_norm < norm:
                following = trial
                accepted = True
                break
        if not accepted:
            return following, history, False
    final = float(np.linalg.norm(
        residual(previous, current, following))/np.sqrt(following.size))
    history.append(final)
    return following, history, final < tolerance


def tangent_guess(previous, current, rest, rest_layers, factor,
                  current_phase):
    """Action-only tangent predictor in the current co-rotating frame."""
    previous_base = phase_rotate(previous, -current_phase)
    current_base = phase_rotate(current, -current_phase)
    delta_previous = previous_base-rest_layers[0]
    delta_current = current_base-rest_layers[1]
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in rest_layers)
    directions = tuple(torch.tensor(value, dtype=torch.float64)
                       for value in (delta_previous, delta_current,
                                     np.zeros_like(rest)))
    _value, rhs = torch.func.jvp(
        lambda a, b, c: action.del_residual(a, b, c, DT),
        layers, directions)
    delta_following_base = lu_solve(factor, -rhs.detach().numpy())
    base_prediction = rest_layers[2]+delta_following_base
    return phase_rotate(base_prediction, current_phase)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--output", default="mixed_prong_dirac_nonlinear.json")
    args = parser.parse_args()
    rank.lock_inertia()
    bare, _ = prior_action.expected_blocks()
    rest = bare["rest"]
    theta = float(np.arccos(1-DT**2*rank.R**2/2))
    rest_previous = phase_rotate(rest, -theta)
    rest_following = phase_rotate(rest, theta)
    rest_layers_np = (rest_previous, rest, rest_following)
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in rest_layers_np)
    following_jacobian = torch.func.jacrev(
        lambda value: action.del_residual(
            tensors[0], tensors[1], value, DT))(tensors[2]).detach().numpy()
    factor = lu_factor(following_jacobian)

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

    rows = []
    specifications = list(itertools.product(
        dirac.flavor_bases().items(), (0, 3), (-1, 1)))[:args.max_cases]
    for (flavor, basis), axis, sign in specifications:
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites,
            dirac.OFFSETS, scales[flavor])
        qbasis = envelope.orth(raw)
        packet_previous, packet_current, _, momentum = dirac.packet(
            axis, sign, alpha, beta, DT)
        q_previous = raw@dirac.real_coordinates(packet_previous)
        q_current = raw@dirac.real_coordinates(packet_current)
        scale = AMPLITUDE*np.linalg.norm(rest)/np.linalg.norm(q_current)
        previous = rest_previous + phase_rotate(scale*q_previous, -theta)
        current = rest + scale*q_current
        dirac_current = packet_current.copy()
        samples = []
        converged = True
        print(f"case {flavor} axis={axis} sign={sign}", flush=True)
        for step in range(1, args.updates+1):
            dirac_current = dirac.dirac_step(
                dirac_current, alpha, beta, 1, DT)
            rest_next = phase_rotate(rest, step*theta)
            current_phase = (step-1)*theta
            guess = tangent_guess(
                previous, current, rest, rest_layers_np, factor,
                current_phase)
            following, history, passed = solve(
                previous, current, guess, factor, current_phase)
            if not passed:
                converged = False
                samples.append({
                    "step": step,
                    "solver_converged": False,
                    "solver_residual_RMS": history[-1],
                })
                break
            perturbation = phase_rotate(following-rest_next, -step*theta)
            target = scale*raw@dirac.real_coordinates(dirac_current)
            retained = qbasis@(qbasis.T@perturbation)
            samples.append({
                "step": step,
                "solver_converged": True,
                "newton_iterations": len(history)-1,
                "native_vs_dirac_lift_relative_error": float(
                    np.linalg.norm(perturbation-target)
                    / np.linalg.norm(target)),
                "outside_packet_lift_fraction": float(
                    np.linalg.norm(perturbation-retained)
                    / max(np.linalg.norm(perturbation), 1e-30)),
                "norm_ratio": float(
                    np.linalg.norm(perturbation)/np.linalg.norm(target)),
            })
            previous, current = current, following
        rows.append({
            "flavor": flavor,
            "direction": "diagonal" if axis == 3 else "x",
            "sign": sign,
            "momentum": momentum.tolist(),
            "completed_updates": len(samples),
            "converged": converged,
            "samples": samples,
        })
    physical = [sample for row in rows for sample in row["samples"]
                if sample.get("solver_converged")]
    all_converged = all(row["converged"]
                        and row["completed_updates"] == args.updates for row in rows)
    worst_error = max((sample["native_vs_dirac_lift_relative_error"]
                       for sample in physical), default=float("inf"))
    worst_leakage = max((sample["outside_packet_lift_fraction"]
                         for sample in physical), default=float("inf"))
    result = {
        "status": "V4_MIXED_PRONG_DIRAC_NONLINEAR",
        "action_weights": {
            "common": action.COMMON_WEIGHT,
            "relative_prongs": action.RELATIVE_WEIGHT,
            "provenance": "one common and three direct prong channels",
        },
        "amplitude_fraction_of_rest_norm": AMPLITUDE,
        "requested_updates": args.updates,
        "all_solvers_converged": all_converged,
        "maximum_relative_packet_error": worst_error,
        "maximum_outside_packet_lift_fraction": worst_leakage,
        "rows": rows,
        "verdict": (
            "MIXED_PRONG_NONLINEAR_SHORT_RUN_PASS"
            if all_converged and worst_error < 3e-2
            and worst_leakage < 2e-2 else
            "MIXED_PRONG_NONLINEAR_SHORT_RUN_OPEN"),
        "claim_boundary": (
            "Four unrestricted nonlinear updates at 0.01% tangent amplitude "
            "for both copies, both signs, one axis and a diagonal. Current, "
            "two translations, 100 cycles and refinement remain."),
    }
    path = HERE/args.output
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
