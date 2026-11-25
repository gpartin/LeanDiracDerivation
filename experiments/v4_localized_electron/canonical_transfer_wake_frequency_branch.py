"""Joint state/frequency continuation for the odd rotor-wake metric.

The branch parameter is fixed by the action-derived Psi U(1) charge, not by
holding a raw amplitude.  No Dirac quantity enters the nonlinear solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_wake_kinetic_gate as wake


HERE = Path(__file__).resolve().parent
MODE = "transfer_odd"
STRENGTH = 1e-4
THETA0 = float(np.arccos(1-small.DT**2*small.CHI0**2/2))


def phase_rotate(raw: torch.Tensor, theta: torch.Tensor,
                 sign: float) -> torch.Tensor:
    psi, chi = box.unpack(raw, large.SIDE)
    phase = torch.complex(torch.cos(theta), sign*torch.sin(theta))
    rotated = phase*psi
    return torch.stack((torch.real(rotated), torch.imag(rotated),
                        torch.real(chi), torch.imag(chi)),
                       dim=-2).reshape(-1)


def psi_generator(raw: torch.Tensor) -> torch.Tensor:
    psi, chi = box.unpack(raw, large.SIDE)
    zero = torch.zeros_like(chi)
    return torch.stack((torch.real(1j*psi), torch.imag(1j*psi),
                        torch.real(zero), torch.imag(zero)),
                       dim=-2).reshape(-1)


def main() -> None:
    canonical = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    rest = canonical["rest"]
    prior_path = HERE / "canonical_transfer_wake_branch.npz"
    start_state = (np.load(prior_path)["state"] if prior_path.exists()
                   else rest.copy())

    def interval(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (box.interval(left, right, large.SIDE)
                + STRENGTH*wake.extra_interval(left, right, MODE))

    candidate_del = grad(
        lambda p, c, f: interval(p, c)+interval(c, f), argnums=1)
    candidate_momentum = grad(interval, argnums=1)
    canonical_momentum = grad(
        lambda p, c: box.interval(p, c, large.SIDE), argnums=1)

    rest_t = torch.tensor(rest, dtype=torch.float64)
    theta0_t = torch.tensor(THETA0, dtype=torch.float64)
    rest_previous = phase_rotate(rest_t, theta0_t, -1.0)
    target_charge = torch.dot(
        canonical_momentum(rest_previous, rest_t), psi_generator(rest_t)
    ).detach()
    charge_scale = max(abs(float(target_charge)), 1.0)

    def equations(raw: torch.Tensor) -> torch.Tensor:
        state, fractional_shift = raw[:-1], raw[-1]
        theta = THETA0*(1+fractional_shift)
        previous = phase_rotate(state, theta, -1.0)
        following = phase_rotate(state, theta, 1.0)
        field = candidate_del(previous, state, following)
        charge = torch.dot(
            candidate_momentum(previous, state), psi_generator(state))
        constraint = (charge-target_charge)/charge_scale
        return torch.cat((field, constraint.reshape(1)))

    start_shift = 4.500950199968191e-6/THETA0
    checkpoint = HERE / "canonical_transfer_wake_frequency_branch.npz"
    if checkpoint.exists():
        saved = np.load(checkpoint)
        vector = saved["vector"]
    else:
        vector = np.r_[start_state, start_shift]
    trace = []
    for iteration in range(8):
        tensor = torch.tensor(vector, dtype=torch.float64)
        value = equations(tensor).detach().numpy()
        field_rms = np.linalg.norm(value[:-1])/np.sqrt(value.size-1)
        trace.append({
            "iteration": iteration,
            "field_residual_rms": float(field_rms),
            "relative_charge_residual": float(value[-1]),
            "frequency_shift_fraction": float(vector[-1]),
            "theta": float(THETA0*(1+vector[-1])),
        })
        if field_rms < 1e-10 and abs(value[-1]) < 1e-10:
            break
        jacobian = torch.autograd.functional.jacobian(
            equations, tensor, vectorize=True).detach().numpy()
        u, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
        active = singular > 1e-10*max(singular[0], 1e-300)
        correction = -vh[active].T@(
            (u[:, active].T@value)/singular[active])
        accepted = False
        score = np.sqrt(field_rms**2+value[-1]**2)
        for power in range(18):
            trial = vector+0.5**power*correction
            trial_value = equations(torch.tensor(
                trial, dtype=torch.float64)).detach().numpy()
            trial_field = (np.linalg.norm(trial_value[:-1])
                           / np.sqrt(trial_value.size-1))
            trial_score = np.sqrt(trial_field**2+trial_value[-1]**2)
            if trial_score < score:
                vector = trial
                trace[-1]["accepted_scale"] = float(0.5**power)
                trace[-1]["trial_score"] = float(trial_score)
                accepted = True
                break
        np.savez_compressed(checkpoint, vector=vector, rest=rest,
                            strength=STRENGTH, theta0=THETA0,
                            target_charge=float(target_charge))
        if not accepted:
            trace[-1]["accepted_scale"] = 0.0
            break

    final = equations(torch.tensor(vector, dtype=torch.float64)).detach().numpy()
    state = vector[:-1]
    psi0, _chi0 = large.unpack_np(rest)
    psi, chi = large.unpack_np(state)
    core = np.sum(abs(psi0)**2, axis=-1) > 1e-12
    final_rms = float(np.linalg.norm(final[:-1])/np.sqrt(final.size-1))
    result = {
        "status": "V4_CANONICAL_TRANSFER_WAKE_FREQUENCY_BRANCH",
        "mode": MODE,
        "strength": STRENGTH,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "constraint": "candidate-action Psi U(1) charge equals canonical rest charge",
        "target_charge": float(target_charge),
        "trace": trace,
        "final_field_residual_rms": final_rms,
        "final_relative_charge_residual": float(final[-1]),
        "frequency_shift_fraction": float(vector[-1]),
        "final_theta": float(THETA0*(1+vector[-1])),
        "state_displacement_over_rest": float(
            np.linalg.norm(state-rest)/np.linalg.norm(rest)),
        "halo_psi_power_fraction": float(
            np.sum(abs(psi[~core])**2)/max(np.sum(abs(psi)**2), 1e-300)),
        "halo_chi_deviation_norm": float(np.linalg.norm(
            chi[~core]-np.asarray((0, 0, small.CHI0)))),
        "verdict": (
            "TRANSFER_WAKE_STATE_FREQUENCY_ORBIT_SOLVED"
            if final_rms < 1e-9 and abs(final[-1]) < 1e-9 else
            "TRANSFER_WAKE_STATE_FREQUENCY_ORBIT_NOT_YET_SOLVED"),
        "claim_boundary": (
            "Joint native state/frequency solve at fixed action-derived charge. "
            "A solved orbit would still require coefficient provenance, "
            "spectrum, current, continuum, and finite-translation gates."),
    }
    output = HERE / "canonical_transfer_wake_frequency_branch.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "verdict", "final_field_residual_rms",
        "final_relative_charge_residual", "frequency_shift_fraction",
        "state_displacement_over_rest", "halo_psi_power_fraction",
        "halo_chi_deviation_norm")}, indent=2))
    print(json.dumps({"last_trace": trace[-4:], "output": str(output)},
                     indent=2))


if __name__ == "__main__":
    main()
