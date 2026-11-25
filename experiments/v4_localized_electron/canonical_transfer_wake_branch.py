"""Continue the exact rest orbit under the odd rotor-wake kinetic metric."""

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
STRENGTH = 1e-4
MODE = "transfer_odd"


def main() -> None:
    rest = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)
    canonical_del = grad(
        lambda p, c, f: box.interval(p, c, large.SIDE)
        + box.interval(c, f, large.SIDE), argnums=1)
    candidate_del = grad(
        lambda p, c, f: box.interval(p, c, large.SIDE)
        + box.interval(c, f, large.SIDE)
        + STRENGTH*(wake.extra_interval(p, c, MODE)
                    + wake.extra_interval(c, f, MODE)), argnums=1)

    def mapped(raw: torch.Tensor, candidate: bool):
        previous = box.phase_rotate(
            raw, np.conjugate(phase), large.SIDE)
        following = box.phase_rotate(raw, phase, large.SIDE)
        return (candidate_del if candidate else canonical_del)(
            previous, raw, following)

    def residual(state: np.ndarray) -> np.ndarray:
        return mapped(torch.tensor(state, dtype=torch.float64), True) \
            .detach().numpy()

    checkpoint = HERE / "canonical_transfer_wake_branch.npz"
    state = (np.load(checkpoint)["state"] if checkpoint.exists()
             else rest.copy())
    trace = []
    left_null = np.zeros((rest.size, 0))
    for iteration in range(8):
        value = residual(state)
        rms = np.linalg.norm(value)/np.sqrt(value.size)
        tensor = torch.tensor(state, dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(
            lambda raw: mapped(raw, True), tensor,
            vectorize=True).detach().numpy()
        u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
        active = singular > 1e-10*max(singular[0], 1e-300)
        left = u[:, active]
        right = vh[active].T
        values = singular[active]
        left_null = u[:, ~active]
        null_norm = np.linalg.norm(left_null.T@value)
        trace.append({
            "iteration": iteration,
            "residual_rms": float(rms),
            "left_null_norm": float(null_norm),
            "jacobian_nullity": int(np.sum(~active)),
            "smallest_retained_singular_value": float(values[-1]),
        })
        if rms < 1e-10:
            break
        correction = -right@((left.T@value)/values)
        accepted = False
        for power in range(16):
            trial = state+0.5**power*correction
            trial_rms = np.linalg.norm(residual(trial))/np.sqrt(value.size)
            if trial_rms < rms:
                state = trial
                trace[-1]["accepted_scale"] = float(0.5**power)
                trace[-1]["trial_residual_rms"] = float(trial_rms)
                accepted = True
                break
        if not accepted:
            trace[-1]["accepted_scale"] = 0.0
            break

    final = residual(state)
    psi0, _chi0 = large.unpack_np(rest)
    psi, chi = large.unpack_np(state)
    core = np.sum(abs(psi0)**2, axis=-1) > 1e-12
    final_rms = float(np.linalg.norm(final)/np.sqrt(final.size))
    result = {
        "status": "V4_CANONICAL_TRANSFER_WAKE_BRANCH",
        "mode": MODE,
        "strength": STRENGTH,
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "trace": trace,
        "final_residual_rms": final_rms,
        "state_displacement_over_rest": float(
            np.linalg.norm(state-rest)/np.linalg.norm(rest)),
        "final_left_null_norm": float(np.linalg.norm(left_null.T@final)),
        "halo_psi_power_fraction": float(
            np.sum(abs(psi[~core])**2)/max(np.sum(abs(psi)**2), 1e-300)),
        "halo_chi_deviation_norm": float(np.linalg.norm(
            chi[~core]-np.asarray((0, 0, small.CHI0)))),
        "verdict": (
            "TRANSFER_WAKE_DEFORMED_REST_ORBIT_SOLVED"
            if final_rms < 1e-9 else
            "TRANSFER_WAKE_DEFORMED_REST_ORBIT_NOT_YET_SOLVED"),
        "claim_boundary": (
            "Fixed-frequency pseudoinverse continuation using the canonical "
            "DEL Jacobian. A solved orbit would still require coefficient "
            "provenance, updated transport/current, stability, and finite "
            "translation gates."),
    }
    output = HERE / "canonical_transfer_wake_branch.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    np.savez_compressed(HERE / "canonical_transfer_wake_branch.npz",
                        rest=rest, state=state, strength=STRENGTH,
                        phase=phase)
    print(json.dumps({key: result[key] for key in (
        "verdict", "final_residual_rms", "state_displacement_over_rest",
        "final_left_null_norm", "halo_psi_power_fraction",
        "halo_chi_deviation_norm")}, indent=2))
    print(json.dumps({"last_trace": trace[-5:], "output": str(output)},
                     indent=2))


if __name__ == "__main__":
    main()
