"""Continue the solvable linked Psi/chi clock tangent to finite amplitude.

The full Fredholm calculation selects dthetaChi/dthetaPsi ~= -0.99862.  This
script changes the clocks by +/-1%, initializes with the derived full-field
tangent, and performs damped minimum-norm Newton corrections on every native
Psi/chi coordinate.  No carrier ray, halo, or exterior coordinate is held.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
FRACTION = 0.01
CONTINUATION_FRACTIONS = (
    1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
MAX_ITERATIONS = 15


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, theta0 = candidate.base.inverse.rest_layers()
    q0 = np.asarray(current, dtype=np.float64)
    tangent_data = np.load(
        HERE / "c27_body_gyro_charge_tangent_solvability.npz")
    linked_tangent = tangent_data["linked_clock_tangent"]
    direction = tangent_data["linked_clock_direction"]
    ratio = float(direction[1] / direction[0])
    a = 2 * spectrum.DT / np.sin(theta0)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    del_residual, right_momentum = candidate.make_derivatives(
        1.0, coefficients)

    def rotation(theta_psi, theta_chi, exponent):
        return relative.raw_rotation_matrix(
            spectrum.SIDE, theta_psi, theta_chi, exponent)

    def solve_at(delta: float, initial: np.ndarray,
                 initial_theta_chi: float):
        theta_psi = theta0 + delta
        theta_chi = initial_theta_chi

        def force_tensor(q, clock):
            minus = torch.tensor(
                rotation(theta_psi, clock, -1), dtype=torch.float64)
            plus = torch.tensor(
                rotation(theta_psi, clock, 1), dtype=torch.float64)
            return del_residual(minus @ q, q, plus @ q)

        q = initial.copy()
        trace = []
        clock_trace = []
        range_remainders = []
        for iteration in range(MAX_ITERATIONS):
            qt = torch.tensor(q, dtype=torch.float64, requires_grad=True)
            residual = force_tensor(qt, theta_chi)
            value = residual.detach().numpy()
            rms = float(np.linalg.norm(value) / np.sqrt(value.size))
            trace.append(rms)
            clock_trace.append(theta_chi)
            if rms < 1e-10:
                break
            jacobian = torch.autograd.functional.jacobian(
                lambda state: force_tensor(state, theta_chi),
                qt, vectorize=True).detach().numpy()
            clock_step = 1e-7
            clock_column = (
                force_tensor(qt.detach(), theta_chi + clock_step)
                - force_tensor(qt.detach(), theta_chi - clock_step)
            ).detach().numpy() / (2 * clock_step)
            augmented = np.column_stack((jacobian, clock_column))
            correction = np.linalg.lstsq(
                augmented, -value, rcond=1e-10)[0]
            range_remainders.append(float(np.linalg.norm(
                augmented @ correction + value) / max(
                    np.linalg.norm(value), 1e-300)))
            accepted = False
            for power in range(12):
                step = 0.5**power
                trial = q + step * correction[:-1]
                trial_clock = theta_chi + step * correction[-1]
                trial_value = force_tensor(
                    torch.tensor(trial, dtype=torch.float64),
                    trial_clock).detach().numpy()
                trial_rms = float(
                    np.linalg.norm(trial_value) / np.sqrt(trial_value.size))
                if trial_rms < rms:
                    q = trial
                    theta_chi = trial_clock
                    accepted = True
                    break
            if not accepted:
                break

        minus_raw = rotation(theta_psi, theta_chi, -1) @ q
        plus_raw = rotation(theta_psi, theta_chi, 1) @ q
        tensors = tuple(torch.tensor(value, dtype=torch.float64)
                        for value in (minus_raw, q, plus_raw))
        final = del_residual(*tensors).detach().numpy()
        momentum = right_momentum(tensors[0], tensors[1]).detach().numpy()
        psi, chi = relative.unpack_raw(q, spectrum.SIDE)
        psi0, chi0 = relative.unpack_raw(q0, spectrum.SIDE)
        occupied = np.sum(abs(psi0)**2, axis=-1) > 1e-12
        psi_density = np.sum(abs(psi)**2, axis=-1)
        chi_density = np.sum(abs(chi)**2, axis=-1)
        generator_psi = relative.pack_raw(1j * psi, np.zeros_like(chi))
        generator_chi_psi = np.zeros_like(psi)
        generator_chi_psi[..., 2] = -1j * psi[..., 2]
        generator_chi = np.zeros_like(chi)
        generator_chi[..., :2] = 1j * chi[..., :2]
        generator_internal = relative.pack_raw(
            generator_chi_psi, generator_chi)
        return {
            "fraction": float(delta / theta0),
            "theta_psi": theta_psi,
            "theta_chi": theta_chi,
            "omega_psi": theta_psi / spectrum.DT,
            "omega_chi": theta_chi / spectrum.DT,
            "trace": trace,
            "theta_chi_trace": clock_trace,
            "linear_range_remainders": range_remainders,
            "iterations": len(trace),
            "final_rest_DEL_rms": float(
                np.linalg.norm(final) / np.sqrt(final.size)),
            "relative_state_change": float(
                np.linalg.norm(q - q0) / np.linalg.norm(q0)),
            "carrier_psi_density_minmax": [
                float(np.min(psi_density[occupied])),
                float(np.max(psi_density[occupied]))],
            "carrier_chi_density_minmax": [
                float(np.min(chi_density[occupied])),
                float(np.max(chi_density[occupied]))],
            "exterior_chi_density_rms_change": float(np.sqrt(np.mean(
                (chi_density[~occupied]
                 - np.sum(abs(chi0[~occupied])**2, axis=-1))**2))),
            "noether_charge_psi": float(np.dot(momentum, generator_psi)),
            "noether_charge_chi_internal": float(
                np.dot(momentum, generator_internal)),
            "state": q,
            "previous": minus_raw,
            "following": plus_raw,
        }

    def continue_sign(sign: float):
        previous_q = q0
        previous_previous_q = None
        previous_delta = 0.0
        previous_previous_delta = None
        previous_clock = 0.0
        previous_previous_clock = None
        steps = []
        for step_index, fraction in enumerate(
                CONTINUATION_FRACTIONS, start=1):
            delta = sign * fraction * theta0
            if previous_previous_q is None:
                initial = q0 + delta * linked_tangent
                initial_clock = ratio * delta
            else:
                scale = ((delta - previous_delta)
                         / (previous_delta - previous_previous_delta))
                initial = previous_q + scale * (
                    previous_q - previous_previous_q)
                initial_clock = previous_clock + scale * (
                    previous_clock - previous_previous_clock)
            row = solve_at(delta, initial, initial_clock)
            row["continuation_step"] = step_index
            row["sign"] = sign
            steps.append(row)
            if row["final_rest_DEL_rms"] >= 1e-8:
                return {"sign": sign, "closed": False, "steps": steps}
            previous_previous_q = previous_q
            previous_previous_delta = previous_delta
            previous_previous_clock = previous_clock
            previous_q = row["state"]
            previous_delta = delta
            previous_clock = row["theta_chi"]
        return {"sign": sign, "closed": True, "steps": steps}

    branches = [continue_sign(-1.0), continue_sign(1.0)]
    solved = [branch["steps"][-1] for branch in branches]
    np.savez_compressed(
        HERE / "c27_body_gyro_linked_clock_continuation.npz",
        minus_state=solved[0]["state"],
        minus_previous=solved[0]["previous"],
        minus_following=solved[0]["following"],
        plus_state=solved[1]["state"],
        plus_previous=solved[1]["previous"],
        plus_following=solved[1]["following"],
    )
    for branch in branches:
        for row in branch["steps"]:
            row.pop("state")
            row.pop("previous")
            row.pop("following")
    closed = all(branch["closed"] for branch in branches)
    result = {
        "status": "V4_C27_BODY_GYRO_LINKED_CLOCK_CONTINUATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "fraction": FRACTION,
        "continuation_fractions": list(CONTINUATION_FRACTIONS),
        "linked_clock_ratio_dthetaChi_over_dthetaPsi": ratio,
        "branches": branches,
        "verdict": (
            "C27_BODY_GYRO_LINKED_CLOCK_FINITE_FAMILY_PASS" if closed else
            "C27_BODY_GYRO_LINKED_CLOCK_FINITE_CONTINUATION_INCONCLUSIVE"),
        "claim_boundary": (
            "Finite +/-1% relative equilibria only. Passing supplies an exact "
            "perturbation family but not nonlinear orbital stability, "
            "translation, or Dirac-current closure."),
    }
    output = HERE / "c27_body_gyro_linked_clock_continuation.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "linked_clock_ratio": ratio,
        "branches": branches,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
