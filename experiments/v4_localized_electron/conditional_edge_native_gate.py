"""Test cubical conditional edge motion against the native second-order step.

The forward/backward projectors are used only to construct and measure the
independent comparator. They never enter the native force. The calculation
separates the centered velocity (which retains gamma) from the centered second
difference (where gamma squares to one).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import cubical_edge_shift_channels as cube
import edge_snap_action_rank as rank
import edge_snap_verlet_gate as rigid


HERE = Path(__file__).resolve().parent
SIDE = 5


def core_sites() -> list[tuple[int, int, int]]:
    center = SIDE // 2 - 1
    return [tuple(center+x for x in corner)
            for corner in itertools.product((0, 1), repeat=3)]


def apply_fiber_operator(field: np.ndarray, operator: np.ndarray) -> np.ndarray:
    sites = core_sites()
    values = np.stack([field[site] for site in sites])
    transformed = operator @ values
    result = np.zeros_like(field)
    for site, value in zip(sites, transformed):
        result[site] = value
    return result


def conditional_step(deviation: np.ndarray, axis: int, direction: int,
                     inverse: bool = False) -> np.ndarray:
    gamma = cube.gamma(axis)
    identity = np.eye(8, dtype=complex)
    forward = apply_fiber_operator(deviation, (identity+gamma)/2)
    backward = apply_fiber_operator(deviation, (identity-gamma)/2)
    orientation = -direction if inverse else direction
    return (np.roll(forward, orientation, axis=axis)
            + np.roll(backward, -orientation, axis=axis))


def layers(field: np.ndarray, vacuum: np.ndarray, axis: int, direction: int,
           phase: float) -> tuple[np.ndarray, np.ndarray]:
    deviation = field-vacuum
    following = vacuum + np.exp(1j*phase)*conditional_step(
        deviation, axis, direction, inverse=False)
    previous = vacuum + np.exp(-1j*phase)*conditional_step(
        deviation, axis, direction, inverse=True)
    return previous, following


def target(phases, psi, chi, axis, direction):
    zero = np.zeros_like(psi)
    vacuum = np.zeros_like(chi)
    vacuum[..., 2] = rank.R
    pp, fp = layers(psi, zero, axis, direction, phases[0])
    pc, fc = layers(chi, vacuum, axis, direction, phases[1])
    required = rank.native.pack(fp-2*psi+pp, fc-2*chi+pc)
    return (pp, pc), (fp, fc), required


def classify(axis: int, direction: int, psi: np.ndarray, chi: np.ndarray,
             acceleration: np.ndarray) -> dict:
    def objective(phases):
        _, _, required = target(phases, psi, chi, axis, direction)
        dt_sq = rigid.best_dt_sq(required, acceleration)
        return float(np.linalg.norm(required-dt_sq*acceleration)
                     / max(np.linalg.norm(required), 1e-30))

    global_result = differential_evolution(
        objective, [(-np.pi, np.pi), (-np.pi, np.pi)],
        seed=40311+10*axis+direction, polish=False, tol=1e-10,
        popsize=12, maxiter=250)
    local_result = minimize(
        objective, global_result.x, method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 2000})
    phases = local_result.x
    previous, following, required = target(
        phases, psi, chi, axis, direction)
    dt_sq = rigid.best_dt_sq(required, acceleration)
    residual = required-dt_sq*acceleration
    dt = float(np.sqrt(dt_sq))

    conditional_velocity = rank.native.pack(
        (following[0]-previous[0])/(2*max(dt, 1e-30)),
        (following[1]-previous[1])/(2*max(dt, 1e-30)))
    zero_phase_previous, zero_phase_following, zero_required = target(
        (0.0, 0.0), psi, chi, axis, direction)
    rigid_zero_required = rank.native.pack(
        np.roll(psi, direction, axis=axis)-2*psi
        + np.roll(psi, -direction, axis=axis),
        np.roll(chi, direction, axis=axis)-2*chi
        + np.roll(chi, -direction, axis=axis))

    rigid_velocity = rank.native.pack(
        (np.roll(psi, direction, axis=axis)
         - np.roll(psi, -direction, axis=axis))/(2*max(dt, 1e-30)),
        (np.roll(chi, direction, axis=axis)
         - np.roll(chi, -direction, axis=axis))/(2*max(dt, 1e-30)))
    return {
        "axis": axis,
        "direction": direction,
        "best_phase_psi": float(phases[0]),
        "best_phase_chi": float(phases[1]),
        "best_dt": dt,
        "combined_relative_residual": float(
            np.linalg.norm(residual)/max(np.linalg.norm(required), 1e-30)),
        "zero_phase_conditional_vs_rigid_acceleration_defect": float(
            np.linalg.norm(zero_required-rigid_zero_required)
            / max(np.linalg.norm(rigid_zero_required), 1e-30)),
        "conditional_vs_rigid_velocity_relative_difference": float(
            np.linalg.norm(conditional_velocity-rigid_velocity)
            / max(np.linalg.norm(conditional_velocity), 1e-30)),
        "optimizer_success": bool(local_result.success),
    }


def flavor_weights(field: np.ndarray) -> dict:
    values = np.stack([field[site] for site in core_sites()])
    identity = np.eye(8, dtype=complex)
    flavor = cube.flavor()
    plus = (identity+flavor)/2 @ values
    minus = (identity-flavor)/2 @ values
    total = float(np.vdot(values, values).real)
    return {
        "plus": float(np.vdot(plus, plus).real/total),
        "minus": float(np.vdot(minus, minus).real/total),
        "reconstruction_defect": float(np.linalg.norm(plus+minus-values)),
    }


def main() -> None:
    rank.lock_inertia()
    amplitude_sq = rank.R**2/(2+54/rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    acceleration = rank.native.pack(force_psi, force_chi)
    rows = [classify(axis, direction, psi, chi, acceleration)
            for axis in range(3) for direction in (-1, 1)]
    result = {
        "status": "V4_CONDITIONAL_EDGE_NATIVE_GATE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "readout_only_projectors": True,
        "rows": rows,
        "carrier_flavor_weights": {
            "psi": flavor_weights(psi),
            "chi_core": flavor_weights(
                chi*(rank.native.norm2(psi)>0)[..., None]),
        },
        "acceleration_identity": (
            "At zero mass phase, C+C^-1-2 equals S+ + S^-1 -2 "
            "because P+ + P-=I and gamma^2=I."),
        "interpretation": (
            "The native second-order equation tests the squared spatial "
            "response and cannot by itself distinguish rigid from conditional "
            "current. The distinction lives in the two committed layers, "
            "their centered velocity, and the frozen four-component readout."),
        "claim_boundary": (
            "A small acceleration residual would not prove Dirac motion. A "
            "pass additionally requires persistence of the signed channel "
            "velocity and beta mixing under unrestricted evolution."),
    }
    (HERE/"conditional_edge_native_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
