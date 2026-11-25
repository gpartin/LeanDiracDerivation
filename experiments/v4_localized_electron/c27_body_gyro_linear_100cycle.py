"""One-hundred-cycle linear perturbation gate for the C27/gyro candidate.

Unit-circle eigensolvers split highly degenerate symmetry modes by about
5e-8.  Direct propagation of carrier-supported perturbations distinguishes
that numerical splitting from a Jordan drift or a resolved instability.
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
CYCLES = (0, 1, 10, 100)


def monodromy(use_candidate: bool, layers, theta):
    a = 2 * spectrum.DT / np.sin(theta)
    if use_candidate:
        potential = 1.0
        coefficients = torch.tensor((
            a + candidate.base.inverse.rank.B / 4,
            a - candidate.base.inverse.rank.B / 4,
            -a / 2,
            0.0,
        ), dtype=torch.float64)
    else:
        potential = 0.0
        coefficients = torch.zeros(4, dtype=torch.float64)

    def action(pm, q, pp):
        return candidate.three_layer_action(
            pm, q, pp, potential, coefficients)

    variables = tuple(value.detach().requires_grad_(True) for value in layers)
    hessian = torch.autograd.functional.hessian(
        action, variables, vectorize=True)
    previous = hessian[1][0].detach().numpy()
    current = hessian[1][1].detach().numpy()
    following = hessian[1][2].detach().numpy()
    rotation = relative.raw_rotation_matrix(spectrum.SIDE, theta, 0.0, 1)
    lhs = following @ rotation
    forward = -np.linalg.solve(lhs, current)
    backward = -np.linalg.solve(lhs, previous @ rotation.T)
    identity = np.eye(forward.shape[0])
    return np.block([
        [forward, backward],
        [identity, np.zeros_like(identity)],
    ])


def perturbations(current):
    raw = current.detach().numpy()
    psi, chi = relative.unpack_raw(raw, spectrum.SIDE)
    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    zero_psi = np.zeros_like(psi)
    zero_chi = np.zeros_like(chi)
    rows = {
        "psi_phase": relative.pack_raw(1j * psi, zero_chi),
        "chi_phase": relative.pack_raw(zero_psi, 1j * chi),
    }
    for axis in range(3):
        rows[f"translation_axis_{axis}"] = relative.pack_raw(
            np.roll(psi, 1, axis=axis) - psi,
            np.roll(chi, 1, axis=axis) - chi)
    rng = np.random.default_rng(40027)
    random_psi = np.zeros_like(psi)
    random_chi = np.zeros_like(chi)
    random_psi[occupied] = (
        rng.normal(size=random_psi[occupied].shape)
        + 1j * rng.normal(size=random_psi[occupied].shape))
    random_chi[occupied] = (
        rng.normal(size=random_chi[occupied].shape)
        + 1j * rng.normal(size=random_chi[occupied].shape))
    rows["random_carrier"] = relative.pack_raw(random_psi, random_chi)
    return {name: value / max(np.linalg.norm(value), 1e-300)
            for name, value in rows.items()}


def classify(matrix, vectors, theta):
    steps_per_cycle = int(round(2 * np.pi / theta))
    cycle = np.linalg.matrix_power(matrix, steps_per_cycle)
    tangent_names = (
        "psi_phase", "chi_phase", "translation_axis_0",
        "translation_axis_1", "translation_axis_2")
    tangent_matrix = np.column_stack([vectors[name]
                                      for name in tangent_names])
    quotient_basis = np.linalg.qr(tangent_matrix)[0]
    rows = {}
    for name, vector in vectors.items():
        for layer_relation, previous in (
                ("co_rotating_zero_velocity", vector),
                ("co_rotating_velocity", -vector)):
            state = np.r_[vector, previous]
            initial = np.linalg.norm(state)
            samples = {"0": {
                "raw_state_amplification": 1.0,
                "shape_residual": float(np.linalg.norm(
                    vector - quotient_basis
                    @ (quotient_basis.T @ vector))),
            }}
            for cycle_index in range(1, max(CYCLES) + 1):
                state = cycle @ state
                if cycle_index in CYCLES:
                    current = state[:vector.size]
                    shape = (current - quotient_basis
                             @ (quotient_basis.T @ current))
                    samples[str(cycle_index)] = {
                        "raw_state_amplification": float(
                            np.linalg.norm(state) / initial),
                        "shape_residual": float(np.linalg.norm(shape)),
                    }
            rows[f"{name}:{layer_relation}"] = samples
    return steps_per_cycle, rows


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    layers = tuple(torch.tensor(value, dtype=torch.float64)
                   for value in (previous, current, following))
    vectors = perturbations(layers[1])
    baseline_matrix = monodromy(False, layers, theta)
    candidate_matrix = monodromy(True, layers, theta)
    steps, baseline = classify(baseline_matrix, vectors, theta)
    _, trial = classify(candidate_matrix, vectors, theta)
    baseline_max = max(
        value["100"]["raw_state_amplification"]
        for value in baseline.values())
    trial_max = max(
        value["100"]["raw_state_amplification"]
        for value in trial.values())
    baseline_shape = max(
        value["100"]["shape_residual"] for value in baseline.values())
    trial_shape = max(
        value["100"]["shape_residual"] for value in trial.values())
    bounded = trial_shape < 10.0
    result = {
        "status": "V4_C27_BODY_GYRO_LINEAR_100_CYCLE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "theta": float(theta),
        "steps_per_sampled_cycle": steps,
        "cycles": list(CYCLES),
        "baseline": baseline,
        "candidate": trial,
        "baseline_maximum_100_cycle_amplification": float(baseline_max),
        "candidate_maximum_100_cycle_amplification": float(trial_max),
        "baseline_maximum_100_cycle_shape_residual": float(baseline_shape),
        "candidate_maximum_100_cycle_shape_residual": float(trial_shape),
        "candidate_shape_bounded_after_phase_translation_alignment": bounded,
        "verdict": (
            "C27_BODY_GYRO_LINEAR_100_CYCLE_PASS" if bounded else
            "C27_BODY_GYRO_LINEAR_100_CYCLE_FAIL"),
        "claim_boundary": (
            "Linearized carrier-supported controls only. It does not replace "
            "unrestricted nonlinear implicit-action evolution or box and "
            "perturbation refinement."),
    }
    output = HERE / "c27_body_gyro_linear_100cycle.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "steps_per_cycle": steps,
        "baseline_maximum_100_cycle_amplification": baseline_max,
        "candidate_maximum_100_cycle_amplification": trial_max,
        "baseline_maximum_100_cycle_shape_residual": baseline_shape,
        "candidate_maximum_100_cycle_shape_residual": trial_shape,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
