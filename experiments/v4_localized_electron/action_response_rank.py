"""Project the complete research action vocabulary onto the failing modes.

This is the inexpensive gate that must precede coefficient continuation or a
nonlinear orbit solve.  It holds the matched exact rest orbit fixed, extracts
the continuous co-rotating generator, and differentiates that generator with
respect to every declared action coefficient.  The result reports

* first-order eigenvalue motion for the leading unstable modes;
* response rank across those modes; and
* response on configuration and velocity translation tangents in x, y, z.

No coefficient is fitted here and no action is promoted.  A zero column is a
scoped proof that the corresponding basis cannot repair the measured defect
at first order.  A nonzero column only earns a later unrestricted rest solve.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import continuous_tangent as tangent
import field_berry_probe as field


HERE = Path(__file__).resolve().parent


def coefficient_names() -> list[str]:
    parameters = list(inspect.signature(field.del_residual).parameters)
    names = parameters[5:]
    if len(names) != 47:
        raise RuntimeError(f"expected 47 action coefficients, found {len(names)}")
    return names


def exact_rest(rate: int):
    side = 3
    dt = 1.0 / rate
    coefficients = tangent.candidate_coefficients()
    guess = np.asarray((np.log(178.1135296575095),
                        21.128280812869438 * dt,
                        0.09336897508566924 * dt))
    solution, layers, residual = field.solve_rest(
        side, dt, coefficients, guess)
    rotation = field.prior.bond.local.common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    jacobians = [field.jacobian(i, layers, side, dt, coefficients,
                                step=5e-7)
                 for i in range(3)]
    a_minus = jacobians[0] @ rotation.T
    a_zero = jacobians[1]
    a_plus = jacobians[2] @ rotation
    mass = -0.5 * dt * (a_plus + a_minus)
    gyro = -(a_plus - a_minus)
    stiffness = -(a_plus + a_zero + a_minus) / dt
    dimension = mass.shape[0]
    generator = np.block([
        [np.zeros_like(mass), np.eye(dimension)],
        [-np.linalg.solve(mass, stiffness),
         -np.linalg.solve(mass, gyro)],
    ])
    return (side, dt, coefficients, solution, layers, residual, rotation,
            mass, gyro, stiffness, generator)


def spectral_derivative(raw: np.ndarray, side: int, axis: int) -> np.ndarray:
    psi, chi = field.prior.bond.local.common.unpack_raw(raw, side)
    wave = 2 * np.pi * np.fft.fftfreq(side)
    shape = [1, 1, 1, 1]
    shape[axis] = side
    multiplier = (1j * wave).reshape(shape)

    def derivative(value):
        transformed = np.fft.fftn(value, axes=(0, 1, 2))
        return np.fft.ifftn(multiplier * transformed, axes=(0, 1, 2))

    return field.prior.bond.local.common.pack_raw(
        derivative(psi), derivative(chi))


class BasisResponse:
    def __init__(self, side: int, dt: float, layers, rotation,
                 mass: np.ndarray, gyro: np.ndarray,
                 stiffness: np.ndarray, step: float):
        self.side = side
        self.dt = dt
        self.layers = layers
        self.rotation = rotation
        self.mass = mass
        self.gyro = gyro
        self.stiffness = stiffness
        self.step = step
        self.count = len(coefficient_names())
        self.zeros = (0.0,) * self.count
        self._zero_cache: dict[tuple[int, bytes, int], np.ndarray] = {}

    def _residual(self, argument: int, direction: np.ndarray, sign: int,
                  coefficients: tuple[float, ...]) -> np.ndarray:
        trial = [layer.copy() for layer in self.layers]
        trial[argument] += sign * self.step * direction
        return field.del_residual(
            *trial, self.side, self.dt, *coefficients)

    def _real_hvp(self, index: int, argument: int,
                  direction: np.ndarray) -> np.ndarray:
        if np.linalg.norm(direction) == 0:
            return np.zeros_like(direction)
        key_base = (argument, np.asarray(direction, dtype=float).tobytes())
        zero_values = []
        for sign in (-1, 1):
            key = key_base + (sign,)
            if key not in self._zero_cache:
                self._zero_cache[key] = self._residual(
                    argument, direction, sign, self.zeros)
            zero_values.append(self._zero_cache[key])
        coefficients = [0.0] * self.count
        coefficients[index] = 1.0
        minus = (self._residual(argument, direction, -1,
                                tuple(coefficients)) - zero_values[0])
        plus = (self._residual(argument, direction, 1,
                               tuple(coefficients)) - zero_values[1])
        return (plus - minus) / (2 * self.step)

    def hvp(self, index: int, argument: int,
            direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction)
        result = self._real_hvp(index, argument, direction.real)
        if np.iscomplexobj(direction) and np.linalg.norm(direction.imag) > 0:
            result = result + 1j * self._real_hvp(
                index, argument, direction.imag)
        return result

    def blocks(self, index: int, direction: np.ndarray):
        minus = self.hvp(index, 0, self.rotation.T @ direction)
        zero = self.hvp(index, 1, direction)
        plus = self.hvp(index, 2, self.rotation @ direction)
        dmass = -0.5 * self.dt * (plus + minus)
        dgyro = -(plus - minus)
        dstiffness = -(plus + zero + minus) / self.dt
        return dmass, dgyro, dstiffness

    def generator_action(self, index: int, state: np.ndarray) -> np.ndarray:
        dimension = self.mass.shape[0]
        q = state[:dimension]
        velocity = state[dimension:]
        acceleration = -np.linalg.solve(
            self.mass, self.stiffness @ q + self.gyro @ velocity)
        dmass_a, _, _ = self.blocks(index, acceleration)
        _, dgyro_v, _ = self.blocks(index, velocity)
        _, _, dstiffness_q = self.blocks(index, q)
        correction = -np.linalg.solve(
            self.mass, dmass_a + dgyro_v + dstiffness_q)
        return np.concatenate((np.zeros(dimension, dtype=correction.dtype),
                               correction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--unstable-count", type=int, default=12)
    parser.add_argument("--hvp-step", type=float, default=2e-6)
    parser.add_argument("--output", default="action_response_rank.json")
    args = parser.parse_args()

    field.set_chi_inertia(229.0)
    (side, dt, coefficients, solution, layers, residual, rotation, mass,
     gyro, stiffness, generator) = exact_rest(args.rate)
    values, left, right = eig(generator, left=True, right=True,
                              check_finite=False)
    unstable = np.where(values.real > 1e-6)[0]
    unstable = unstable[np.argsort(values[unstable].real)[::-1]]
    unstable = unstable[:args.unstable_count]
    response = BasisResponse(side, dt, layers, rotation, mass, gyro,
                             stiffness, args.hvp_step)

    dimension = mass.shape[0]
    translations = []
    for axis in range(3):
        dq = spectral_derivative(layers[1], side, axis)
        dq /= max(np.linalg.norm(dq), 1e-30)
        translations.append((
            np.concatenate((dq, np.zeros(dimension))),
            np.concatenate((np.zeros(dimension), dq))))

    rows = []
    sensitivity_matrix = np.empty((len(unstable), len(coefficient_names())))
    for index, name in enumerate(coefficient_names()):
        sensitivities = []
        mode_response_norms = []
        for row, mode_index in enumerate(unstable):
            delta = response.generator_action(index, right[:, mode_index])
            denominator = np.vdot(left[:, mode_index], right[:, mode_index])
            sensitivity = np.vdot(left[:, mode_index], delta) / denominator
            sensitivities.append({
                "real": float(sensitivity.real),
                "imag": float(sensitivity.imag),
            })
            sensitivity_matrix[row, index] = sensitivity.real
            mode_response_norms.append(float(np.linalg.norm(delta)))
        translation_rows = []
        for axis, pair in enumerate(translations):
            translation_rows.append({
                "axis": axis,
                "configuration_response_norm": float(np.linalg.norm(
                    response.generator_action(index, pair[0]))),
                "velocity_response_norm": float(np.linalg.norm(
                    response.generator_action(index, pair[1]))),
            })
        rows.append({
            "index": index,
            "name": name,
            "current_coefficient": float(coefficients[index]),
            "unstable_mode_eigenvalue_sensitivities": sensitivities,
            "unstable_mode_response_norms": mode_response_norms,
            "translation_responses": translation_rows,
        })

    singular_values = np.linalg.svd(sensitivity_matrix, compute_uv=False)
    tolerance = max(sensitivity_matrix.shape) * singular_values[0] * 1e-10
    result = {
        "status": "V4_ACTION_RESPONSE_RANK_RESEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "rate": args.rate,
        "dt": dt,
        "rest_del_rms_residual": residual,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "omega_psi": float(solution.x[1] / dt),
        "omega_chi": float(solution.x[2] / dt),
        "unstable_eigenvalues": [
            {"real": float(values[i].real), "imag": float(values[i].imag)}
            for i in unstable],
        "real_sensitivity_rank": int(np.sum(singular_values > tolerance)),
        "real_sensitivity_singular_values": [float(x)
                                                for x in singular_values],
        "actions": rows,
        "claim_boundary": (
            "First-order response screen only. Nonzero response is not a "
            "stable orbit, motion pass, Dirac pass, or admitted action."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rest_del_rms_residual": residual,
        "unstable_eigenvalues": result["unstable_eigenvalues"],
        "real_sensitivity_rank": result["real_sensitivity_rank"],
        "top_singular_values": result["real_sensitivity_singular_values"][:8],
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
