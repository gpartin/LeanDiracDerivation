"""Take a damped inverse-action step from the projected response matrix.

This is not coefficient fitting to Dirac.  The target is the independently
measured native positive growth rates of the exact rest orbit.  Only
velocity-linear action one-forms, whose coefficients may have either sign and
whose Hamiltonian contribution cancels identically, participate in this
diagnostic step.  Each proposed step is followed by a fresh exact-rest solve
and a full continuous co-rotating spectrum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

import action_response_rank as rank
import continuous_tangent as tangent
import field_berry_probe as field


HERE = Path(__file__).resolve().parent

# Complete signed one-form vocabulary in the current research action.  The
# symmetric kinetic and potential terms are deliberately excluded because a
# least-squares solve would otherwise request inadmissible negative energies.
SIGNED_ONE_FORMS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 26, 27,
    29, 30, 32, 33, 34, 38, 39, 40, 41, 42, 43,
]


def branch_representatives(eigenvalues):
    result = []
    seen = set()
    for index, value in enumerate(eigenvalues):
        key = (round(value["real"], 8), round(abs(value["imag"]), 8))
        if key not in seen:
            seen.add(key)
            result.append(index)
    return result


def inverse_delta(response):
    representatives = branch_representatives(response["unstable_eigenvalues"])
    matrix = np.asarray([
        [response["actions"][action]
         ["unstable_mode_eigenvalue_sensitivities"][mode]["real"]
         for action in SIGNED_ONE_FORMS]
        for mode in representatives])
    target = -np.asarray([
        response["unstable_eigenvalues"][mode]["real"]
        for mode in representatives])
    solution = np.linalg.lstsq(matrix, target, rcond=1e-10)[0]
    delta = np.zeros(len(response["actions"]))
    delta[SIGNED_ONE_FORMS] = solution
    return representatives, matrix, target, delta


def evaluate(coefficients, rate, guess):
    side = 3
    dt = 1.0 / rate
    solution, layers, residual = field.solve_rest(
        side, dt, tuple(coefficients), guess)
    matrices = [field.jacobian(i, layers, side, dt, tuple(coefficients),
                               step=5e-7)
                for i in range(3)]
    rotation = field.prior.bond.local.common.raw_rotation_matrix(
        side, solution.x[1], solution.x[2], 1)
    a_minus = matrices[0] @ rotation.T
    a_zero = matrices[1]
    a_plus = matrices[2] @ rotation
    mass = -0.5 * dt * (a_plus + a_minus)
    gyro = -(a_plus - a_minus)
    stiffness = -(a_plus + a_zero + a_minus) / dt
    dimension = mass.shape[0]
    generator = np.block([
        [np.zeros_like(mass), np.eye(dimension)],
        [-np.linalg.solve(mass, stiffness),
         -np.linalg.solve(mass, gyro)],
    ])
    values = eig(generator, right=False, check_finite=False)
    order = np.argsort(values.real)[::-1]
    mass_values = np.linalg.eigvalsh((mass + mass.T) / 2)
    return {
        "rest_del_rms_residual": residual,
        "amplitude_sq": float(np.exp(solution.x[0])),
        "omega_psi": float(solution.x[1] / dt),
        "omega_chi": float(solution.x[2] / dt),
        "continuous_growth_rate": float(values[order[0]].real),
        "leading_eigenvalues": [
            {"real": float(values[i].real), "imag": float(values[i].imag)}
            for i in order[:16]],
        "mass_minimum_eigenvalue": float(mass_values[0]),
        "mass_negative_count": int(np.sum(mass_values < -1e-8)),
    }, solution.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", default="action_response_rank.json")
    parser.add_argument("--rate", type=int, default=1024)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.02, 0.05, 0.1])
    parser.add_argument("--output", default="inverse_action_step.json")
    args = parser.parse_args()
    field.set_chi_inertia(229.0)
    response = json.loads((HERE / args.response).read_text(encoding="utf-8"))
    representatives, matrix, target, delta = inverse_delta(response)
    baseline = np.asarray(tangent.candidate_coefficients(), dtype=float)
    dt = 1.0 / args.rate
    guess = np.asarray((np.log(178.1135296575095),
                        21.128280812869438 * dt,
                        0.09336897508566924 * dt))
    rows = []
    for fraction in [0.0] + args.fractions:
        coefficients = baseline + fraction * delta
        row, solved = evaluate(coefficients, args.rate, guess)
        row.update({
            "fraction": fraction,
            "predicted_branch_growth": [
                float(-target[i] * (1 - fraction))
                for i in range(len(target))],
        })
        rows.append(row)
        guess = solved
    named_delta = [
        {"index": i, "name": rank.coefficient_names()[i],
         "delta": float(delta[i])}
        for i in SIGNED_ONE_FORMS if abs(delta[i]) > 1e-10]
    result = {
        "status": "V4_INVERSE_ACTION_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "response_rank": int(np.linalg.matrix_rank(matrix)),
        "branch_representatives": representatives,
        "unit_inverse_delta": named_delta,
        "rows": rows,
        "claim_boundary": (
            "Native-rest inverse diagnostic only. Coefficients lack final "
            "geometric provenance; no stability, motion, or Dirac pass."),
    }
    (HERE / args.output).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "response_rank": result["response_rank"],
        "growth_by_fraction": [
            [row["fraction"], row["continuous_growth_rate"]]
            for row in rows],
        "output": str(HERE / args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
