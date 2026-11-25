"""Exact inverse phase-action requirement for the retained C19 cube twist.

The calculation is a finite-dimensional Lyapunov--Schmidt gate.  It asks for
a reciprocal bond potential made from harmonics ``1-cos(n Delta theta)`` that

* has zero phase torque on the retained eight-corner pattern;
* has the same unit quadratic stiffness about aligned fields on face and edge
  bonds; and
* has curvature at least one in every non-global cube phase direction.

The first two harmonics are infeasible.  Allowing harmonics through four gives
the unique minimum-L1 solution ``(1-cos(4 Delta theta))/16`` on both bond
classes.  That result is an inverse requirement, not an admitted action: a
direct fourth harmonic would explicitly favor quarter turns and must instead
be derived from native Psi--chi/cube dynamics.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
CORNERS = list(itertools.product((0, 1), repeat=3))
TARGET = np.asarray((0, 1.5, 1, 0.5, 1, 0.5, 0, 1.5))*np.pi
MODES = [
    np.asarray([(-1)**sum(ki*xi for ki, xi in zip(k, x))
                for x in CORNERS])/np.sqrt(8)
    for k in CORNERS[1:]
]


def response(harmonic: int, distance: int):
    gradient = np.zeros(8)
    hessian = np.zeros((8, 8))
    for i, left in enumerate(CORNERS):
        for j in range(i+1, 8):
            right = CORNERS[j]
            if sum(a != b for a, b in zip(left, right)) != distance:
                continue
            delta = TARGET[i]-TARGET[j]
            sine = harmonic*np.sin(harmonic*delta)
            cosine = harmonic**2*np.cos(harmonic*delta)
            gradient[i] += sine
            gradient[j] -= sine
            hessian[i, i] += cosine
            hessian[j, j] += cosine
            hessian[i, j] -= cosine
            hessian[j, i] -= cosine
    eigen_response = np.asarray([mode@hessian@mode for mode in MODES])
    return gradient, eigen_response


def solve(max_harmonic: int):
    labels = [(distance, harmonic)
              for distance in (1, 2)
              for harmonic in range(1, max_harmonic+1)]
    responses = [response(harmonic, distance)
                 for distance, harmonic in labels]
    gradient = np.column_stack([item[0] for item in responses])
    curvature = np.column_stack([item[1] for item in responses])
    count = len(labels)

    # x = positive-negative.  Minimize ||x||_1 subject to the exact phase
    # torque, aligned stiffness, and target-curvature requirements.
    objective = np.ones(2*count)
    inequality_matrix = []
    inequality_rhs = []
    for row in curvature:
        inequality_matrix.append(np.r_[-row, row])
        inequality_rhs.append(-1.0)
    equality_matrix = []
    equality_rhs = []
    for row in gradient:
        if np.linalg.norm(row) > 1e-12:
            equality_matrix.append(np.r_[row, -row])
            equality_rhs.append(0.0)
    for distance in (1, 2):
        aligned = np.asarray([
            harmonic**2 if bond_distance == distance else 0.0
            for bond_distance, harmonic in labels
        ])
        equality_matrix.append(np.r_[aligned, -aligned])
        equality_rhs.append(1.0)
    solution = linprog(
        objective,
        A_ub=np.asarray(inequality_matrix), b_ub=np.asarray(inequality_rhs),
        A_eq=np.asarray(equality_matrix), b_eq=np.asarray(equality_rhs),
        bounds=[(0, None)]*(2*count), method="highs")
    result = {
        "maximum_harmonic": max_harmonic,
        "feasible": bool(solution.success),
        "solver_message": solution.message,
    }
    if solution.success:
        coefficients = solution.x[:count]-solution.x[count:]
        result.update({
            "labels_distance_then_harmonic": [list(x) for x in labels],
            "coefficients": coefficients.tolist(),
            "l1_norm": float(np.sum(abs(coefficients))),
            "phase_torque_max_abs": float(np.max(abs(gradient@coefficients))),
            "target_mode_curvatures": (curvature@coefficients).tolist(),
            "aligned_face_stiffness": float(sum(
                coefficient*harmonic**2
                for coefficient, (distance, harmonic)
                in zip(coefficients, labels) if distance == 1)),
            "aligned_edge_stiffness": float(sum(
                coefficient*harmonic**2
                for coefficient, (distance, harmonic)
                in zip(coefficients, labels) if distance == 2)),
        })
    return result


def main():
    low = solve(2)
    full = solve(4)
    result = {
        "status": "V4_C19_PHASE_ACTION_INVERSE_REQUIREMENT",
        "target_phases_over_pi": (TARGET/np.pi).tolist(),
        "harmonics_one_and_two": low,
        "harmonics_through_four": full,
        "verdict": "FOURTH_HARMONIC_OR_NATIVE_MEDIATOR_REQUIRED",
        "research_action_admitted": False,
        "canonical_action_changed": False,
        "reason_not_installed": (
            "A direct fourth harmonic explicitly selects quarter-turn phase "
            "differences.  It is an oracle for what native complex chi or a "
            "cube-mediated reciprocal interaction must generate, not itself "
            "acceptable evidence of emergence."
        ),
    }
    destination = HERE/"phase_action_inverse.json"
    destination.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
