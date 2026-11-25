"""Compare the triangle quadratic Noether current with cubical Dirac current.

For a translation-invariant quadratic U(1)-invariant mode with squared
frequency m^2+lambda(k), the action-derived phase current divided by charge is
grad(lambda)/(2 omega).  The independent cubical Dirac Hamiltonian is

  H(k) = m beta + 2 alpha.s19(k).

Hellmann--Feynman gives its normalized positive-energy current.  This audit
compares those two quantities for both four-component copies without inserting
the Dirac matrices into the triangle action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import c19_triangle_continuum_symbol_audit as symbol
import cube_incidence_extended_packet_gate as dirac


HERE = Path(__file__).resolve().parent
MASS = 19.0
MOMENTA = (0.4, 0.2, 0.1, 0.05, 0.025)
DIRECTIONS = symbol.DIRECTIONS


def fitted_power(rows: list[dict]) -> float:
    x = np.log([row["momentum_norm"] for row in rows])
    y = np.log([max(row["worst_current_relative_error"], 1e-300)
                for row in rows])
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    flavor_data = {
        name: dirac.compressed_clifford(basis)
        for name, basis in dirac.flavor_bases().items()
    }
    direction_rows = []
    for direction_name, direction in DIRECTIONS.items():
        rows = []
        for norm in MOMENTA:
            momentum = norm * direction
            _, lam = symbol.body_symbol(momentum)
            triangle_energy = np.sqrt(MASS**2 + lam)
            triangle_current = symbol.triangle_gradient(momentum) / (
                2.0 * triangle_energy)
            first, jacobian = symbol.c19_symbol_and_jacobian(momentum)
            copy_rows = []
            for flavor, (alpha, beta) in flavor_data.items():
                hamiltonian = MASS * beta + sum(
                    2.0 * first[axis] * alpha[axis] for axis in range(3))
                values, vectors = np.linalg.eigh(hamiltonian)
                positive = np.where(values > 0)[0]
                spin_rows = []
                for index in positive:
                    state = vectors[:, index]
                    current = []
                    for spatial_axis in range(3):
                        derivative = sum(
                            2.0 * jacobian[component, spatial_axis]
                            * alpha[component] for component in range(3))
                        current.append(float(np.real(
                            np.vdot(state, derivative @ state))))
                    spin_rows.append({
                        "energy": float(values[index]),
                        "current": current,
                        "triangle_current_relative_error": float(
                            np.linalg.norm(np.asarray(current)-triangle_current)
                            / max(np.linalg.norm(current), 1e-300)),
                    })
                copy_rows.append({"flavor": flavor, "positive_states": spin_rows})
            worst = max(
                state["triangle_current_relative_error"]
                for copy_row in copy_rows
                for state in copy_row["positive_states"])
            rows.append({
                "momentum_norm": norm,
                "triangle_action_Noether_current_over_charge": (
                    triangle_current.tolist()),
                "copy_rows": copy_rows,
                "worst_current_relative_error": worst,
            })
        direction_rows.append({
            "direction": direction_name,
            "rows": rows,
            "current_error_power": fitted_power(rows),
        })
    powers = [row["current_error_power"] for row in direction_rows]
    passed = all(1.9 < power < 2.1 for power in powers)
    result = {
        "status": "V4_C19_TRIANGLE_NOETHER_DIRAC_CURRENT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "mass": MASS,
        "native_current_rule": (
            "quadratic U(1) Noether current/charge = grad(lambda)/(2 omega)"),
        "dirac_current_rule": (
            "positive-energy Hellmann-Feynman current of the unchanged "
            "cubical H=m beta+2 alpha.s19"),
        "direction_rows": direction_rows,
        "current_error_powers": powers,
        "verdict": (
            "TRIANGLE_QUADRATIC_NOETHER_CURRENT_CONVERGES_TO_DIRAC"
            if passed else "TRIANGLE_DIRAC_CURRENT_CONTINUUM_GATE_FAIL"),
        "claim_boundary": (
            "Quadratic Bloch/current-symbol continuum result for both copies. "
            "It is not the nonlinear localized live-field bond current, "
            "100-cycle motion, or V4-006."),
    }
    output = HERE / "c19_triangle_noether_dirac_current.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "current_error_powers": powers,
        "finest_worst_errors": [row["rows"][-1][
            "worst_current_relative_error"] for row in direction_rows],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
