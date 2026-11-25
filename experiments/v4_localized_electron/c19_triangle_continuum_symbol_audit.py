"""Audit the low-momentum symbol of the C19 triangle transport channel.

The exact C19 face--edge factorization has the same translation symbol as
the eight body endpoints, but it does not introduce a fundamental C27 read.
For a plane wave with lattice momentum k, the symmetric endpoint factors are

  C_body(k) = 8 cos(kx) cos(ky) cos(kz),
  lambda_body(k) = 8 - C_body(k).

The latter must approach 4 |k|^2 isotropically.  The native gyro coefficient
a(dt)=2 dt/sin(theta), theta=chi0 dt, must approach 2/chi0.  These are
necessary continuum-Dirac compatibility conditions for a second-order action.

The stronger check below uses the already-derived cubical Clifford embedding,
without refitting it.  If s_19(k) is the exact C19 first-moment symbol, then

  [2 alpha.s_19(k)]^2 = 4 |s_19(k)|^2 I_4.

We compare the triangle symbol, its massive dispersion, and its group velocity
with that independent Dirac square.  Agreement in the continuum is stronger
than merely observing an isotropic wave symbol, but it still does not prove
that the full live-Psi/live-chi tangent preserves this four-component sector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import cube_incidence_extended_packet_gate as dirac
import localized_stability as c19


HERE = Path(__file__).resolve().parent
CHI0 = 19.0
MOMENTA = (0.4, 0.2, 0.1, 0.05, 0.025)
DIRECTIONS = {
    "axis": np.asarray((1.0, 0.0, 0.0)),
    "face_diagonal": np.asarray((1.0, 1.0, 0.0)) / np.sqrt(2.0),
    "body_diagonal": np.asarray((1.0, 1.0, 1.0)) / np.sqrt(3.0),
}
RATES = (128, 256, 512, 1024, 2048)


def body_symbol(momentum: np.ndarray) -> tuple[float, float]:
    product = float(np.prod(np.cos(momentum)))
    return 8.0 * product, 8.0 * (1.0 - product)


def triangle_gradient(momentum: np.ndarray) -> np.ndarray:
    result = np.zeros(3)
    for axis in range(3):
        value = 8.0 * np.sin(momentum[axis])
        for other in range(3):
            if other != axis:
                value *= np.cos(momentum[other])
        result[axis] = value
    return result


def c19_symbol_and_jacobian(momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symbol = np.zeros(3)
    jacobian = np.zeros((3, 3))
    for offset, weight in c19.DIRECTIONS:
        direction = np.asarray(offset, dtype=float)
        phase = float(momentum @ direction)
        symbol += weight * direction * np.sin(phase)
        jacobian += weight * np.outer(direction, direction) * np.cos(phase)
    return symbol / 12.0, jacobian / 12.0


def fitted_power(rows: list[dict], key: str) -> float:
    x = np.log([row["momentum_norm"] for row in rows])
    y = np.log([max(abs(row[key]), 1e-300) for row in rows])
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    direction_rows = []
    for name, direction in DIRECTIONS.items():
        rows = []
        for norm in MOMENTA:
            momentum = norm * direction
            correlation, laplacian = body_symbol(momentum)
            first_symbol, first_jacobian = c19_symbol_and_jacobian(momentum)
            dirac_square = 4.0 * float(first_symbol @ first_symbol)
            triangle_energy = np.sqrt(CHI0**2 + laplacian)
            dirac_energy = np.sqrt(CHI0**2 + dirac_square)
            triangle_velocity = triangle_gradient(momentum) / (
                2.0 * triangle_energy)
            dirac_square_gradient = 8.0 * first_jacobian.T @ first_symbol
            dirac_velocity = dirac_square_gradient / (2.0 * dirac_energy)
            rows.append({
                "momentum_norm": norm,
                "C_body": correlation,
                "lambda_body": laplacian,
                "lambda_over_4_k2": laplacian / (4.0 * norm**2),
                "relative_quadratic_remainder": (
                    laplacian / (4.0 * norm**2) - 1.0),
                "four_times_C19_first_symbol_squared": dirac_square,
                "triangle_vs_dirac_square_relative_error": (
                    laplacian / dirac_square - 1.0),
                "massive_triangle_energy": triangle_energy,
                "massive_dirac_energy": dirac_energy,
                "massive_dispersion_relative_error": (
                    triangle_energy / dirac_energy - 1.0),
                "triangle_group_velocity": triangle_velocity.tolist(),
                "dirac_group_velocity": dirac_velocity.tolist(),
                "group_velocity_relative_error": float(
                    np.linalg.norm(triangle_velocity - dirac_velocity)
                    / max(np.linalg.norm(dirac_velocity), 1e-300)),
            })
        direction_rows.append({
            "direction": name,
            "rows": rows,
            "relative_remainder_power": fitted_power(
                rows, "relative_quadratic_remainder"),
            "square_match_error_power": fitted_power(
                rows, "triangle_vs_dirac_square_relative_error"),
            "group_velocity_error_power": fitted_power(
                rows, "group_velocity_relative_error"),
        })

    clifford_rows = []
    identity4 = np.eye(4, dtype=complex)
    for flavor, basis in dirac.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        square_defects = [float(np.linalg.norm(matrix @ matrix - identity4))
                          for matrix in alpha]
        anticommutator_defects = []
        for first in range(3):
            for second in range(first + 1, 3):
                anticommutator_defects.append(float(np.linalg.norm(
                    alpha[first] @ alpha[second]
                    + alpha[second] @ alpha[first])))
        beta_defects = [float(np.linalg.norm(
            beta @ matrix + matrix @ beta)) for matrix in alpha]
        momentum = np.asarray((0.173, -0.119, 0.071))
        first_symbol, _ = c19_symbol_and_jacobian(momentum)
        spatial = sum(2.0 * first_symbol[axis] * alpha[axis]
                      for axis in range(3))
        clifford_rows.append({
            "flavor": flavor,
            "alpha_square_defects": square_defects,
            "alpha_cross_anticommutator_defects": anticommutator_defects,
            "beta_alpha_anticommutator_defects": beta_defects,
            "sample_spatial_square_defect": float(np.linalg.norm(
                spatial @ spatial
                - 4.0 * float(first_symbol @ first_symbol) * identity4)),
        })

    isotropy_rows = []
    for index, norm in enumerate(MOMENTA):
        values = [row["rows"][index]["lambda_body"]
                  for row in direction_rows]
        isotropy_rows.append({
            "momentum_norm": norm,
            "relative_directional_spread": (
                max(values) - min(values)) / np.mean(values),
        })

    coefficient_rows = []
    for rate in RATES:
        dt = 1.0 / rate
        theta = CHI0 * dt
        value = 2.0 * dt / np.sin(theta)
        coefficient_rows.append({
            "rate": rate,
            "dt": dt,
            "a": value,
            "continuum_limit": 2.0 / CHI0,
            "relative_error": value / (2.0 / CHI0) - 1.0,
        })
    coefficient_power = float(np.polyfit(
        np.log([row["dt"] for row in coefficient_rows]),
        np.log([abs(row["relative_error"]) for row in coefficient_rows]),
        1)[0])

    remainder_powers = [row["relative_remainder_power"]
                        for row in direction_rows]
    square_powers = [row["square_match_error_power"]
                     for row in direction_rows]
    velocity_powers = [row["group_velocity_error_power"]
                       for row in direction_rows]
    clifford_exact = all(
        max(row["alpha_square_defects"]
            + row["alpha_cross_anticommutator_defects"]
            + row["beta_alpha_anticommutator_defects"]
            + [row["sample_spatial_square_defect"]]) < 1e-12
        for row in clifford_rows)
    continuum_compatible = bool(
        all(1.95 < power < 2.05 for power in remainder_powers)
        and all(1.95 < power < 2.05 for power in square_powers)
        and all(1.90 < power < 2.10 for power in velocity_powers)
        and 1.95 < coefficient_power < 2.05
        and isotropy_rows[-1]["relative_directional_spread"] < 1e-4
        and clifford_exact
    )
    result = {
        "status": "V4_C19_TRIANGLE_CONTINUUM_SYMBOL_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "symbol_identity": (
            "lambda_triangle(k)=8[1-cos(kx)cos(ky)cos(kz)] "
            "=4|k|^2+O(|k|^4)"),
        "dirac_square_identity": (
            "[2 alpha.s19(k)]^2=4|s19(k)|^2 I4 exactly, and "
            "lambda_triangle(k)=4|s19(k)|^2+O(|k|^4)"),
        "clifford_rows": clifford_rows,
        "direction_rows": direction_rows,
        "isotropy_rows": isotropy_rows,
        "coefficient_rows": coefficient_rows,
        "coefficient_relative_error_power": coefficient_power,
        "continuum_quadratic_symbol_compatible": continuum_compatible,
        "verdict": (
            "TRIANGLE_TRANSPORT_HAS_ISOTROPIC_QUADRATIC_CONTINUUM_SYMBOL"
            if continuum_compatible else
            "TRIANGLE_TRANSPORT_CONTINUUM_SYMBOL_GATE_FAIL"),
        "dirac_answer": (
            "The spatial continuum gate passes more strongly than a generic "
            "wave check: the unchanged cubical alpha matrices square the "
            "exact C19 first-order symbol, while the triangle action tends to "
            "that square with O(k^4) absolute error; massive dispersion and "
            "group velocity therefore converge to the same Dirac values. A "
            "full Dirac result is not yet proved because invariance of this "
            "same four-component sector and equality of its action-derived "
            "Noether current remain open for the full live-Psi/live-chi "
            "triangle tangent."),
        "claim_boundary": (
            "Analytic Clifford-square, low-momentum dispersion/group-velocity, "
            "and timestep-coefficient audit. The alpha/beta matrices are the "
            "pre-existing cubical readout and are not inserted into the "
            "action. This is not full-tangent sector invariance, Noether-current "
            "equality, nonlinear movement, or a V4-006 pass."),
    }
    output = HERE / "c19_triangle_continuum_symbol_audit.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "remainder_powers": remainder_powers,
        "square_match_error_powers": square_powers,
        "group_velocity_error_powers": velocity_powers,
        "clifford_exact": clifford_exact,
        "coefficient_relative_error_power": coefficient_power,
        "finest_isotropy_spread": isotropy_rows[-1][
            "relative_directional_spread"],
        "output": str(output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
