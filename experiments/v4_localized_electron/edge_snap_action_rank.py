"""Response-rank audit for the literal one-edge carrier translation.

This calculation reuses the variational derivatives already checked in
``gyroscopic_verify.py``, but resets every imported module to the locked
V40/V4 chi inertia ``B = 1197`` and uses the canonical position action.
Historical probes used ``B = 229`` and a diagnostic bond polynomial, so their
reported coefficients are deliberately not reused here.

The target is the exact same-assignment one-edge recurrence recorded by
``edge_snap_verlet_gate.json``.  Dirac matrices, carrier projectors, prepared
corner labels, and target forces do not enter any candidate action column.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls

import field_berry_probe as field
import localized_stability as native
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
R = 19.0
B = 1197.0
KAPPA = 1.0 / 63.0
C19_WEIGHT_SUM = 24.0
SIDE = 5


def lock_inertia() -> None:
    """Undo the historical B=229 import side effect in the probe modules."""
    native.B = B
    relative.B = B
    relative.base.B = B
    field.set_chi_inertia(B)


PARAMETERS = list(inspect.signature(field.all_derivatives).parameters)[5:]
GYROSCOPIC = [
    name for name in PARAMETERS
    if "kinetic" not in name and name != "normalized_area_wave"
]
SYMMETRIC_KINETIC = ["relative_kinetic"] + [
    name for name in PARAMETERS
    if "kinetic" in name or name == "normalized_area_wave"
]


def pack(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    return relative.pack_raw(psi, chi)


def ordinary_derivatives(left, right, dt):
    velocity_psi = (right[0] - left[0]) / dt
    velocity_chi = (right[1] - left[1]) / dt
    return (velocity_psi, B * velocity_chi,
            np.zeros_like(velocity_psi), np.zeros_like(velocity_chi))


def action_increment(left, right, dt, name):
    ordinary = ordinary_derivatives(left, right, dt)
    if name == "relative_kinetic":
        values = relative.kinetic_derivatives(
            *left, *right, dt, 1.0)
    else:
        coefficients = {parameter: 0.0 for parameter in PARAMETERS}
        coefficients[name] = 1.0
        values = field.all_derivatives(
            *left, *right, dt, **coefficients)
    return tuple(value - baseline
                 for value, baseline in zip(values, ordinary))


def action_delta(previous, current, following, dt, name):
    minus = action_increment(previous, current, dt, name)
    plus = action_increment(current, following, dt, name)
    return pack(
        minus[0] - plus[0] + dt * (minus[2] + plus[2]) / 2,
        minus[1] - plus[1] + dt * (minus[3] + plus[3]) / 2,
    )


def canonical_residual(previous, current, following, dt, force_psi,
                       force_chi):
    velocity_minus_psi = (current[0] - previous[0]) / dt
    velocity_plus_psi = (following[0] - current[0]) / dt
    velocity_minus_chi = (current[1] - previous[1]) / dt
    velocity_plus_chi = (following[1] - current[1]) / dt
    return pack(
        velocity_minus_psi - velocity_plus_psi + dt * force_psi,
        B * (velocity_minus_chi - velocity_plus_chi) + dt * B * force_chi,
    )


def translated_layers(row, psi, chi):
    axis = row["axis"]
    sign = row["sign"]

    def translate(value, amount):
        return np.roll(value, amount, axis=axis)

    previous = (
        np.exp(-1j * row["best_phase_psi"]) * translate(psi, -sign),
        np.exp(-1j * row["best_phase_chi"]) * translate(chi, -sign),
    )
    following = (
        np.exp(1j * row["best_phase_psi"]) * translate(psi, sign),
        np.exp(1j * row["best_phase_chi"]) * translate(chi, sign),
    )
    return previous, (psi, chi), following


def assemble(names, rows, psi, chi, force_psi, force_chi):
    matrices = []
    targets = []
    raw_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = translated_layers(row, psi, chi)
        residual = canonical_residual(
            *layers, dt, force_psi, force_chi)
        matrix = np.column_stack([
            action_delta(*layers, dt, name) for name in names
        ])
        norm = np.linalg.norm(residual)
        matrices.append(matrix / norm)
        targets.append(-residual / norm)
        correlations = []
        for column in range(matrix.shape[1]):
            column_norm = np.linalg.norm(matrix[:, column])
            correlations.append(float(
                np.vdot(matrix[:, column], -residual).real
                / max(column_norm * norm, 1e-30)))
        raw_rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "canonical_del_residual_norm": float(norm),
            "basis_rank": int(np.linalg.matrix_rank(matrix, tol=1e-11)),
            "maximum_absolute_single_column_correlation": float(
                max(abs(value) for value in correlations)),
        })
    return matrices, targets, raw_rows


def parity_ratios(matrices, targets, rows):
    basis_even = []
    basis_odd = []
    target_even = []
    target_odd = []
    for axis in range(3):
        indices = [index for index, row in enumerate(rows)
                   if row["axis"] == axis]
        minus, plus = indices
        basis_even.append(matrices[plus] + matrices[minus])
        basis_odd.append(matrices[plus] - matrices[minus])
        target_even.append(targets[plus] + targets[minus])
        target_odd.append(targets[plus] - targets[minus])
    be = np.linalg.norm(np.vstack(basis_even))
    bo = np.linalg.norm(np.vstack(basis_odd))
    te = np.linalg.norm(np.concatenate(target_even))
    to = np.linalg.norm(np.concatenate(target_odd))
    return {
        "basis_even_norm": float(be),
        "basis_odd_norm": float(bo),
        "basis_odd_over_even": float(bo / max(be, 1e-30)),
        "target_even_norm": float(te),
        "target_odd_norm": float(to),
        "target_odd_over_even": float(to / max(te, 1e-30)),
    }


def fit_span(names, matrices, targets, nonnegative=False):
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    column_scale = np.linalg.norm(matrix, axis=0)
    active = column_scale > 1e-14
    normalized = matrix[:, active] / column_scale[active]
    active_names = [name for name, keep in zip(names, active) if keep]
    if nonnegative:
        coefficients, residual_norm = nnls(
            normalized, target, maxiter=10000)
    else:
        coefficients, _, _, _ = np.linalg.lstsq(
            normalized, target, rcond=1e-12)
        residual_norm = np.linalg.norm(normalized @ coefficients - target)
    relative_residual = residual_norm / np.linalg.norm(target)
    physical_coefficients = coefficients / column_scale[active]
    return {
        "active_columns": len(active_names),
        "response_rank": int(np.linalg.matrix_rank(normalized, tol=1e-10)),
        "relative_residual": float(relative_residual),
        "explained_squared_fraction": float(1 - relative_residual**2),
        "coefficients": {
            name: float(value) for name, value in
            zip(active_names, physical_coefficients)
            if abs(value) > 1e-12
        },
    }


def best_single(names, matrices, targets):
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    rows = []
    for index, name in enumerate(names):
        column = matrix[:, index]
        denominator = np.vdot(column, column).real
        if denominator <= 1e-30:
            continue
        coefficient = np.vdot(column, target).real / denominator
        residual = np.linalg.norm(
            coefficient * column - target) / np.linalg.norm(target)
        rows.append({
            "basis": name,
            "coefficient": float(coefficient),
            "relative_residual": float(residual),
        })
    return min(rows, key=lambda row: row["relative_residual"])


def rest_matrix(names, dt, psi, chi):
    theta = float(np.arccos(1 - dt**2 * R**2 / 2))
    previous = (np.exp(-1j * theta) * psi, chi.copy())
    current = (psi, chi)
    following = (np.exp(1j * theta) * psi, chi.copy())
    return np.column_stack([
        action_delta(previous, current, following, dt, name)
        for name in names
    ])


def constrained_fit(names, matrices, targets, rest):
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    scale = np.linalg.norm(matrix, axis=0)
    active = scale > 1e-14
    movement = matrix[:, active] / scale[active]
    rest_normalized = rest[:, active] / scale[active]
    _, singular, vh = np.linalg.svd(rest_normalized, full_matrices=True)
    tolerance = max(rest_normalized.shape) * max(singular[0], 1.0) * 1e-12
    rank = int(np.sum(singular > tolerance))
    nullspace = vh[rank:].T
    reduced, _, _, _ = np.linalg.lstsq(
        movement @ nullspace, target, rcond=1e-12)
    normalized_coefficients = nullspace @ reduced
    residual = np.linalg.norm(
        movement @ normalized_coefficients - target) / np.linalg.norm(target)
    rest_defect = np.linalg.norm(rest_normalized @ normalized_coefficients)
    return {
        "rest_response_rank": rank,
        "rest_preserving_dimension": int(nullspace.shape[1]),
        "movement_relative_residual": float(residual),
        "movement_explained_squared_fraction": float(1 - residual**2),
        "absolute_scaled_rest_defect": float(rest_defect),
    }


def c19_dirichlet(value):
    return native.c19_dirichlet(value)


def filtered_chi_delta(previous, current, following, dt, power):
    velocity_minus = (current[1] - previous[1]) / dt
    velocity_plus = (following[1] - current[1]) / dt

    def operator(value):
        first = c19_dirichlet(value) / C19_WEIGHT_SUM
        if power == 1:
            return first
        return c19_dirichlet(first) / C19_WEIGHT_SUM

    return pack(np.zeros_like(current[0]),
                B * (operator(velocity_minus) - operator(velocity_plus)))


def filtered_metric_screen(rows, psi, chi, force_psi, force_chi):
    matrices = []
    targets = []
    for row in rows:
        dt = row["best_dt"]
        layers = translated_layers(row, psi, chi)
        residual = canonical_residual(
            *layers, dt, force_psi, force_chi)
        matrices.append(np.column_stack([
            filtered_chi_delta(*layers, dt, 1),
            filtered_chi_delta(*layers, dt, 2),
        ]) / np.linalg.norm(residual))
        targets.append(-residual / np.linalg.norm(residual))
    matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    unconstrained, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    unconstrained_residual = np.linalg.norm(
        matrix @ unconstrained - target) / np.linalg.norm(target)

    # The normalized C19 symbol x=lambda/24 lies in [0,4/3].  Positivity of
    # the replacement metric is 1+a*x+b*x^2 >= 0 over that interval.
    x = np.linspace(0.0, 4.0 / 3.0, 4001)

    def objective(coefficients):
        return float(np.sum((matrix @ coefficients - target)**2))

    constraint = {
        "type": "ineq",
        "fun": lambda coefficients: (
            1 + coefficients[0] * x + coefficients[1] * x**2 - 1e-8),
    }
    optimum = minimize(
        objective, np.asarray((-2.0, 1.0)), method="SLSQP",
        constraints=constraint,
        options={"ftol": 1e-14, "maxiter": 2000})
    positive_residual = np.sqrt(optimum.fun) / np.linalg.norm(target)

    geometric = np.asarray((-2.0, 1.0))
    geometric_residual = np.linalg.norm(
        matrix @ geometric - target) / np.linalg.norm(target)
    epsilon_metric = geometric.copy()
    # Adding kappa I changes the total response by kappa times the original
    # chi inertia.  It is evaluated separately below in the full residual.
    epsilon_rows = []
    for row in rows:
        dt = row["best_dt"]
        layers = translated_layers(row, psi, chi)
        baseline = canonical_residual(
            *layers, dt, force_psi, force_chi)
        delta = (geometric[0] * filtered_chi_delta(*layers, dt, 1)
                 + geometric[1] * filtered_chi_delta(*layers, dt, 2))
        velocity_minus = (layers[1][1] - layers[0][1]) / dt
        velocity_plus = (layers[2][1] - layers[1][1]) / dt
        local_delta = pack(
            np.zeros_like(psi),
            KAPPA * B * (velocity_minus - velocity_plus))
        epsilon_rows.append({
            "axis": row["axis"],
            "sign": row["sign"],
            "semidefinite_relative_residual": float(
                np.linalg.norm(baseline + delta) / np.linalg.norm(baseline)),
            "kappa_regularized_relative_residual": float(
                np.linalg.norm(baseline + delta + local_delta)
                / np.linalg.norm(baseline)),
        })
    polynomial = 1 + optimum.x[0] * x + optimum.x[1] * x**2
    return {
        "action_family": (
            "T_chi=(B/2)<chiDot,[I+a D19/24+b (D19/24)^2]chiDot>"),
        "unconstrained_coefficients": [float(value) for value in unconstrained],
        "unconstrained_relative_residual": float(unconstrained_residual),
        "positive_boundary_optimizer_success": bool(optimum.success),
        "positive_boundary_coefficients": [float(value) for value in optimum.x],
        "positive_boundary_relative_residual": float(positive_residual),
        "positive_boundary_minimum_symbol": float(np.min(polynomial)),
        "positive_boundary_minimum_x": float(x[np.argmin(polynomial)]),
        "geometric_factorization": (
            "I-2(D19/24)+(D19/24)^2=(I-D19/24)^2=(A19/24)^2"),
        "geometric_coefficients": [-2.0, 1.0],
        "geometric_relative_residual": float(geometric_residual),
        "geometric_symbol_minimum": 0.0,
        "geometric_zero_mode": "D19 eigenvalue 24",
        "kappa_regularization": KAPPA,
        "direction_rows": epsilon_rows,
        "interpretation": (
            "The C19-average square has the correct even-in-velocity channel, "
            "but without kappa its kinetic metric is singular.  Kappa makes "
            "the update invertible but the fixed-phase exact snap still has "
            "a finite residual.  This is a response oracle, not an admitted "
            "action or a moving solution."),
    }


def main() -> None:
    lock_inertia()
    rows = json.loads(
        (HERE / "edge_snap_verlet_gate.json").read_text(encoding="utf-8"))[
            "rows"]
    amplitude_sq = R**2 / (2 + 54 / R**2)
    psi, chi = native.carrier_fields(SIDE, amplitude_sq)
    force_psi, force_chi = native.force(psi, chi)

    gyro_matrices, gyro_targets, gyro_rows = assemble(
        GYROSCOPIC, rows, psi, chi, force_psi, force_chi)
    kinetic_matrices, kinetic_targets, kinetic_rows = assemble(
        SYMMETRIC_KINETIC, rows, psi, chi, force_psi, force_chi)
    dt = float(np.mean([row["best_dt"] for row in rows]))
    gyro_rest = rest_matrix(GYROSCOPIC, dt, psi, chi)
    kinetic_rest = rest_matrix(SYMMETRIC_KINETIC, dt, psi, chi)

    gyro_fit = fit_span(GYROSCOPIC, gyro_matrices, gyro_targets)
    kinetic_fit = fit_span(
        SYMMETRIC_KINETIC, kinetic_matrices, kinetic_targets)
    kinetic_positive = fit_span(
        SYMMETRIC_KINETIC, kinetic_matrices, kinetic_targets,
        nonnegative=True)

    result = {
        "status": "V4_EDGE_SNAP_ACTION_RESPONSE_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "B": B,
        "historical_B_rejected": 229.0,
        "side": SIDE,
        "amplitude_sq": amplitude_sq,
        "target": (
            "same-assignment one-C19-face-edge shift in both signs of all "
            "three axes, with the fixed phases from edge_snap_verlet_gate"),
        "gyroscopic": {
            "basis_count": len(GYROSCOPIC),
            "basis_names": GYROSCOPIC,
            "direction_rows": gyro_rows,
            "direction_reversal": parity_ratios(
                gyro_matrices, gyro_targets, rows),
            "unconstrained_fit": gyro_fit,
            "exact_rest_constrained_fit": constrained_fit(
                GYROSCOPIC, gyro_matrices, gyro_targets, gyro_rest),
            "verdict": (
                "WRONG_DIRECTION_REVERSAL_PARITY: velocity-linear one-forms "
                "reverse while the required second-order snap response is "
                "even; the complete tested basis explains too little even "
                "before positivity and rest constraints."),
        },
        "symmetric_kinetic": {
            "basis_count": len(SYMMETRIC_KINETIC),
            "basis_names": SYMMETRIC_KINETIC,
            "direction_rows": kinetic_rows,
            "direction_reversal": parity_ratios(
                kinetic_matrices, kinetic_targets, rows),
            "unconstrained_fit": kinetic_fit,
            "nonnegative_additions_fit": kinetic_positive,
            "best_single_basis": best_single(
                SYMMETRIC_KINETIC, kinetic_matrices, kinetic_targets),
            "exact_rest_constrained_fit": constrained_fit(
                SYMMETRIC_KINETIC, kinetic_matrices, kinetic_targets,
                kinetic_rest),
            "verdict": (
                "CORRECT_PARITY_BUT_WRONG_POSITIVE_ADDITION_SIGN: the even "
                "kinetic class contains the missing channel, but an arbitrary "
                "sum of the tested positive squares cannot reduce the defect."
            ),
        },
        "factorized_c19_metric": filtered_metric_screen(
            rows, psi, chi, force_psi, force_chi),
        "root_cause_update": (
            "Literal translation needs an even-in-velocity neighbor momentum "
            "response.  The earlier gyroscopic hypothesis has the wrong "
            "direction-reversal parity.  A factorized C19 kinetic metric "
            "nearly supplies the response without changing the rest orbit, "
            "but its unregularized limit is singular and therefore erases "
            "resistance rather than proving a deterministic native snap."),
        "next_test": (
            "Derive the smallest positive, invertible, field-dependent C19 "
            "kinetic square whose connection force closes the remaining "
            "receiver residual; then derive its implicit reciprocal update "
            "and re-solve the phases and translation time."),
        "claim_boundary": (
            "This is a discrete Euler-Lagrange response-rank calculation, "
            "not a freely moving carrier, action admission, Dirac pass, or "
            "identification of one numerical update with a substrate tick."),
    }
    (HERE / "edge_snap_action_rank.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
