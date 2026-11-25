"""Response-rank audit for direct same-component C19 communication.

The candidate basis never mixes internal component labels.  It contains the
lowest position and velocity bond squares for each native field, C3 component,
and cubic face/edge bond orbit.  The exact one-cell DEL defect is used only as
an external response target; it is never inserted into an action.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import edge_snap_action_rank as rank


HERE = Path(__file__).resolve().parent


def class_dirichlet(value: np.ndarray, taxicab: int) -> np.ndarray:
    return sum(
        weight * (value - rank.native.shift(value, offset))
        for offset, weight in rank.native.DIRECTIONS
        if sum(abs(entry) for entry in offset) == taxicab)


def class_weight(taxicab: int) -> float:
    return float(sum(
        weight for offset, weight in rank.native.DIRECTIONS
        if sum(abs(entry) for entry in offset) == taxicab))


def component_pack(template, field_index: int, component: int,
                   values: np.ndarray) -> np.ndarray:
    fields = [np.zeros_like(template[0]), np.zeros_like(template[1])]
    fields[field_index][..., component] = values
    return rank.pack(*fields)


def position_column(layers, dt: float, field_index: int,
                    component: int, taxicab: int) -> np.ndarray:
    current = layers[1]
    value = current[field_index][..., component]
    response = -(2 * dt / class_weight(taxicab)) * class_dirichlet(
        value, taxicab)
    return component_pack(current, field_index, component, response)


def kinetic_column(layers, dt: float, field_index: int,
                   component: int, taxicab: int) -> np.ndarray:
    previous, current, following = layers
    velocity_minus = (
        current[field_index][..., component]
        - previous[field_index][..., component]) / dt
    velocity_plus = (
        following[field_index][..., component]
        - current[field_index][..., component]) / dt
    inertia = 1.0 if field_index == 0 else rank.B
    response = (2 * inertia / class_weight(taxicab)) * (
        class_dirichlet(velocity_minus, taxicab)
        - class_dirichlet(velocity_plus, taxicab))
    return component_pack(current, field_index, component, response)


def fit(matrix: np.ndarray, target: np.ndarray,
        names: list | None = None) -> dict:
    scale = np.linalg.norm(matrix, axis=0)
    active = scale > 1e-13
    normalized = matrix[:, active] / scale[active]
    coefficients, _, _, _ = np.linalg.lstsq(
        normalized, target, rcond=1e-12)
    prediction = normalized @ coefficients
    residual = np.linalg.norm(prediction - target) / np.linalg.norm(target)
    nonnegative, nonnegative_norm = nnls(
        normalized, target, maxiter=10000)
    physical = np.zeros(matrix.shape[1])
    physical[active] = coefficients / scale[active]
    positive_physical = np.zeros(matrix.shape[1])
    positive_physical[active] = nonnegative / scale[active]
    labels = names if names is not None else list(range(matrix.shape[1]))
    return {
        "active_columns": int(np.sum(active)),
        "rank": int(np.linalg.matrix_rank(normalized, tol=1e-10)),
        "relative_residual": float(residual),
        "explained_squared_fraction": float(1 - residual**2),
        "coefficients": {
            str(name): float(value) for name, value in zip(labels, physical)
            if abs(value) > 1e-12},
        "nonnegative_relative_residual": float(
            nonnegative_norm / np.linalg.norm(target)),
        "nonnegative_explained_squared_fraction": float(
            1 - (nonnegative_norm / np.linalg.norm(target))**2),
        "nonnegative_coefficients": {
            str(name): float(value)
            for name, value in zip(labels, positive_physical)
            if value > 1e-12},
    }


def tied_matrix(matrix: np.ndarray, labels: list[dict], key_names) -> tuple:
    groups = []
    names = []
    for label in labels:
        key = tuple(label[name] for name in key_names)
        if key not in names:
            names.append(key)
    for key in names:
        indices = [index for index, label in enumerate(labels)
                   if tuple(label[name] for name in key_names) == key]
        groups.append(np.sum(matrix[:, indices], axis=1))
    return np.column_stack(groups), names


def main() -> None:
    rank.lock_inertia()
    evidence = json.loads((HERE / "edge_snap_verlet_gate.json").read_text(
        encoding="utf-8"))
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(rank.SIDE, amplitude_sq)
    force_psi, force_chi = rank.native.force(psi, chi)
    labels = []
    for kind in ("position", "kinetic"):
        for field_index, field in enumerate(("Psi", "chi")):
            for component in range(3):
                for taxicab, bond in ((1, "face"), (2, "edge")):
                    labels.append({
                        "kind": kind, "field": field,
                        "field_index": field_index,
                        "component": component,
                        "taxicab": taxicab, "bond": bond})

    matrices = []
    targets = []
    rows = []
    for row in evidence["rows"]:
        dt = row["best_dt"]
        layers = rank.translated_layers(row, psi, chi)
        residual = rank.canonical_residual(
            *layers, dt, force_psi, force_chi)
        columns = []
        for label in labels:
            function = (position_column if label["kind"] == "position"
                        else kinetic_column)
            columns.append(function(
                layers, dt, label["field_index"], label["component"],
                label["taxicab"]))
        matrix = np.column_stack(columns)
        norm = np.linalg.norm(residual)
        matrices.append(matrix / norm)
        targets.append(-residual / norm)
        rows.append({
            "axis": row["axis"], "sign": row["sign"],
            "canonical_DEL_norm": float(norm),
            "basis_rank": int(np.linalg.matrix_rank(matrix, tol=1e-10)),
        })

    full_matrix = np.vstack(matrices)
    target = np.concatenate(targets)
    cubic_matrix, cubic_names = tied_matrix(
        full_matrix, labels, ("kind", "field", "bond"))
    field_matrix, field_names = tied_matrix(
        full_matrix, labels, ("kind", "field"))
    result = {
        "status": "V4_DIRECT_PRONG_RESPONSE_RANK",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target_pattern": (
            "exact same-assignment one-cell translation in both signs of "
            "all three cubic axes"),
        "candidate_basis": (
            "positive position and velocity C19 bond squares with each "
            "Psi/chi component connected only to the same component"),
        "rows": rows,
        "fully_component_resolved": fit(
            full_matrix, target,
            [f"{x['kind']}:{x['field']}:{x['component']}:{x['bond']}"
             for x in labels]),
        "cubic_tied_components": {
            "groups": [list(name) for name in cubic_names],
            **fit(cubic_matrix, target, [str(name) for name in cubic_names]),
        },
        "face_edge_tied": {
            "groups": [list(name) for name in field_names],
            **fit(field_matrix, target, [str(name) for name in field_names]),
        },
        "verdict": "DIRECT_PRONG_RESPONSE_SPAN_DIAGNOSED",
        "claim_boundary": (
            "An unconstrained response projection is not an admitted action. "
            "It diagnoses whether this action family can supply the missing "
            "DEL response before any coefficient or nonlinear activation is "
            "proposed."),
    }
    path = HERE / "direct_prong_response_rank.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
