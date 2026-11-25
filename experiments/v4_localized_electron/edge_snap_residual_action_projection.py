"""Project the saved edge-snap residual onto verified native action bases.

This is a response-rank diagnostic.  Coefficients from least squares are not
admitted action constants.  The target residual never enters a force law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import edge_snap_action_rank as rank
import edge_snap_relative_orbit_search as orbit


HERE = Path(__file__).resolve().parent
DT = 0.1
ALL = rank.GYROSCOPIC + [
    name for name in rank.SYMMETRIC_KINETIC if name != "relative_kinetic"
] + ["relative_kinetic"]
ORIENTATION = [
    "triangle_common", "triangle_psi", "triangle_chi",
    "area_plaquette_relative_berry", "area_gradient_kinetic",
    "area_local_kinetic", "area_relative_gradient_kinetic",
    "psi_orientation_kinetic", "chi_orientation_kinetic",
    "normalized_area_common", "normalized_area_relative",
    "normalized_area_bond_common", "normalized_area_bond_relative",
    "normalized_area_wave",
]


def unpack(raw):
    return rank.relative.unpack_raw(raw, orbit.SIDE)


def layers_from_artifact(input_stem):
    data = np.load(HERE / f"{input_stem}.npz")
    q0 = data["q0"]
    middle = list(data["middle"])
    qn = data["qn"]
    qm1 = orbit.translate(
        orbit.torch.tensor(middle[-1], dtype=orbit.torch.float64),
        -orbit.SIGN, -3.3287798462704368, 6.889772046440938e-05,
    ).detach().numpy()
    states = [q0] + middle + [qn]
    triples = [(qm1, q0, middle[0])]
    triples.extend((states[index - 1], states[index], states[index + 1])
                   for index in range(1, 4))
    return [tuple(unpack(value) for value in triple) for triple in triples], data


def response_matrix(names, triples):
    columns = []
    for name in names:
        pieces = [rank.action_delta(*triple, DT, name) for triple in triples]
        columns.append(np.concatenate(pieces))
    return np.column_stack(columns)


def fit(names, matrix, target, nonnegative=False):
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > 1e-12
    normalized = matrix[:, active] / norms[active]
    if nonnegative:
        coefficients, residual_norm = nnls(
            normalized, target, maxiter=10000)
    else:
        coefficients, _, _, _ = np.linalg.lstsq(
            normalized, target, rcond=1e-12)
        residual_norm = np.linalg.norm(normalized @ coefficients - target)
    active_names = [name for name, keep in zip(names, active) if keep]
    fitted = {
        name: float(value / norm)
        for name, value, norm in zip(active_names, coefficients, norms[active])
        if abs(value) > 1e-10
    }
    relative = float(residual_norm / np.linalg.norm(target))
    return {
        "active_columns": int(np.sum(active)),
        "rank": int(np.linalg.matrix_rank(normalized, tol=1e-11)),
        "relative_residual": relative,
        "explained_squared_fraction": float(1 - relative**2),
        "coefficients": fitted,
    }


def rest_constrained_fit(names, matrix, target, psi, chi):
    rest = rank.rest_matrix(names, DT, psi, chi)
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > 1e-12
    movement = matrix[:, active] / norms[active]
    rest_normalized = rest[:, active] / norms[active]
    _, singular, vh = np.linalg.svd(rest_normalized, full_matrices=True)
    tolerance = max(rest_normalized.shape) * max(singular[0], 1.0) * 1e-12
    rest_rank = int(np.sum(singular > tolerance))
    nullspace = vh[rest_rank:].T
    if nullspace.shape[1] == 0:
        return {
            "rest_response_rank": rest_rank,
            "rest_preserving_dimension": 0,
            "relative_residual": 1.0,
            "explained_squared_fraction": 0.0,
        }
    coefficient, _, _, _ = np.linalg.lstsq(
        movement @ nullspace, target, rcond=1e-12)
    residual = np.linalg.norm(
        movement @ nullspace @ coefficient - target) / np.linalg.norm(target)
    return {
        "rest_response_rank": rest_rank,
        "rest_preserving_dimension": int(nullspace.shape[1]),
        "relative_residual": float(residual),
        "explained_squared_fraction": float(1 - residual**2),
    }


def main(input_stem: str = "edge_snap_four_update_orbit",
         output_stem: str = "edge_snap_residual_action_projection") -> None:
    rank.lock_inertia()
    triples, data = layers_from_artifact(input_stem)
    target = -data["residuals"].reshape(-1)
    target /= np.linalg.norm(target)
    matrix = response_matrix(ALL, triples)
    matrix /= np.linalg.norm(data["residuals"])
    index = {name: ALL.index(name) for name in ALL}

    def subset(names):
        return matrix[:, [index[name] for name in names]]

    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(orbit.SIDE, amplitude_sq)
    correlations = {}
    for name in ALL:
        column = matrix[:, index[name]]
        correlations[name] = float(
            np.dot(column, target)
            / max(np.linalg.norm(column) * np.linalg.norm(target), 1e-300))
    strongest = sorted(correlations.items(), key=lambda item: -abs(item[1]))[:10]
    symmetric = rank.SYMMETRIC_KINETIC
    result = {
        "status": "V4_EDGE_SNAP_RESIDUAL_ACTION_PROJECTION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "target": (
            f"saved residual from {input_stem}"),
        "all_verified_additive_actions": fit(ALL, matrix, target),
        "orientation_multisite_actions": fit(
            ORIENTATION, subset(ORIENTATION), target),
        "gyroscopic_actions": fit(
            rank.GYROSCOPIC, subset(rank.GYROSCOPIC), target),
        "positive_symmetric_kinetic_additions": fit(
            symmetric, subset(symmetric), target, nonnegative=True),
        "rest_constrained_all_actions": rest_constrained_fit(
            ALL, matrix, target, psi, chi),
        "strongest_single_correlations": [
            {"action": name, "correlation": value}
            for name, value in strongest],
        "verdict": "RESPONSE_RANK_CLASSIFICATION_ONLY",
        "claim_boundary": (
            "Unrestricted fitted coefficients are inverse diagnostics. They "
            "are not native provenance, positivity proofs, rest-stability "
            "passes, or action admissions."),
    }
    (HERE / f"{output_stem}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-stem", default="edge_snap_four_update_orbit")
    parser.add_argument(
        "--output-stem", default="edge_snap_residual_action_projection")
    args = parser.parse_args()
    main(args.input_stem, args.output_stem)
