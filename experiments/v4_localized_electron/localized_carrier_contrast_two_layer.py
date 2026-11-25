"""Apply the native two-layer update to the carrier-contrast envelope.

The spatial audit shows that the exact carrier-minus-vacuum profile spans all
three physical translation tangents only when both cubical flavor copies are
kept.  This script asks the next question: does the actual implicit native
update preserve that neighboring-center envelope after its previous layer is
derived from the discrete Euler--Lagrange recurrence?

The side-three periodic box is a cheap algebraic closure control.  It is not a
finite moving-carrier or large-box result.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import edge_snap_action_rank as rank
import localized_carrier_contrast_translation as contrast
import localized_embedding_native_closure as local
import localized_embedding_translation_tangent as tangent
import localized_embedding_two_layer_closure as two_layer


HERE = Path(__file__).resolve().parent
SIDE = 3
ORIGIN = SIDE // 2 - 1


def centers() -> list[tuple[int, int, int]]:
    return [(0, 0, 0)] + [
        tuple(sign if axis == index else 0 for index in range(3))
        for axis in range(3) for sign in (-1, 1)
    ]


def profile_columns(
        basis: np.ndarray, psi_profile: np.ndarray,
        chi_profile: np.ndarray,
        offset: tuple[int, int, int]) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for component in range(basis.shape[1]):
        dpsi = np.zeros((SIDE, SIDE, SIDE, 3), dtype=complex)
        dchi = np.zeros_like(dpsi)
        for corner, bits in enumerate(itertools.product((0, 1), repeat=3)):
            site = tuple((ORIGIN + offset[axis] + bits[axis]) % SIDE
                         for axis in range(3))
            amplitude = basis[corner, component]
            dpsi[site] += amplitude * psi_profile[corner]
            dchi[site] += amplitude * chi_profile[corner]
        columns.extend((
            rank.pack(dpsi, dchi),
            rank.pack(1j * dpsi, 1j * dchi),
        ))
    return columns


def envelope(bases: list[np.ndarray]) -> np.ndarray:
    psi_profile, chi_profile = contrast.carrier_contrast()
    raw = np.column_stack([
        column
        for basis in bases
        for center in centers()
        for column in profile_columns(
            basis, psi_profile, chi_profile, center)
    ])
    return tangent.orth(raw)


def translation_matrix() -> np.ndarray:
    amplitude_sq = rank.R**2 / (2 + 54 / rank.R**2)
    psi, chi = rank.native.carrier_fields(SIDE, amplitude_sq)
    columns = []
    for axis in range(3):
        dpsi = (np.roll(psi, 1, axis=axis)
                - np.roll(psi, -1, axis=axis)) / 2
        dchi = (np.roll(chi, 1, axis=axis)
                - np.roll(chi, -1, axis=axis)) / 2
        vector = rank.pack(dpsi, dchi)
        vector /= np.linalg.norm(vector)
        columns.append(vector)
    return np.column_stack(columns)


def translation_capture(q: np.ndarray) -> dict:
    translations = translation_matrix()
    defects = []
    for axis in range(3):
        vector = translations[:, axis]
        defects.append(float(np.linalg.norm(vector-q @ (q.T @ vector))))
    return {
        "axis_relative_defects": defects,
        "worst_axis_relative_defect": float(max(defects)),
    }


def joint_singularity_diagnostic(
        forward: np.ndarray, backward: np.ndarray,
        q: np.ndarray) -> dict:
    complement = np.eye(q.shape[0]) - q @ q.T
    a = complement @ backward @ q
    target = -complement @ forward @ q
    c, _, _, singular = np.linalg.lstsq(a, target, rcond=1e-12)
    tolerance = 1e-12 * max(float(singular[0]), 1.0)
    _, _, vh = np.linalg.svd(a, full_matrices=True)
    rank_a = int(np.sum(singular > tolerance))
    null = vh[rank_a:].T
    coordinates = q.T @ translation_matrix()
    if null.shape[1]:
        fractions = [
            float(np.linalg.norm(null.T @ coordinates[:, axis])
                  / max(np.linalg.norm(coordinates[:, axis]), 1e-30))
            for axis in range(3)
        ]
    else:
        fractions = [0.0, 0.0, 0.0]

    cross = a @ c - target
    psi_power = 0.0
    chi_power = 0.0
    psi_normal = 0.0
    chi_normal = 0.0
    for column in cross.T:
        psi, chi = rank.native.unpack(column, SIDE)
        psi_power += float(np.sum(abs(psi)**2))
        chi_power += float(np.sum(abs(chi)**2))
        psi_normal += float(np.sum(abs(psi[..., 2])**2))
        chi_normal += float(np.sum(abs(chi[..., 2])**2))
    total = max(psi_power + chi_power, 1e-30)
    return {
        "cross_block_operator_rank": rank_a,
        "previous_layer_underdetermined_dimension": int(null.shape[1]),
        "translation_coordinate_nullspace_fractions": fractions,
        "cross_block_raw_coordinate_fractions": {
            "Psi": float(psi_power / total),
            "chi": float(chi_power / total),
            "Psi_third_component": float(psi_normal / total),
            "chi_vacuum_normal_component": float(chi_normal / total),
        },
        "least_norm_previous_graph_condition": float(np.linalg.cond(c)),
    }


def regularized_native_graph(
        forward: np.ndarray, backward: np.ndarray,
        q: np.ndarray) -> dict:
    """Resolve the leakage-null directions with the native rest time graph."""
    complement = np.eye(q.shape[0]) - q @ q.T
    a = complement @ backward @ q
    target = -complement @ forward @ q
    _, theta = two_layer.rest.orbit_layers()
    following_rotation = two_layer.relative.raw_rotation_matrix(
        SIDE, theta, 0.0, 1)
    native_previous = following_rotation.T
    c0 = q.T @ native_previous @ q
    ata = a.T @ a
    at_target = a.T @ target
    scale = max(float(np.linalg.norm(a, 2)**2), 1e-30)
    rows = []

    def classify(label: str, c: np.ndarray, regularization: float | None):
        condition = float(np.linalg.cond(c))
        cross = a @ c - target
        cross_relative = float(
            np.linalg.norm(cross) / max(np.linalg.norm(target), 1e-30))
        if not np.isfinite(condition) or condition > 1e14:
            full = None
            radius = None
        else:
            generator = np.linalg.inv(c)
            top = forward @ q + backward @ q @ c
            full = float(np.linalg.norm(top-q @ generator)
                         / max(np.linalg.norm(top), 1e-30))
            radius = float(max(abs(np.linalg.eigvals(generator))))
        rows.append({
            "label": label,
            "regularization_relative_to_cross_scale": regularization,
            "cross_block_relative_residual": cross_relative,
            "full_invariance_relative_residual": full,
            "previous_layer_map_condition": condition,
            "retained_generator_spectral_radius": radius,
            "distance_from_native_rest_graph": float(
                np.linalg.norm(c-c0) / max(np.linalg.norm(c0), 1e-30)),
        })

    classify("native_rest_graph", c0, None)
    identity = np.eye(q.shape[1])
    for exponent in range(-16, 3):
        relative_lambda = 10.0**exponent
        lam = relative_lambda * scale
        c = np.linalg.solve(
            ata + lam * identity,
            at_target + lam * c0)
        classify(
            f"native_regularized_1e{exponent}", c, relative_lambda)
    finite = [row for row in rows
              if row["full_invariance_relative_residual"] is not None]
    best = min(
        finite,
        key=lambda row: row["full_invariance_relative_residual"])
    return {
        "native_rest_graph_projection_condition": float(np.linalg.cond(c0)),
        "rows": rows,
        "best": best,
        "selection_rule": (
            "minimize full native two-layer invariance residual over a "
            "declared Tikhonov sweep whose only reference is the exact "
            "carrier rest-layer rotation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=two_layer.rest.DT)
    parser.add_argument(
        "--output", default="localized_carrier_contrast_two_layer.json")
    args = parser.parse_args()
    two_layer.rest.DT = args.dt
    rank.lock_inertia()
    forward, backward, diagnostics = two_layer.native_update_blocks()
    flavors = local.flavor_bases()
    families = {
        "plus": [flavors["plus"]],
        "minus": [flavors["minus"]],
        "both_copies": [flavors["plus"], flavors["minus"]],
    }
    rows = []
    for name, bases in families.items():
        q = envelope(bases)
        closure = two_layer.solve_graph(forward, backward, q)
        full_residual = closure["full_invariance_relative_residual"]
        closure["previous_layer_graph_singular"] = not np.isfinite(
            full_residual)
        if not np.isfinite(full_residual):
            closure["full_invariance_relative_residual"] = None
        closure.update({
            "family": name,
            "configuration_real_rank": int(q.shape[1]),
            "translation_capture": translation_capture(q),
        })
        rows.append(closure)
    joint = next(row for row in rows if row["family"] == "both_copies")
    joint_full = joint["full_invariance_relative_residual"]
    joint_q = envelope(families["both_copies"])
    singularity = joint_singularity_diagnostic(
        forward, backward, joint_q)
    native_graph = regularized_native_graph(
        forward, backward, joint_q)
    result = {
        "status": "V4_LOCALIZED_CARRIER_CONTRAST_TWO_LAYER_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "native_action": (
            "transfer-softened incumbent plus transfer-activated chi bond "
            "and 1/(2 chi0) native frame reflection"),
        "box": "side-three periodic algebraic control",
        "dt": float(args.dt),
        "update_diagnostics": diagnostics,
        "rows": rows,
        "joint_singularity_diagnostic": singularity,
        "joint_native_graph_resolution": native_graph,
        "verdict": (
            "BOTH_COPY_CONTRAST_TWO_LAYER_CLOSURE_PASS"
            if (joint["translation_capture"]["worst_axis_relative_defect"]
                < 1e-10
                and joint_full is not None and joint_full < 1e-8)
            else "BOTH_COPY_CONTRAST_TWO_LAYER_CLOSURE_OPEN"),
        "claim_boundary": (
            "This is a linear side-three periodic invariant-graph test.  It "
            "does not establish nonlinear finite translation, large-box "
            "localization, bond-current agreement, or rest stability."),
        "corrective_action": (
            "If the translation-containing envelope is not invariant, "
            "decompose the two-layer cross block into Psi companion, chi "
            "normal, and exterior channels before changing the action.  If "
            "it is invariant, derive the bond current on the retained graph "
            "and repeat with matrix-free side-five response."),
    }
    output = HERE / args.output
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
