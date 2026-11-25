"""Test the C19-triangle DEL tangent on the frozen carrier embedding.

The continuum-symbol audit proves that the triangle spatial symbol approaches
the square of the existing C19 Dirac symbol.  This script tests the stronger
state-space obligation: the same cubical four-component carrier, including
live chi and the previous committed layer, must be preserved by the full
discrete Euler--Lagrange tangent.

No projector is used during evolution.  Orthogonal projections appear only in
the diagnostic readout.  The action coefficients are the native-inertia
normalization fixed by the rest clock; no Dirac endpoint is fitted.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve, qr

import c27_body_gyro_basis_probe as candidate
import localized_carrier_contrast_inverse_launch as inverse
import localized_carrier_contrast_phase_ramp as ramp


HERE = Path(__file__).resolve().parent
OFFSETS = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))


def raw_envelope(flavor: str, psi_phase: complex) -> np.ndarray:
    """All translated columns of one frozen cubical flavor copy."""
    psi_profile, chi_profile = inverse.contrast.carrier_contrast()
    basis = inverse.local.flavor_bases()[flavor]
    columns = []
    for offset in OFFSETS:
        columns.extend(inverse.profile_columns(
            basis, psi_phase * psi_profile, chi_profile, offset))
    return np.column_stack(columns)


def component_targets(axis: int, psi_phase: complex) -> tuple[list[str], np.ndarray]:
    """Physical unscaled-field version of the cubical component ramps.

    The historical helper unpacked an ordinary raw vector with the
    mass-normalized ``localized_stability.unpack`` routine, shrinking chi by
    ``1/sqrt(B)`` before repacking it.  Here the packing convention stays
    ordinary throughout, as required by the discrete action.
    """
    offset_plus = tuple(1 if index == axis else 0 for index in range(3))
    offset_minus = tuple(-1 if index == axis else 0 for index in range(3))
    labels = []
    columns = []
    for flavor in ("plus", "minus"):
        basis = inverse.local.flavor_bases()[flavor]
        for index in range(4):
            amplitudes = basis[:, index]
            odd = (ramp.placed_copy(amplitudes, offset_plus, psi_phase)
                   - ramp.placed_copy(amplitudes, offset_minus, psi_phase)) / 2
            psi, chi = inverse.rank.relative.unpack_raw(odd, inverse.SIDE)
            columns.append(inverse.rank.pack(1j * psi, 1j * chi))
            labels.append(f"{flavor}_{index}")
    return labels, np.column_stack(columns)


def orth(raw: np.ndarray) -> np.ndarray:
    # Translated carrier columns are strongly dependent.  Pivoting is
    # essential: filtering an unpivoted R diagonal can discard later
    # independent columns and even reject a vector used to build ``raw``.
    q, r, _ = qr(raw, mode="economic", pivoting=True, check_finite=False)
    diagonal = abs(np.diag(r))
    tolerance = 1e-11 * max(float(diagonal[0]), 1.0)
    return q[:, diagonal > tolerance]


def relative_leakage(vector: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    projected = q @ (q.T @ vector)
    leakage = vector - projected
    return (float(np.linalg.norm(leakage)
                  / max(np.linalg.norm(vector), 1e-300)), leakage)


def field_decomposition(raw: np.ndarray) -> dict:
    psi, chi = inverse.rank.native.unpack(raw, inverse.SIDE)
    carrier = np.zeros((inverse.SIDE,) * 3, dtype=bool)
    for bits in itertools.product((0, 1), repeat=3):
        carrier[tuple(inverse.ORIGIN + bit for bit in bits)] = True
    psi_power = inverse.rank.native.norm2(psi)
    chi_power = inverse.rank.B * inverse.rank.native.norm2(chi)
    total = max(float(np.sum(psi_power) + np.sum(chi_power)), 1e-300)
    return {
        "psi_fraction": float(np.sum(psi_power) / total),
        "chi_fraction": float(np.sum(chi_power) / total),
        "carrier_fraction": float(
            (np.sum(psi_power[carrier]) + np.sum(chi_power[carrier])) / total),
        "exterior_fraction": float(
            (np.sum(psi_power[~carrier]) + np.sum(chi_power[~carrier])) / total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=256)
    parser.add_argument("--output", default="c19_triangle_same_embedding_closure.json")
    args = parser.parse_args()
    dt_requested = 1.0 / args.rate
    inverse.DT = dt_requested
    inverse.rest.DT = dt_requested
    inverse.current_audit.DT = dt_requested
    candidate.base.inverse.DT = dt_requested
    candidate.base.inverse.rest.DT = dt_requested
    candidate.base.inverse.current_audit.DT = dt_requested
    candidate.base.inverse.rank.lock_inertia()
    candidate.base.inverse.rest.SIDE = candidate.base.inverse.SIDE
    candidate.base.inverse.rest.DT = candidate.base.inverse.DT
    previous, current, following, theta = candidate.base.inverse.rest_layers()
    tensors = tuple(torch.tensor(value, dtype=torch.float64)
                    for value in (previous, current, following))
    dt = candidate.base.inverse.DT
    a = 2.0 * dt / np.sin(theta)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4.0,
        a - candidate.base.inverse.rank.B / 4.0,
        -a / 2.0,
        0.0,
    ))
    del_residual, right_momentum = candidate.make_derivatives(
        1.0, coefficients)
    candidate.base.del_residual = del_residual
    candidate.base.right_momentum = right_momentum

    rest_residual = del_residual(*tensors).detach().numpy()
    dimension = previous.size
    previous_block = candidate.base.batched_jvp(
        "previous", tensors, np.eye(dimension))
    factor = lu_factor(previous_block)

    envelopes = {}
    for flavor in ("plus", "minus"):
        envelopes[(flavor, "previous")] = orth(raw_envelope(
            flavor, np.exp(-1j * theta)))
        envelopes[(flavor, "following")] = orth(raw_envelope(
            flavor, np.exp(1j * theta)))
    combined_previous = orth(np.column_stack([
        raw_envelope("plus", np.exp(-1j * theta)),
        raw_envelope("minus", np.exp(-1j * theta)),
    ]))

    rows = []
    axis_rows = []
    for axis in range(3):
        labels, targets = component_targets(axis, np.exp(1j * theta))
        images = candidate.base.batched_jvp("following", tensors, targets)
        launches = lu_solve(factor, -images)
        responses = np.column_stack([
            candidate.base.response(
                previous, current, following,
                launches[:, column], targets[:, column])
            for column in range(targets.shape[1])
        ])
        charge_rank = np.linalg.matrix_rank(responses[:2], tol=1e-8)
        fixed_charge_null = np.linalg.svd(
            responses[:2], full_matrices=True)[2][charge_rank:].T
        fixed_charge_center = responses[2:] @ fixed_charge_null
        singular = np.linalg.svd(fixed_charge_center, compute_uv=False)
        threshold = 1e-8 * max(float(singular[0]), 1e-300)
        axis_rows.append({
            "axis": axis,
            "fixed_charge_center_rank": int(np.sum(singular > threshold)),
            "fixed_charge_center_singular_values": [float(value)
                                                     for value in singular],
            "relative_third_singular_value": float(
                singular[2] / max(singular[0], 1e-300)),
        })
        for column, label in enumerate(labels):
            flavor = label.split("_", 1)[0]
            own_previous = envelopes[(flavor, "previous")]
            own_following = envelopes[(flavor, "following")]
            launch_fraction, launch_leakage = relative_leakage(
                launches[:, column], own_previous)
            combined_fraction, _ = relative_leakage(
                launches[:, column], combined_previous)
            target_fraction, _ = relative_leakage(
                targets[:, column], own_following)
            rows.append({
                "axis": axis,
                "component": label,
                "own_copy_previous_layer_leakage": launch_fraction,
                "both_copy_previous_layer_leakage": combined_fraction,
                "own_copy_following_layer_target_leakage": target_fraction,
                "own_copy_launch_leakage_decomposition": field_decomposition(
                    launch_leakage),
            })

    worst_own = max(row["own_copy_previous_layer_leakage"] for row in rows)
    worst_combined = max(
        row["both_copy_previous_layer_leakage"] for row in rows)
    worst_target = max(
        row["own_copy_following_layer_target_leakage"] for row in rows)
    result = {
        "status": "V4_C19_TRIANGLE_SAME_EMBEDDING_CLOSURE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": inverse.SIDE,
        "dt": dt,
        "a": float(a),
        "coefficients": {
            name: float(value)
            for name, value in zip(candidate.BASIS_NAMES, coefficients)
        },
        "coefficient_rule": (
            "native inertia/rest-clock normalization; no Dirac fit"),
        "rest_DEL_rms": float(
            np.linalg.norm(rest_residual) / np.sqrt(rest_residual.size)),
        "envelope_real_ranks": {
            f"{flavor}_{layer}": int(envelopes[(flavor, layer)].shape[1])
            for flavor in ("plus", "minus")
            for layer in ("previous", "following")
        },
        "combined_previous_real_rank": int(combined_previous.shape[1]),
        "corrected_physical_coordinate_axis_rows": axis_rows,
        "rows": rows,
        "worst_own_copy_previous_layer_leakage": worst_own,
        "worst_both_copy_previous_layer_leakage": worst_combined,
        "worst_following_target_embedding_leakage": worst_target,
        "all_axes_rank_three": all(
            row["fixed_charge_center_rank"] == 3 for row in axis_rows),
        "verdict": (
            "TRIANGLE_FULL_TANGENT_PRESERVES_FROZEN_FOUR_COMPONENT_SECTOR"
            if (worst_own < 1e-8 and worst_target < 1e-8
                and all(row["fixed_charge_center_rank"] == 3
                        for row in axis_rows)) else
            "TRIANGLE_FULL_TANGENT_FOUR_COMPONENT_SECTOR_NOT_CLOSED"),
        "claim_boundary": (
            "Linear full-field two-layer closure for the frozen cubical "
            "embedding and directional component ramps. It is not Noether-"
            "current equality, nonlinear endurance, or a V4-006 pass."),
    }
    output = HERE / args.output
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "rest_DEL_rms": result["rest_DEL_rms"],
        "envelope_real_ranks": result["envelope_real_ranks"],
        "combined_previous_real_rank": result["combined_previous_real_rank"],
        "fixed_charge_center_ranks": [
            row["fixed_charge_center_rank"] for row in axis_rows],
        "relative_third_singular_values": [
            row["relative_third_singular_value"] for row in axis_rows],
        "worst_own_copy_previous_layer_leakage": worst_own,
        "worst_both_copy_previous_layer_leakage": worst_combined,
        "worst_following_target_embedding_leakage": worst_target,
        "output": str(output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
