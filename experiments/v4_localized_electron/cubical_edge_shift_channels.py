"""Classify exact edge-hopping channels of the derived cubical algebra.

This is a readout/geometry calculation, not a native action modification.
It answers whether the two four-component flavor copies are the two sides of
an edge hop. They are not: flavor commutes with every spatial incidence
generator. Within each flavor copy, the +/- eigenspaces of a cubical gamma
are the independently conditional forward/backward edge channels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


FLIPS = (
    (4, 5, 6, 7, 0, 1, 2, 3),
    (2, 3, 0, 1, 6, 7, 4, 5),
    (1, 0, 3, 2, 5, 4, 7, 6),
)
BITS = (
    (0, 0, 0, 0, 1, 1, 1, 1),
    (0, 0, 1, 1, 0, 0, 1, 1),
    (0, 1, 0, 1, 0, 1, 0, 1),
)


def gamma(axis: int) -> np.ndarray:
    result = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        if axis == 0:
            sign = 1
        elif axis == 1:
            sign = 1 if BITS[0][corner] == 0 else -1
        else:
            sign = (1 if (BITS[0][corner] + BITS[1][corner]) % 2 == 0
                    else -1)
        result[corner, FLIPS[axis][corner]] = sign
    return result


def beta() -> np.ndarray:
    return np.diag([
        1 if sum(BITS[axis][corner] for axis in range(3)) % 2 == 0 else -1
        for corner in range(8)
    ]).astype(complex)


def flavor() -> np.ndarray:
    result = np.zeros((8, 8), dtype=complex)
    for corner in range(8):
        target = FLIPS[1][FLIPS[2][corner]]
        sign = 1 if BITS[2][corner] == 0 else -1
        result[corner, target] = 1j * sign
    return result


def matrix_rank(matrix: np.ndarray, tolerance: float = 1e-10) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def main() -> None:
    identity = np.eye(8, dtype=complex)
    gammas = [gamma(axis) for axis in range(3)]
    beta_matrix = beta()
    flavor_matrix = flavor()
    flavor_projectors = {
        "minus": (identity - flavor_matrix) / 2,
        "plus": (identity + flavor_matrix) / 2,
    }

    axes = []
    for axis, generator in enumerate(gammas):
        forward = (identity + generator) / 2
        backward = (identity - generator) / 2
        momentum = 0.371
        conditional_shift = (
            np.exp(-1j * momentum) * forward
            + np.exp(1j * momentum) * backward)
        finite_difference_generator = (
            (np.exp(-1j * 1e-7) * forward
             + np.exp(1j * 1e-7) * backward - identity) / 1e-7)
        flavor_rows = {}
        for name, copy in flavor_projectors.items():
            forward_rank = matrix_rank(copy @ forward)
            backward_rank = matrix_rank(copy @ backward)
            flavor_rows[name] = {
                "copy_rank": matrix_rank(copy),
                "forward_rank_inside_copy": forward_rank,
                "backward_rank_inside_copy": backward_rank,
                "one_axis_walk_index": forward_rank - backward_rank,
                "conditional_shift_copy_leakage_norm": float(np.linalg.norm(
                    (identity-copy) @ conditional_shift @ copy)),
            }
        scalar_shift = np.exp(-1j * momentum) * identity
        axes.append({
            "axis": axis,
            "gamma_square_defect": float(np.linalg.norm(
                generator @ generator - identity)),
            "forward_projector_rank": matrix_rank(forward),
            "backward_projector_rank": matrix_rank(backward),
            "whole_conditional_walk_index": (
                matrix_rank(forward) - matrix_rank(backward)),
            "one_way_rigid_shift_index_control": 8,
            "forward_backward_overlap_norm": float(np.linalg.norm(
                forward @ backward)),
            "conditional_shift_unitarity_defect": float(np.linalg.norm(
                conditional_shift.conj().T @ conditional_shift - identity)),
            "small_momentum_generator_defect": float(np.linalg.norm(
                finite_difference_generator + 1j * generator)),
            "scalar_whole_carrier_shift_difference_norm": float(
                np.linalg.norm(conditional_shift - scalar_shift)),
            "flavor_commutator_norm": float(np.linalg.norm(
                flavor_matrix @ generator - generator @ flavor_matrix)),
            "beta_anticommutator_norm": float(np.linalg.norm(
                beta_matrix @ generator + generator @ beta_matrix)),
            "mass_grading_flips_forward_to_backward_defect": float(
                np.linalg.norm(beta_matrix @ forward - backward @ beta_matrix)),
            "flavor_sectors": flavor_rows,
        })

    cross_anticommutators = []
    for first in range(3):
        for second in range(first+1, 3):
            cross_anticommutators.append({
                "axes": [first, second],
                "defect": float(np.linalg.norm(
                    gammas[first] @ gammas[second]
                    + gammas[second] @ gammas[first])),
            })

    result = {
        "status": "V4_CUBICAL_EDGE_SHIFT_CHANNEL_CLASSIFICATION",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "source": "LeanDirac/V2/CubicalClifford.lean",
        "axes": axes,
        "cross_axis_anticommutators": cross_anticommutators,
        "flavor_involution_defect": float(np.linalg.norm(
            flavor_matrix @ flavor_matrix - identity)),
        "flavor_beta_commutator_norm": float(np.linalg.norm(
            flavor_matrix @ beta_matrix - beta_matrix @ flavor_matrix)),
        "index_reference": (
            "Gross-Nesme-Vogts-Werner, Index Theory of One Dimensional "
            "Quantum Walks and Cellular Automata, arXiv:0910.3675"),
        "index_interpretation": (
            "Along every chosen axis each four-component copy has two "
            "right-moving and two left-moving channels, hence zero net walk "
            "index. The conditional shift is therefore in the balanced class "
            "compatible with partitioned local unitaries. A rigid one-way "
            "eight-channel translation has nonzero index and is a different "
            "target. This classification does not prove that the native LFM "
            "action supplies the needed local exchange."),
        "interpretation": (
            "A nearest-neighbor Dirac walk conditionally shifts the +gamma "
            "and -gamma channels in opposite edge directions. Each of the "
            "two four-component flavor copies contains two forward and two "
            "backward channels and is preserved by transport. The two copies "
            "therefore are not a leading/trailing pair."),
        "action_implication": (
            "The failed whole-carrier shift used exp(-ik) times the identity. "
            "The geometry-derived Dirac transport instead uses exp(-ik gamma). "
            "A native repair must generate the conditional incidence response "
            "from Psi-chi geometry; ordinary scalar advection cannot do so."),
        "claim_boundary": (
            "This proves an algebraic edge-shift decomposition of the derived "
            "cubical readout. It does not prove that the current two-field "
            "action dynamically executes that shift or that either copy is a "
            "separate physical electron."),
    }
    (HERE / "cubical_edge_shift_channels.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
