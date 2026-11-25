"""Audit the complete cubical Hodge/Clifford carrier against the reduced one-form chart.

This calculation changes no action.  It asks a narrower algebraic question:
which cubical degrees must be retained before a native mass grading can exist?
The answer is exact.  The scalar-plus-vector Hodge chart is a single chiral
Cl(3) representation and admits no complex-linear beta.  The complete
0/1/2/3 cubical cochain space is the eight-corner representation already used
by the Lean derivation; degree parity supplies beta and the native commutant
splits it into the two four-component copies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubical_edge_shift_channels import beta, flavor, gamma


HERE = Path(__file__).resolve().parent
TOL = 1.0e-12


def matrix_rank(a: np.ndarray, tol: float = TOL) -> int:
    return int(np.linalg.matrix_rank(a, tol=tol))


def exterior_gamma(axis: int) -> np.ndarray:
    """Wedge plus contraction on masks ordered as cube binary corners."""
    result = np.zeros((8, 8), dtype=complex)
    # CubicalClifford orders the physical x/y/z bits from most to least
    # significant.  The Koszul sign counts occupied physical axes preceding
    # the one being inserted/removed.
    bit = 2 - axis
    preceding_mask = sum(1 << (2 - earlier) for earlier in range(axis))
    for mask in range(8):
        target = mask ^ (1 << bit)
        sign = -1 if ((mask & preceding_mask).bit_count() % 2) else 1
        result[target, mask] = sign
    return result


def exterior_beta() -> np.ndarray:
    return np.diag([
        1 if mask.bit_count() % 2 == 0 else -1 for mask in range(8)
    ]).astype(complex)


def reduced_hodge_alpha() -> list[np.ndarray]:
    # Coordinates are (tau, W_x, W_y, W_z).  These are i times the
    # scalar/vector Hodge matrices in grad/div/curl convention.
    matrices = [
        [[0, -1, 0, 0], [1, 0, 0, 0],
         [0, 0, 0, -1], [0, 0, 1, 0]],
        [[0, 0, -1, 0], [0, 0, 0, 1],
         [1, 0, 0, 0], [0, -1, 0, 0]],
        [[0, 0, 0, -1], [0, 0, -1, 0],
         [0, 1, 0, 0], [1, 0, 0, 0]],
    ]
    return [1j * np.asarray(a, dtype=complex) for a in matrices]


def anticommutant_nullity(generators: list[np.ndarray]) -> int:
    n = generators[0].shape[0]
    identity = np.eye(n, dtype=complex)
    blocks = [
        np.kron(g.T, identity) + np.kron(identity, g)
        for g in generators
    ]
    constraint = np.concatenate(blocks, axis=0)
    return n * n - matrix_rank(constraint)


def relative_defect(a: np.ndarray, scale: float) -> float:
    return float(np.linalg.norm(a) / max(scale, 1.0e-30))


def main() -> None:
    identity8 = np.eye(8, dtype=complex)
    exterior = [exterior_gamma(axis) for axis in range(3)]
    corner = [gamma(axis) for axis in range(3)]
    degree_beta = exterior_beta()
    corner_beta = beta()
    copy = flavor()

    generator_rows = []
    for axis in range(3):
        generator_rows.append({
            "axis": axis,
            "exterior_equals_corner_defect": float(np.linalg.norm(
                exterior[axis] - corner[axis])),
            "square_defect": float(np.linalg.norm(
                exterior[axis] @ exterior[axis] - identity8)),
            "hermiticity_defect": float(np.linalg.norm(
                exterior[axis].conj().T - exterior[axis])),
            "beta_anticommutator_defect": float(np.linalg.norm(
                degree_beta @ exterior[axis]
                + exterior[axis] @ degree_beta)),
            "flavor_commutator_defect": float(np.linalg.norm(
                copy @ exterior[axis] - exterior[axis] @ copy)),
        })

    cross_rows = []
    for first in range(3):
        for second in range(first + 1, 3):
            cross_rows.append({
                "axes": [first, second],
                "anticommutator_defect": float(np.linalg.norm(
                    exterior[first] @ exterior[second]
                    + exterior[second] @ exterior[first])),
            })

    projectors = {
        "minus": (identity8 - copy) / 2,
        "plus": (identity8 + copy) / 2,
    }
    copy_rows = {}
    for name, projector in projectors.items():
        copy_rows[name] = {
            "rank": matrix_rank(projector),
            "projector_defect": float(np.linalg.norm(
                projector @ projector - projector)),
            "beta_leakage": float(np.linalg.norm(
                (identity8 - projector) @ degree_beta @ projector)),
            "transport_leakage_by_axis": [
                float(np.linalg.norm(
                    (identity8 - projector) @ a @ projector))
                for a in exterior
            ],
        }

    reduced = reduced_hodge_alpha()
    reduced_identity = np.eye(4, dtype=complex)
    reduced_cross = [
        float(np.linalg.norm(
            reduced[first] @ reduced[second]
            + reduced[second] @ reduced[first]))
        for first in range(3) for second in range(first + 1, 3)
    ]
    reduced_volume = reduced[0] @ reduced[1] @ reduced[2]
    reduced_nullity = anticommutant_nullity(reduced)

    full_nullity = anticommutant_nullity(exterior)
    complete_pass = (
        max(row["exterior_equals_corner_defect"] for row in generator_rows) < TOL
        and max(row["square_defect"] for row in generator_rows) < TOL
        and max(row["beta_anticommutator_defect"] for row in generator_rows) < TOL
        and max(row["flavor_commutator_defect"] for row in generator_rows) < TOL
        and max(row["anticommutator_defect"] for row in cross_rows) < TOL
        and float(np.linalg.norm(degree_beta - corner_beta)) < TOL
        and all(row["rank"] == 4 for row in copy_rows.values())
        and reduced_nullity == 0
    )

    result = {
        "status": "V4_COMPLETE_CUBICAL_HODGE_CLIFFORD_AUDIT",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "basis": {
            "complete": "eight cube masks = exterior degrees 0,1,2,3",
            "reduced": "relative tau plus W = scalar/vector chiral chart",
        },
        "complete_cubical_space": {
            "dimension_complex": 8,
            "generators": generator_rows,
            "cross_axis_checks": cross_rows,
            "degree_beta_matches_corner_parity_defect": float(np.linalg.norm(
                degree_beta - corner_beta)),
            "beta_square_defect": float(np.linalg.norm(
                degree_beta @ degree_beta - identity8)),
            "beta_flavor_commutator_defect": float(np.linalg.norm(
                degree_beta @ copy - copy @ degree_beta)),
            "beta_anticommutant_dimension": full_nullity,
            "flavor_square_defect": float(np.linalg.norm(
                copy @ copy - identity8)),
            "four_component_copies": copy_rows,
        },
        "reduced_scalar_vector_space": {
            "dimension_complex": 4,
            "generator_square_defects": [
                float(np.linalg.norm(a @ a - reduced_identity)) for a in reduced
            ],
            "cross_axis_anticommutator_defects": reduced_cross,
            "oriented_volume_minus_i_identity_defect": float(np.linalg.norm(
                reduced_volume - 1j * reduced_identity)),
            "beta_anticommutant_dimension": reduced_nullity,
        },
        "verdict": (
            "COMPLETE_CUBICAL_COCHAINS_SUPPLY_BETA_AND_TWO_DIRAC_COPIES"
            if complete_pass else "COMPLETE_CUBICAL_CLIFFORD_AUDIT_FAILED"
        ),
        "root_cause": (
            "The tau/W readout identifies two-form and three-form information "
            "with a scalar/vector chart.  That selects one spatial chirality. "
            "A single complex chiral Cl(3) representation has no nonzero "
            "complex-linear matrix that anticommutes with all three transport "
            "generators, so it cannot carry the Dirac mass grading.  Keeping "
            "all cube degrees restores corner parity beta and leaves two "
            "closed rank-four flavor copies."
        ),
        "action_implication": (
            "The relative tau/W Hodge response may remain a diagnostic or one "
            "chiral block, but it cannot by itself close V4-006.  The next "
            "native response calculation must retain the full eight-corner "
            "cochain information (or an exactly equivalent native channel) "
            "and test whether the current action transports both parity grades."
        ),
        "claim_boundary": (
            "This is an exact cube-geometry identity.  It derives no new "
            "action coefficient and does not prove that the current nonlinear "
            "Psi/chi action dynamically realizes the complete cochain "
            "transport on the localized carrier."
        ),
    }
    output = HERE / "cube_incidence_full_hodge_clifford.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
