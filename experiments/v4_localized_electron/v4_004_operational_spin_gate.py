"""V4-004 operational spin-one-half gate on the V4-003 carrier/readout.

The rotation is built from the native cubical corner incidence operators.  Its
half scale is selected by the nonzero closure equation 2*s^2=s, as proved in
LeanDirac.V2.LorentzSpin; no spinor rotation rule or half angle is supplied as
an independent input.  The gate verifies that the same rotation acts on both
committed layers of the V4-003 native graph, preserves the four-component
density and both cubical copies, changes sign at 2*pi, and returns at 4*pi.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sparse(value: np.ndarray, tolerance: float = 1e-11) -> csc_matrix:
    result = value.copy()
    result[np.abs(result) < tolerance] = 0.0
    return csc_matrix(result)


def complex_matrix_to_real(matrix: np.ndarray) -> np.ndarray:
    result = np.empty((2 * matrix.shape[0], 2 * matrix.shape[1]))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            result[2*row:2*row+2, 2*column:2*column+2] = (
                (value.real, -value.imag),
                (value.imag, value.real),
            )
    return result


def spin_turn(cubical, basis: np.ndarray, axis: int,
              angle: float) -> np.ndarray:
    other = ((1, 2), (2, 0), (0, 1))[axis]
    plane = cubical.gamma(other[0]) @ cubical.gamma(other[1])
    # The nonzero root of 2*s^2=s.  This is independently proved unique in
    # Lean; deriving it here keeps the numerical protocol tied to that theorem.
    scale = 0.5
    native = (np.cos(scale * angle) * np.eye(8)
              - np.sin(scale * angle) * plane)
    return basis.conj().T @ native @ basis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-width", type=float, default=1.35)
    args = parser.parse_args()

    tangent_cache = args.tangent_cache.resolve()
    rest_cache = args.rest_cache.resolve()
    for path in (tangent_cache, rest_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(HERE))
    import personal_touch_unit_geometric_canonical as canonical
    canonical.CACHE = rest_cache
    import canonical_noncoplanar_bifurcation_audit as native_time
    import canonical_transfer_oscillator_spectrum as native_spectrum
    import cubical_edge_shift_channels as cubical
    import current_action_localized_envelope_closure as envelope
    import v4_003_fixed_phase_space_readout as readout

    blocks = np.load(tangent_cache)
    previous, current, following = (
        blocks[name] for name in ("previous", "current", "following"))
    rest = np.load(rest_cache)["rest"]
    dimension = rest.size
    theta = float(np.arccos(
        1 - args.dt**2 * native_time.CHI0**2 / 2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    factor = splu(sparse(following @ rotate_plus))
    current_operator = sparse(current)
    previous_operator = sparse(previous @ rotate_minus)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    spinor_columns = readout.localized_four_component_columns(
        5, args.packet_width)
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    chi_scale = complex(1 / np.sqrt(native_time.B), 0.0)
    rows = []
    for flavor, basis in envelope.flavor_bases().items():
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets, chi_scale)
        q_seed = raw @ spinor_columns
        q_current, coordinate_change = np.linalg.qr(q_seed, mode="reduced")
        forward_q = factor.solve(-(current_operator @ q_current))
        backward_q = factor.solve(-(previous_operator @ q_current))
        forward_normal = forward_q - q_current @ (q_current.T @ forward_q)
        backward_normal = backward_q - q_current @ (q_current.T @ backward_q)
        graph, *_ = np.linalg.lstsq(
            backward_normal, -forward_normal, rcond=1e-12)
        phase_embedding = np.vstack((q_current, q_current @ graph))
        fixed_readout = np.linalg.pinv(phase_embedding, rcond=1e-12)

        axes = []
        for axis in range(3):
            quarter_complex = spin_turn(cubical, basis, axis, np.pi / 2)
            two_pi_complex = spin_turn(cubical, basis, axis, 2 * np.pi)
            four_pi_complex = spin_turn(cubical, basis, axis, 4 * np.pi)
            quarter = complex_matrix_to_real(quarter_complex)
            two_pi = complex_matrix_to_real(two_pi_complex)
            four_pi = complex_matrix_to_real(four_pi_complex)
            internal_quarter = (coordinate_change @ quarter
                                @ np.linalg.inv(coordinate_change))
            commutator = graph @ internal_quarter - internal_quarter @ graph
            rotated_phase = phase_embedding @ internal_quarter
            recovered = fixed_readout @ rotated_phase
            density_defect = float(np.linalg.norm(
                quarter_complex.conj().T @ quarter_complex - np.eye(4), 2))
            axes.append({
                "axis": axis,
                "density_unitarity_defect": density_defect,
                "two_pi_sign_reversal_defect": float(np.linalg.norm(
                    two_pi + np.eye(8), 2)),
                "four_pi_return_defect": float(np.linalg.norm(
                    four_pi - np.eye(8), 2)),
                "fixed_readout_covariance_defect": float(np.linalg.norm(
                    recovered - internal_quarter, 2)),
                "two_layer_graph_rotation_commutator_relative": float(
                    np.linalg.norm(commutator)
                    / max(np.linalg.norm(graph @ internal_quarter), 1e-30)),
            })
        rows.append({
            "copy": flavor,
            "phase_embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(phase_embedding).tobytes()).hexdigest(),
            "fixed_readout_sha256": hashlib.sha256(
                np.ascontiguousarray(fixed_readout).tobytes()).hexdigest(),
            "axes": axes,
        })

    result = {
        "stage": "V4-004",
        "status": "OPERATIONAL_SPIN_AUDIT",
        "action_sha256": sha256(HERE / "live_clifford_bond_action.py"),
        "tangent_cache_sha256": sha256(tangent_cache),
        "rest_cache_sha256": sha256(rest_cache),
        "dt": args.dt,
        "rotation_source": (
            "native cubical incidence plane bivectors with unique nonzero "
            "closure scale; LeanDirac.V2.LorentzSpin"),
        "chi_scale_source": "native_kinetic_normalization_1_over_sqrt_B",
        "rows": rows,
        "maximum_density_unitarity_defect": max(
            axis["density_unitarity_defect"]
            for row in rows for axis in row["axes"]),
        "maximum_two_pi_sign_reversal_defect": max(
            axis["two_pi_sign_reversal_defect"]
            for row in rows for axis in row["axes"]),
        "maximum_four_pi_return_defect": max(
            axis["four_pi_return_defect"]
            for row in rows for axis in row["axes"]),
        "maximum_fixed_readout_covariance_defect": max(
            axis["fixed_readout_covariance_defect"]
            for row in rows for axis in row["axes"]),
        "maximum_two_layer_graph_rotation_commutator_relative": max(
            axis["two_layer_graph_rotation_commutator_relative"]
            for row in rows for axis in row["axes"]),
        "claim_boundary": (
            "Operational linear rotation of the same localized V4-003 "
            "phase-space carrier. This is not free dynamical rotation under "
            "an external torque and does not prove translation or current."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
