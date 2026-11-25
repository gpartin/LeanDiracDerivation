"""V4-005 short-time transport on the frozen V4-003 carrier rule.

The native previous/current layers are prepared from the independent exact
C19/centered-time Dirac packet.  The frozen action tangent advances those
layers once.  A translation-covariant extension of V4-003's local cubical
Psi/chi lift reads the resulting four-component field; it is constructed
once per copy and is never fitted to the target or native response.

Dirac supplies initial data and the comparator only.  It never enters the
native action, tangent, solve, or readout.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
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


def complex_field(real: np.ndarray, side: int = 5) -> np.ndarray:
    paired = real.reshape(-1, 2)
    return (paired[:, 0] + 1j * paired[:, 1]).reshape(
        side, side, side, 4)


def readout_solver(raw: np.ndarray):
    """Fixed minimum-norm local carrier readout without target fitting."""
    gram = raw.T @ raw
    ridge = 1e-13 * max(float(np.linalg.norm(gram, 2)), 1.0)
    factor = cho_factor(gram + ridge * np.eye(gram.shape[0]),
                        check_finite=False)

    def solve(columns: np.ndarray) -> np.ndarray:
        return cho_solve(factor, raw.T @ columns, check_finite=False)

    return solve, ridge, float(np.linalg.cond(gram + ridge*np.eye(
        gram.shape[0])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wave-number", type=float, default=0.35)
    parser.add_argument("--packet-width", type=float, default=1.35)
    args = parser.parse_args()

    tangent_cache = args.tangent_cache.resolve()
    rest_cache = args.rest_cache.resolve()
    sys.path.insert(0, str(HERE))
    import personal_touch_unit_geometric_canonical as canonical
    canonical.CACHE = rest_cache
    import canonical_noncoplanar_bifurcation_audit as native_time
    import canonical_transfer_oscillator_spectrum as native_spectrum
    import cube_incidence_extended_packet_dirac_current as current
    import cube_incidence_extended_packet_gate as dirac
    import current_action_localized_envelope_closure as envelope

    data = np.load(tangent_cache)
    previous, middle, following = (
        data[name] for name in ("previous", "current", "following"))
    rest = np.load(rest_cache)["rest"]
    dimension = rest.size
    theta = float(np.arccos(
        1 - args.dt**2 * native_time.CHI0**2 / 2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    factor = splu(sparse(following @ rotate_plus))
    middle_operator = sparse(middle)
    previous_operator = sparse(previous @ rotate_minus)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    chi_scale = complex(1 / np.sqrt(native_time.B), 0.0)
    rows = []
    readouts = {}
    for flavor, basis in envelope.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets, chi_scale)
        solve_readout, ridge, condition = readout_solver(raw)
        readouts[flavor] = {
            "raw_sha256": hashlib.sha256(
                np.ascontiguousarray(raw).tobytes()).hexdigest(),
            "ridge": ridge,
            "regularized_gram_condition": condition,
        }
        native_columns = []
        target_columns = []
        metadata = []
        for axis, sign in itertools.product(range(3), (-1, 1)):
            packet_previous, packet_current, packet_target, momentum = (
                dirac.packet(
                    axis, sign, alpha, beta, args.dt,
                    width=args.packet_width,
                    carrier_wave_number=args.wave_number,
                    centered_time=True,
                    frequency_branch="negative",
                    transport_scale=1.0))
            q_previous = raw @ dirac.real_coordinates(packet_previous)
            q_current = raw @ dirac.real_coordinates(packet_current)
            q_target = raw @ dirac.real_coordinates(packet_target)
            q_native = factor.solve(-(
                middle_operator @ q_current
                + previous_operator @ q_previous))
            native_columns.append(q_native)
            target_columns.append(q_target)
            metadata.append((axis, sign, momentum,
                             packet_current, packet_target))

        native_matrix = np.column_stack(native_columns)
        recovered = solve_readout(native_matrix)
        reconstructed = raw @ recovered
        for column, (axis, sign, momentum, packet_current,
                     packet_target) in enumerate(metadata):
            q_native = native_matrix[:, column]
            q_target = target_columns[column]
            native_field = complex_field(recovered[:, column])
            density_native, current_native = current.observables(
                native_field, alpha)
            density_target, current_target = current.observables(
                packet_target, alpha)
            center_native = dirac.wrapped_shift(
                dirac.circular_center(packet_current),
                dirac.circular_center(native_field))
            center_target = dirac.wrapped_shift(
                dirac.circular_center(packet_current),
                dirac.circular_center(packet_target))
            current_scale = max(float(np.linalg.norm(current_target)), 1e-30)
            rows.append({
                "copy": flavor,
                "axis": "xyz"[axis],
                "axis_index": axis,
                "sign": sign,
                "momentum": momentum.tolist(),
                "native_vs_dirac_state_relative_error": float(
                    np.linalg.norm(q_native-q_target)
                    / max(np.linalg.norm(q_target), 1e-30)),
                "outside_fixed_spatial_readout_fraction": float(
                    np.linalg.norm(q_native-reconstructed[:, column])
                    / max(np.linalg.norm(q_native), 1e-30)),
                "density_relative_error": float(
                    abs(density_native-density_target)
                    / max(density_target, 1e-30)),
                "readout_current_relative_error": float(
                    np.linalg.norm(current_native-current_target)
                    / current_scale),
                "native_readout_current": current_native.tolist(),
                "dirac_current": current_target.tolist(),
                "native_center_shift": center_native.tolist(),
                "dirac_center_shift": center_target.tolist(),
                "center_shift_absolute_error": float(
                    np.linalg.norm(center_native-center_target)),
                "longitudinal_sign_matches": bool(
                    np.sign(current_native[axis])
                    == np.sign(current_target[axis])),
            })

    result = {
        "stage": "V4-005",
        "status": "SHORT_TIME_TRANSPORT_AUDIT",
        "action_sha256": sha256(HERE / "live_clifford_bond_action.py"),
        "tangent_cache_sha256": sha256(tangent_cache),
        "rest_cache_sha256": sha256(rest_cache),
        "dt": args.dt,
        "wave_number": args.wave_number,
        "packet_width": args.packet_width,
        "chi_scale_source": "native_kinetic_normalization_1_over_sqrt_B",
        "frequency_branch": "negative_matching_positive_phase_rest_orbit",
        "readout": (
            "translation-covariant all-center extension of the frozen "
            "V4-003 local cubical Psi/chi lift; one fixed minimum-norm "
            "map per copy, constructed before all cases"),
        "readouts": readouts,
        "rows": rows,
        "maximum_state_relative_error": max(
            row["native_vs_dirac_state_relative_error"] for row in rows),
        "maximum_readout_current_relative_error": max(
            row["readout_current_relative_error"] for row in rows),
        "maximum_density_relative_error": max(
            row["density_relative_error"] for row in rows),
        "maximum_center_shift_absolute_error": max(
            row["center_shift_absolute_error"] for row in rows),
        "all_longitudinal_signs_match": all(
            row["longitudinal_sign_matches"] for row in rows),
        "claim_boundary": (
            "One unrestricted tangent update for both copies and both signs "
            "of all three axes. The independently prepared centered-C19 "
            "Dirac packet is comparator data only. The readout current here "
            "must be paired with the action-derived Noether current before "
            "V4-005 can pass."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
