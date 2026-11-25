"""Action-derived Noether current pulled through the V4-003 carrier map.

The full native phase current contains microscopic circulation inside each
eight-corner carrier.  Dirac is a continuum/readout equation, so V4-005 must
vary the phase of the *center-resolved four-component field* and pull that
variation back through the frozen native Psi/chi lift.  This script performs
that chain rule directly on the exact discrete action.  No Dirac current is
inserted into the native force or current definition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


HERE = Path(__file__).resolve().parent
SIDE = 5


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


def phase_coordinate(field: np.ndarray,
                     mask: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(field)
    turned = 1j*value
    if mask is not None:
        turned = turned*mask[..., None]
    result = []
    for entry in turned.reshape(-1, 4):
        for component in entry:
            result.extend((component.real, component.imag))
    return np.asarray(result)


def interval_gradient(action, left: np.ndarray, right: np.ndarray,
                      dt: float) -> tuple[np.ndarray, np.ndarray]:
    left_tensor = torch.tensor(left, dtype=torch.float64,
                               requires_grad=True)
    right_tensor = torch.tensor(right, dtype=torch.float64,
                                requires_grad=True)
    value = action.interval(left_tensor, right_tensor, dt)
    first, second = torch.autograd.grad(value, (left_tensor, right_tensor))
    return first.detach().numpy(), second.detach().numpy()


def linearized_gradients(action, rest_left: np.ndarray,
                         rest_right: np.ndarray, tangent_left: np.ndarray,
                         tangent_right: np.ndarray, amplitude: float,
                         dt: float) -> tuple[np.ndarray, np.ndarray]:
    plus = interval_gradient(
        action, rest_left+amplitude*tangent_left,
        rest_right+amplitude*tangent_right, dt)
    minus = interval_gradient(
        action, rest_left-amplitude*tangent_left,
        rest_right-amplitude*tangent_right, dt)
    return tuple((positive-negative)/(2*amplitude)
                 for positive, negative in zip(plus, minus))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-amplitude", type=float, default=1e-4)
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
    import cube_incidence_extended_packet_gate as dirac
    import current_action_localized_envelope_closure as envelope
    import live_clifford_bond_action as action

    blocks = np.load(tangent_cache)
    previous, middle, following = (
        blocks[name] for name in ("previous", "current", "following"))
    rest = np.load(rest_cache)["rest"]
    dimension = rest.size
    theta = float(np.arccos(
        1-args.dt**2*native_time.CHI0**2/2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    factor = splu(sparse(following @ rotate_plus))
    middle_operator = sparse(middle)
    previous_operator = sparse(previous @ rotate_minus)
    rest_previous = rotate_minus @ rest
    rest_current = rest
    rest_following = rotate_plus @ rest

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    chi_scale = complex(1/np.sqrt(native_time.B), 0.0)
    masks = []
    coordinates = np.arange(SIDE)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = SIDE
        masks.append((coordinates.reshape(shape) <= SIDE//2).astype(float))

    rows = []
    for flavor, basis in envelope.flavor_bases().items():
        alpha, beta = dirac.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets, chi_scale)
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
            scale = (args.relative_amplitude*np.linalg.norm(rest)
                     / max(np.linalg.norm(q_current), 1e-30))
            lab_previous = rotate_minus @ q_previous
            lab_current = q_current
            lab_native = rotate_plus @ q_native
            lab_target = rotate_plus @ q_target

            _, before_right = linearized_gradients(
                action, rest_previous, rest_current,
                lab_previous, lab_current, scale, args.dt)
            native_left, native_right = linearized_gradients(
                action, rest_current, rest_following,
                lab_current, lab_native, scale, args.dt)
            target_left, target_right = linearized_gradients(
                action, rest_current, rest_following,
                lab_current, lab_target, scale, args.dt)
            global_current = raw @ phase_coordinate(packet_current)
            global_target = raw @ phase_coordinate(packet_target)
            mask = masks[axis]
            local_current = raw @ phase_coordinate(packet_current, mask)
            local_target = raw @ phase_coordinate(packet_target, mask)
            # q_native is read through the fixed all-center rule only for the
            # phase generator; solve once without using the Dirac target.
            gram = raw.T@raw
            ridge = 1e-13*max(float(np.linalg.norm(gram, 2)), 1.0)
            native_coordinates = np.linalg.solve(
                gram+ridge*np.eye(gram.shape[0]), raw.T@q_native)
            native_field = (
                native_coordinates.reshape(-1, 2)[:, 0]
                + 1j*native_coordinates.reshape(-1, 2)[:, 1]
            ).reshape(SIDE, SIDE, SIDE, 4)
            global_native = raw @ phase_coordinate(native_field)
            local_native = raw @ phase_coordinate(native_field, mask)

            charge_before = float(before_right @ global_current)
            charge_native = float(native_right @ global_native)
            charge_target = float(target_right @ global_target)
            flux_native = float(
                native_left@local_current + native_right@local_native)
            flux_target = float(
                target_left@local_current + target_right@local_target)
            norm = max(abs(charge_before), 1e-30)
            probability_current = np.sum(abs(packet_current)**2, axis=-1)
            probability_target = np.sum(abs(packet_target)**2, axis=-1)
            probability_flux = float(np.sum(
                mask*(probability_target-probability_current))
                / max(np.sum(probability_current), 1e-30))
            rows.append({
                "copy": flavor,
                "axis": "xyz"[axis],
                "axis_index": axis,
                "sign": sign,
                "momentum": momentum.tolist(),
                "charge_before": charge_before,
                "native_charge_relative_change": float(
                    (charge_native-charge_before)/norm),
                "embedded_dirac_charge_relative_change": float(
                    (charge_target-charge_before)/norm),
                "native_normalized_cut_flux": flux_native/norm,
                "embedded_dirac_normalized_cut_flux": flux_target/norm,
                "dirac_probability_normalized_cut_flux": probability_flux,
                "native_vs_embedded_dirac_cut_flux_absolute_error": float(
                    abs(flux_native-flux_target)/norm),
                "native_target_cut_flux_sign_matches": bool(
                    np.sign(flux_native) == np.sign(flux_target)),
            })

    native_flux = np.asarray(
        [row["native_normalized_cut_flux"] for row in rows])
    target_flux = np.asarray(
        [row["embedded_dirac_normalized_cut_flux"] for row in rows])
    result = {
        "stage": "V4-005",
        "status": "PROJECTED_NATIVE_NOETHER_CURRENT_AUDIT",
        "action_sha256": sha256(HERE / "live_clifford_bond_action.py"),
        "tangent_cache_sha256": sha256(tangent_cache),
        "rest_cache_sha256": sha256(rest_cache),
        "dt": args.dt,
        "relative_amplitude": args.relative_amplitude,
        "current_definition": (
            "local four-component phase variation pulled through the frozen "
            "translation-covariant native Psi/chi lift and contracted with "
            "the exact discrete-action momentum"),
        "rows": rows,
        "maximum_native_charge_relative_change": max(
            abs(row["native_charge_relative_change"]) for row in rows),
        "maximum_cut_flux_absolute_error": max(
            row["native_vs_embedded_dirac_cut_flux_absolute_error"]
            for row in rows),
        "aggregate_cut_flux_relative_error": float(
            np.linalg.norm(native_flux-target_flux)
            / max(np.linalg.norm(target_flux), 1e-30)),
        "all_cut_flux_signs_match": all(
            row["native_target_cut_flux_sign_matches"] for row in rows),
        "claim_boundary": (
            "Center-resolved Noether pullback for one tangent update. "
            "Microscopic within-carrier circulation is excluded by the "
            "derived V4-003 coarse-graining. This is not a full cell crossing."),
    }
    output = args.output if args.output.is_absolute() else HERE/args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "rows"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
