"""V4-003 fixed localized four-component phase-space readout.

The native theory is second order, so a configuration-only corner projection
cannot be the Dirac readout.  This gate fixes one map from the *two committed
native layers* to four complex cubical amplitudes.  The map contains the live
  Psi and chi tangent prepared at every envelope center.  Its previous-layer
  graph and first-order generator are solved from the exact native action
  tangent, never supplied by Dirac and never refitted to a trajectory.

For each cubical copy the test:

* places an arbitrary four-component amplitude in one localized Gaussian
  envelope (all four complex basis directions are tested simultaneously);
* synthesizes both committed native layers, including the fixed live-chi
  lift and the complete side-five surroundings;
* derives the bounded previous-layer invariant graph from the native update;
* constructs the unique least-squares left inverse of that fixed embedding;
* advances the unrestricted exact-action tangent by one native update; and
* separates readout evolution error from the complementary native residual.

V4-003 passes only if recovery is lossless and the explicitly retained
complementary residual shrinks under timestep refinement.  Comparison of the
derived first-order generator with Dirac is deliberately deferred to V4-005.
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


def sparse(value: np.ndarray, tolerance: float) -> csc_matrix:
    result = value.copy()
    result[np.abs(result) < tolerance] = 0.0
    return csc_matrix(result)


def localized_four_component_columns(side: int, width: float) -> np.ndarray:
    offsets = np.asarray(
        list(itertools.product(range(-(side // 2), side // 2 + 1),
                               repeat=3)),
        dtype=float,
    ).reshape(side, side, side, 3)
    envelope = np.exp(-np.sum(offsets**2, axis=-1) / (2 * width**2))
    fields = []
    for component in range(4):
        field = np.zeros((side, side, side, 4), dtype=complex)
        field[..., component] = envelope
        fields.extend((field, 1j * field))
    coordinates = []
    for field in fields:
        row = []
        for value in field.reshape(-1):
            row.extend((value.real, value.imag))
        coordinates.append(row)
    return np.asarray(coordinates).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tangent-cache", type=Path, required=True)
    parser.add_argument("--rest-cache", type=Path, required=True)
    parser.add_argument("--action-module", type=Path,
                        default=Path("live_clifford_bond_action.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--packet-width", type=float, default=1.35)
    parser.add_argument("--sparse-tolerance", type=float, default=1e-11)
    args = parser.parse_args()

    tangent_cache = args.tangent_cache.resolve()
    rest_cache = args.rest_cache.resolve()
    action_module = (args.action_module
                     if args.action_module.is_absolute()
                     else HERE / args.action_module).resolve()
    for path in (tangent_cache, rest_cache, action_module):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(HERE))
    import personal_touch_unit_geometric_canonical as canonical
    canonical.CACHE = rest_cache
    import canonical_noncoplanar_bifurcation_audit as native_time
    import canonical_transfer_oscillator_spectrum as native_spectrum
    import current_action_localized_envelope_closure as envelope

    # This is the native mass-normalized coordinate relation.  It is fixed by
    # B and is not optimized against closure, Dirac, or any trajectory.
    scales = {
        flavor: complex(1 / np.sqrt(native_time.B), 0.0)
        for flavor in ("plus", "minus")
    }
    scale_source = "native_kinetic_normalization_1_over_sqrt_B"

    blocks = np.load(tangent_cache)
    previous, current, following = (
        blocks[name] for name in ("previous", "current", "following"))
    rest = np.load(rest_cache)["rest"]
    dimension = rest.size
    theta = float(np.arccos(
        1 - args.dt**2 * native_time.CHI0**2 / 2))
    rotate_minus = native_spectrum.rotation_matrix(dimension, -theta)
    rotate_plus = native_spectrum.rotation_matrix(dimension, theta)
    lhs = sparse(following @ rotate_plus, args.sparse_tolerance)
    current_operator = sparse(current, args.sparse_tolerance)
    previous_operator = sparse(previous @ rotate_minus,
                               args.sparse_tolerance)
    factor = splu(lhs)

    psi_frame, chi_frame, sites = envelope.carrier_frames(rest)
    spinor_columns = localized_four_component_columns(
        5, args.packet_width)
    rows = []
    offsets = list(itertools.product((-2, -1, 0, 1, 2), repeat=3))
    for flavor, basis in envelope.flavor_bases().items():
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, offsets,
            scales[flavor])

        current_spinor = spinor_columns
        q_seed = raw @ current_spinor
        q_current, coordinate_change = np.linalg.qr(q_seed, mode="reduced")
        coordinate_condition = float(np.linalg.cond(coordinate_change))
        forward_q = factor.solve(-(current_operator @ q_current))
        backward_q = factor.solve(-(previous_operator @ q_current))
        forward_normal = forward_q - q_current @ (
            q_current.T @ forward_q)
        backward_normal = backward_q - q_current @ (
            q_current.T @ backward_q)
        previous_graph, *_ = np.linalg.lstsq(
            backward_normal, -forward_normal, rcond=1e-12)
        graph_condition = float(np.linalg.cond(previous_graph))
        if not np.isfinite(graph_condition) or graph_condition > 1e12:
            raise RuntimeError(
                f"ill-conditioned native graph for {flavor}: "
                f"{graph_condition}")
        generator = np.linalg.inv(previous_graph)
        q_previous = q_current @ previous_graph
        q_native_next = forward_q + backward_q @ previous_graph
        q_expected_next = q_current @ generator
        inverse_defect = float(np.linalg.norm(
            previous_graph @ generator
            - np.eye(previous_graph.shape[0]), 2))
        phase_source_detail = (
            "bounded least-squares invariant graph of exact native "
            "two-layer tangent")

        phase_embedding = np.vstack((q_current, q_previous))
        readout = np.linalg.pinv(phase_embedding, rcond=1e-12)
        recovery = readout @ phase_embedding
        recovery_error = float(np.linalg.norm(
            recovery - np.eye(recovery.shape[0]), 2))
        condition = float(np.linalg.cond(phase_embedding))

        native_next_phase = np.vstack((q_native_next, q_current))
        expected_next_phase = np.vstack((q_expected_next, q_current))
        recovered_next = readout @ native_next_phase
        target_next = readout @ expected_next_phase
        readout_error = float(
            np.linalg.norm(recovered_next - target_next)
            / max(np.linalg.norm(target_next), 1e-30))
        reconstructed = phase_embedding @ recovered_next
        complement = native_next_phase - reconstructed
        complementary_residual = float(
            np.linalg.norm(complement)
            / max(np.linalg.norm(native_next_phase), 1e-30))
        direct_phase_error = float(
            np.linalg.norm(native_next_phase - expected_next_phase)
            / max(np.linalg.norm(expected_next_phase), 1e-30))
        shaped = complement[:dimension].reshape(
            5, 5, 5, 12, -1)
        psi_power = float(np.linalg.norm(shaped[..., :6, :])**2)
        chi_power = float(np.linalg.norm(shaped[..., 6:, :])**2)
        total_power = max(psi_power + chi_power, 1e-300)
        carrier_power = 0.0
        for site in sites:
            carrier_power += float(np.linalg.norm(shaped[site])**2)
        rows.append({
            "copy": flavor,
            "four_complex_components": True,
            "real_component_count": int(spinor_columns.shape[1]),
            "native_phase_rows": int(phase_embedding.shape[0]),
            "live_chi_scale": {
                "real": float(scales[flavor].real),
                "imag": float(scales[flavor].imag),
            },
            "readout_condition_number": condition,
            "cubical_coordinate_condition_number": coordinate_condition,
            "previous_layer_graph_condition_number": graph_condition,
            "phase_source": phase_source_detail,
            "phase_embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(phase_embedding).tobytes()).hexdigest(),
            "fixed_readout_sha256": hashlib.sha256(
                np.ascontiguousarray(readout).tobytes()).hexdigest(),
            "exact_recovery_error": recovery_error,
            "centered_time_inverse_defect": inverse_defect,
            "readout_evolution_relative_error": readout_error,
            "direct_phase_relative_error": direct_phase_error,
            "complementary_sector_relative_residual": (
                complementary_residual),
            "complementary_power_fractions": {
                "Psi": psi_power / total_power,
                "chi": chi_power / total_power,
                "eight_carrier_sites": carrier_power / total_power,
                "surroundings": max(1.0 - carrier_power / total_power, 0.0),
            },
        })

    result = {
        "stage": "V4-003",
        "status": "FIXED_PHASE_SPACE_READOUT_AUDIT",
        "action_module": action_module.name,
        "action_sha256": sha256(action_module),
        "tangent_cache": tangent_cache.name,
        "tangent_cache_sha256": sha256(tangent_cache),
        "rest_cache": rest_cache.name,
        "rest_cache_sha256": sha256(rest_cache),
        "chi_scale_source": scale_source,
        "phase_source": "native_graph",
        "dt": args.dt,
        "packet_width": args.packet_width,
        "readout_construction": (
            "one Moore-Penrose left inverse of the fixed two-layer cubical "
            "Psi/chi envelope embedding; the previous-layer graph is derived "
            "from the exact native action tangent; no trajectory fit or "
            "evolving frame"),
        "rows": rows,
        "maximum_exact_recovery_error": max(
            row["exact_recovery_error"] for row in rows),
        "maximum_complementary_sector_relative_residual": max(
            row["complementary_sector_relative_residual"] for row in rows),
        "maximum_readout_evolution_relative_error": max(
            row["readout_evolution_relative_error"] for row in rows),
        "claim_boundary": (
            "Fixed localized linear phase-space readout at one packet width. "
            "V4-003 requires a separate dt-refinement aggregate before PASS. "
            "This gate does not prove spin, current, or a complete cell crossing."),
    }
    output = args.output if args.output.is_absolute() else HERE / args.output
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
