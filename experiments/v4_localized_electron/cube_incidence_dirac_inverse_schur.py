"""Use Dirac as an inverse diagnostic for the native movement bridge.

For a Dirac plane-wave eigenstate, the fixed overlapping-cube lift generally
is not an invariant native mode.  This audit eliminates the orthogonal native
coordinates by a frequency-resolved Schur complement.  It answers two
separate questions without changing the action:

1. Can an action-derived Psi--chi tail dress the fixed Dirac carrier at the
   *Dirac-prescribed* frequency?
2. After the best such tail is included, does the residual lie in the span of
   already-derived reciprocal action directions?

The fitted action coefficients are diagnostics only.  They are never installed
and have no coefficient provenance.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, qr

import canonical_noncoplanar_bifurcation_audit as native_time
import cube_incidence_extended_packet_gate as gate
import cube_incidence_memory_channel_ablation as ablation
import cube_incidence_nonlinear_action as action
import current_action_localized_envelope_closure as envelope


HERE = Path(__file__).resolve().parent


def complex_native_amplitude(raw: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Complex amplitude whose real time trace is the realified native lift."""
    cosine = raw @ gate.real_coordinates(field)
    sine = raw @ gate.real_coordinates(-1j * field)
    return cosine + 1j * sine


def channel_fractions(vector: np.ndarray) -> dict:
    shaped = vector.reshape(5, 5, 5, 4, 3)
    power = np.sum(abs(shaped) ** 2, axis=(0, 1, 2))
    total = max(float(np.sum(power)), 1e-300)
    return {
        "Psi": float(np.sum(power[:2]) / total),
        "chi": float(np.sum(power[2:]) / total),
        "C3_components": [
            float(value / total) for value in np.sum(power, axis=0)
        ],
    }


def action_directions(base: tuple[np.ndarray, ...]) -> dict:
    bare_blocks = ablation.blocks()[
        "base_without_density_or_cube_connection"]
    density_blocks = ablation.blocks()["base_plus_density_connection"]
    cube_blocks = ablation.blocks()["base_plus_cube_incidence_connection"]
    zero = np.zeros_like(base[0])
    result = {
        "density_connection": tuple(
            density_blocks[i] - bare_blocks[i] for i in range(3)),
        "cube_incidence_connection": tuple(
            cube_blocks[i] - bare_blocks[i] for i in range(3)),
    }
    for filename, key, name in (
        ("joint_phase_lock_action_hessian.npz", "hessian",
         "joint_phase_lock"),
        ("relative_bilinear_phase_action_hessian.npz", "normalized_hessian",
         "relative_bilinear_phase"),
        ("area_bilinear_phase_action_hessian.npz", "normalized_hessian",
         "area_bilinear_phase"),
    ):
        matrix = np.load(HERE / filename)[key]
        result[name] = (zero, zero, matrix)
    pair = np.load(HERE / "density_oriented_pair_connection_matrices.npz")
    for kind in ("common", "relative"):
        result[f"density_oriented_{kind}_pair"] = tuple(
            pair[f"{block}_{kind}_pair_connection"]
            for block in ("mass", "gyro", "stiffness"))
    return result


def direction_cases() -> list[tuple[str, np.ndarray]]:
    magnitude = 0.20
    rows = [("rest", np.zeros(3))]
    for axis in range(3):
        for sign in (-1, 1):
            value = np.zeros(3)
            value[axis] = sign * magnitude
            rows.append((f"{'xyz'[axis]}{'+' if sign > 0 else '-'}", value))
    diagonal = np.full(3, magnitude / np.sqrt(3))
    rows.append(("diagonal+", diagonal))
    rows.append(("diagonal-", -diagonal))
    return rows


def main() -> None:
    bare, matrices = action.expected_blocks()
    mass = matrices["mass"]
    gyro = matrices["gyro"]
    stiffness = matrices["stiffness"]
    base = (mass, gyro, stiffness)
    candidates = action_directions(base)

    psi_frame, chi_frame, sites = envelope.carrier_frames(bare["rest"])
    prior = json.loads(
        (HERE / "cube_incidence_dirac_envelope_gate.json").read_text())
    scales = {}
    for row in prior["rows"]:
        case = next(item for item in row["cases"]
                    if item["envelope"] == "all_125_centers")
        scales[row["flavor"]] = complex(
            case["chi_scale_real"], case["chi_scale_imag"])

    rows = []
    diagnostic_columns = {name: [] for name in candidates}
    diagnostic_targets = []
    for flavor, basis in gate.flavor_bases().items():
        alpha, beta = gate.compressed_clifford(basis)
        raw = envelope.raw_columns(
            basis, psi_frame, chi_frame, sites, gate.OFFSETS, scales[flavor])
        retained = envelope.orth(raw)
        complete, _ = qr(retained, mode="full", check_finite=False)
        normal = complete[:, retained.shape[1]:]

        # Project the native quadratic pencil once.  Each case changes only
        # the scalar frequency.
        projected = {}
        for name, matrix in (("mass", mass), ("gyro", gyro),
                             ("stiffness", stiffness)):
            projected[f"nn_{name}"] = normal.T @ matrix @ normal

        coordinates = np.asarray(gate.OFFSETS, dtype=float).reshape(
            gate.SIDE, gate.SIDE, gate.SIDE, 3)
        for label, momentum in direction_cases():
            transport = gate.c19_first_symbol(momentum)
            hamiltonian = native_time.CHI0 * (beta - np.eye(4))
            for axis in range(3):
                hamiltonian += transport[axis] * alpha[axis]
            values, vectors = np.linalg.eigh(hamiltonian)
            band = int(np.argmax(values))
            omega = float(values[band])
            spinor = vectors[:, band]
            phase = np.exp(1j * np.einsum(
                "...a,a->...", coordinates, momentum))
            field = phase[..., None] * spinor
            undressed = complex_native_amplitude(raw, field)

            pencil = stiffness - omega**2 * mass - 1j * omega * gyro
            normal_pencil = (
                projected["nn_stiffness"]
                - omega**2 * projected["nn_mass"]
                - 1j * omega * projected["nn_gyro"])
            right = -(normal.T @ (pencil @ undressed))
            tail_coordinates, _residuals, rank, singular = lstsq(
                normal_pencil, right, cond=1e-10,
                lapack_driver="gelsy", check_finite=False)
            dressed = undressed + normal @ tail_coordinates
            undressed_residual = pencil @ undressed
            dressed_residual = pencil @ dressed
            retained_residual = retained.T @ dressed_residual
            normal_residual = normal.T @ dressed_residual
            row = {
                "flavor": flavor,
                "direction": label,
                "momentum": momentum.tolist(),
                "C19_transport_symbol": transport.tolist(),
                "Dirac_relative_frequency": omega,
                "normal_solve_rank": int(rank),
                "normal_dimension": int(normal.shape[1]),
                "normal_solve_min_reported_singular": (
                    float(singular[-1]) if singular is not None
                    and len(singular) else None),
                "tail_over_undressed_norm": float(
                    np.linalg.norm(normal @ tail_coordinates)
                    / max(np.linalg.norm(undressed), 1e-300)),
                "undressed_pencil_residual_over_state": float(
                    np.linalg.norm(undressed_residual)
                    / max(np.linalg.norm(undressed), 1e-300)),
                "dressed_pencil_residual_over_state": float(
                    np.linalg.norm(dressed_residual)
                    / max(np.linalg.norm(dressed), 1e-300)),
                "residual_reduction_fraction": float(
                    1 - np.linalg.norm(dressed_residual)
                    / max(np.linalg.norm(undressed_residual), 1e-300)),
                "retained_residual_fraction": float(
                    np.linalg.norm(retained_residual)
                    / max(np.linalg.norm(dressed_residual), 1e-300)),
                "normal_residual_fraction": float(
                    np.linalg.norm(normal_residual)
                    / max(np.linalg.norm(dressed_residual), 1e-300)),
                "residual_channels": channel_fractions(dressed_residual),
            }
            rows.append(row)

            # Common-coefficient inverse action screen.  The rest row is
            # retained because an admissible correction must not spoil it.
            target = -dressed_residual
            diagnostic_targets.append(np.concatenate((target.real,
                                                       target.imag)))
            for name, (dmass, dgyro, dstiffness) in candidates.items():
                response = (dstiffness - omega**2 * dmass
                            - 1j * omega * dgyro) @ dressed
                diagnostic_columns[name].append(np.concatenate(
                    (response.real, response.imag)))

    target = np.concatenate(diagnostic_targets)
    names = list(candidates)
    response = np.column_stack([
        np.concatenate(diagnostic_columns[name]) for name in names])
    coefficients, *_ = np.linalg.lstsq(response, target, rcond=1e-10)
    unrepaired = np.linalg.norm(target)
    repaired = np.linalg.norm(target - response @ coefficients)
    singular = np.linalg.svd(response, compute_uv=False)
    result = {
        "status": "V4_DIRAC_INVERSE_SCHUR_MOVEMENT_DIAGNOSTIC",
        "canonical_action_changed": False,
        "research_action_changed": False,
        "method": (
            "frequency-resolved Schur elimination of every native coordinate "
            "orthogonal to the fixed four-component spatial lift"),
        "rows": rows,
        "maximum_tail_over_undressed_norm": max(
            row["tail_over_undressed_norm"] for row in rows),
        "maximum_dressed_pencil_residual_over_state": max(
            row["dressed_pencil_residual_over_state"] for row in rows),
        "minimum_residual_reduction_fraction": min(
            row["residual_reduction_fraction"] for row in rows),
        "action_inverse_screen": {
            "candidate_count": len(names),
            "response_rank": int(np.sum(
                singular > max(singular[0], 1.0) * 1e-10)),
            "unrepaired_norm": float(unrepaired),
            "best_common_coefficient_relative_residual": float(
                repaired / max(unrepaired, 1e-300)),
            "diagnostic_coefficients": {
                name: float(value) for name, value in zip(
                    names, coefficients, strict=True)
            },
            "coefficients_admitted": False,
        },
        "verdict": "DIRAC_FREQUENCY_DRESSED_RESIDUAL_MEASURED",
        "claim_boundary": (
            "Linear tangent inverse diagnostic. Dirac fixes the comparison "
            "frequency but supplies no native force. Schur tails are solved "
            "state readouts, not runtime support. Fitted action coefficients "
            "are diagnostic only."),
    }
    (HERE / "cube_incidence_dirac_inverse_schur.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "maximum_tail_over_undressed_norm": result[
            "maximum_tail_over_undressed_norm"],
        "maximum_dressed_pencil_residual_over_state": result[
            "maximum_dressed_pencil_residual_over_state"],
        "minimum_residual_reduction_fraction": result[
            "minimum_residual_reduction_fraction"],
        "action_inverse_screen": result["action_inverse_screen"],
        "verdict": result["verdict"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
