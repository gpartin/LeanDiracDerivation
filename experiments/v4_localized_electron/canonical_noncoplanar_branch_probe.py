"""Continue the strongest canonical noncoplanar tangent nonlinearly.

The complete canonical orbit Hessian has a carrier-W quotient after global
symmetries are removed.  This Lyapunov--Schmidt calculation asks whether its
strongest direction survives the nonlinear V40 equations.  It also measures
the locked eight-site identities on every range-corrected state.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_bifurcation_audit as audit
import localized_stability as native


HERE = Path(__file__).resolve().parent
FRACTIONS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
             1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def align_phase(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    overlap = np.vdot(reference, value)
    if abs(overlap) == 0:
        return value
    return value * np.exp(-1j * np.angle(overlap))


def identity_metrics(reference: np.ndarray, state: np.ndarray) -> dict:
    psi0, chi0 = audit.unpack(reference)
    psi, chi = audit.unpack(state)
    occupied = np.argwhere(np.sum(abs(psi0)**2, axis=-1) > 1e-12)
    p0 = np.asarray([psi0[tuple(site)] for site in occupied])
    c0 = np.asarray([chi0[tuple(site)] for site in occupied])
    p = np.asarray([psi[tuple(site)] for site in occupied])
    c = np.asarray([chi[tuple(site)] for site in occupied])
    p = align_phase(p0, p)
    c = align_phase(c0, c)
    psi_error = np.linalg.norm(p - p0) / np.linalg.norm(p0)
    chi_error = np.linalg.norm(c - c0) / np.linalg.norm(c0)

    # Opposite-corner relations are measured against the actual locked
    # reference, avoiding assumptions about enumeration or an omitted twist.
    pair_errors = []
    for index in range(4):
        reference_map = np.vdot(p0[index], p0[7-index]) \
            / max(np.vdot(p0[index], p0[index]).real, 1e-300)
        pair_errors.append(np.linalg.norm(
            p[7-index] - reference_map * p[index])
            / max(np.linalg.norm(p[7-index]), 1e-300))

    w0 = np.cross(c0, p0)
    w = np.cross(c, p)
    w_matrix = np.concatenate((w.real, w.imag), axis=1)
    singular = np.linalg.svd(w_matrix, compute_uv=False)
    exterior = np.ones(psi.shape[:3], dtype=bool)
    exterior[tuple(occupied.T)] = False
    exterior_fraction = float(np.sum(abs(psi[exterior])**2)
                              / max(np.sum(abs(psi)**2), 1e-300))
    return {
        "psi_locked_fidelity": float(max(0, 1 - psi_error**2)),
        "chi_locked_fidelity": float(max(0, 1 - chi_error**2)),
        "maximum_opposite_pair_error": float(max(pair_errors)),
        "carrier_W_singular_values": [float(x) for x in singular],
        "carrier_W_third_over_second": float(
            singular[2] / max(singular[1], 1e-300)),
        "exterior_psi_power_fraction": exterior_fraction,
    }


def main() -> None:
    psi, chi = native.carrier_fields(audit.SIDE, audit.AMPLITUDE_SQ)
    current = audit.pack(psi, chi)
    theta = float(np.arccos(
        1 - audit.DT**2 * audit.CHI0**2 / 2))
    phase = np.exp(1j * theta)

    def orbit_action(raw: torch.Tensor) -> torch.Tensor:
        return (audit.interval_action(
            audit.phase_rotate(raw, np.conjugate(phase)), raw)
            + audit.interval_action(raw, audit.phase_rotate(raw, phase)))

    def residual(raw: np.ndarray) -> np.ndarray:
        variable = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
        return torch.autograd.grad(
            orbit_action(variable), variable)[0].detach().numpy()

    base = torch.tensor(current, dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        orbit_action, base, vectorize=True).detach().numpy()
    hessian = (hessian + hessian.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    scale = max(float(np.max(abs(eigenvalues))), 1e-300)
    null_mask = abs(eigenvalues) < 1e-8 * scale
    null_basis = eigenvectors[:, null_mask]
    range_basis = eigenvectors[:, ~null_mask]
    range_values = eigenvalues[~null_mask]

    occupied = np.sum(abs(psi)**2, axis=-1) > 1e-12
    pair_constraint = np.column_stack([
        audit.opposite_pair_constraint(
            current, null_basis[:, index], occupied)
        for index in range(null_basis.shape[1])
    ])
    pair_kernel = audit.kernel(pair_constraint)
    candidate_basis = null_basis @ pair_kernel
    w_null = np.column_stack([
        audit.carrier_transverse_w_tangent(
            current, candidate_basis[:, index], occupied)
        for index in range(candidate_basis.shape[1])
    ])
    symmetries = audit.global_tangents(current)
    w_symmetry = np.column_stack([
        audit.carrier_transverse_w_tangent(
            current, symmetries[:, index], occupied)
        for index in range(symmetries.shape[1])
    ])
    w_symmetry_basis = audit.orth(w_symmetry)
    quotient = w_null - w_symmetry_basis @ (w_symmetry_basis.T @ w_null)
    _u, singular, vh = np.linalg.svd(quotient, full_matrices=False)
    direction = candidate_basis @ vh[0]
    direction /= np.linalg.norm(direction)
    base_norm = np.linalg.norm(current)
    rows = []
    states = []

    for fraction in FRACTIONS:
        requested = fraction * base_norm
        state = current + requested * direction
        trace = []
        for _iteration in range(12):
            value = residual(state)
            range_coordinates = range_basis.T @ value
            null_coordinates = null_basis.T @ value
            trace.append({
                "total_rms": float(np.linalg.norm(value)
                                   / np.sqrt(value.size)),
                "range_norm": float(np.linalg.norm(range_coordinates)),
                "null_norm": float(np.linalg.norm(null_coordinates)),
            })
            old_range = np.linalg.norm(range_coordinates)
            if old_range < 1e-12:
                break
            correction = -range_basis @ (range_coordinates / range_values)
            accepted = False
            for power in range(12):
                trial = state + 0.5**power * correction
                if np.linalg.norm(range_basis.T @ residual(trial)) < old_range:
                    state = trial
                    accepted = True
                    break
            if not accepted:
                break
        final = residual(state)
        range_final = range_basis.T @ final
        null_final = null_basis.T @ final
        displacement = state - current
        amplitude = float(direction @ displacement)
        row = {
            "fraction_of_rest_norm": fraction,
            "requested_amplitude": requested,
            "retained_direction_amplitude": amplitude,
            "range_correction_norm": float(np.linalg.norm(
                displacement - amplitude * direction)),
            "final_total_rms": float(np.linalg.norm(final)
                                     / np.sqrt(final.size)),
            "final_range_norm": float(np.linalg.norm(range_final)),
            "final_null_solvability_norm": float(np.linalg.norm(null_final)),
            "null_norm_over_amplitude_squared": float(
                np.linalg.norm(null_final) / max(amplitude**2, 1e-300)),
            **identity_metrics(current, state),
            "trace": trace,
        }
        rows.append(row)
        states.append(state)

    solved = [row for row in rows
              if row["final_total_rms"] < 1e-9
              and row["fraction_of_rest_norm"] >= 1e-4]
    identity_preserved = [row for row in solved
                          if row["psi_locked_fidelity"] > 0.999999
                          and row["maximum_opposite_pair_error"] < 1e-6]
    ratios = [row["null_norm_over_amplitude_squared"] for row in rows[-3:]]
    obstructed = (np.median(ratios) > 1e-6
                  and all(row["final_null_solvability_norm"] > 1e-10
                          for row in rows[-3:]))
    if identity_preserved:
        verdict = "CANONICAL_NONCOPLANAR_IDENTITY_PRESERVING_BRANCH_CANDIDATE"
    elif solved:
        verdict = "CANONICAL_NONCOPLANAR_BRANCH_BREAKS_LOCKED_IDENTITY"
    elif obstructed:
        verdict = "CANONICAL_NONCOPLANAR_DIRECTION_NONLINEARLY_OBSTRUCTED"
    else:
        verdict = "CANONICAL_NONCOPLANAR_BRANCH_INCONCLUSIVE"

    result = {
        "status": "V4_CANONICAL_NONCOPLANAR_BRANCH_PROBE",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "action": "locked canonical V40 U_local plus U_area",
        "full_nullity": int(null_basis.shape[1]),
        "opposite_pair_preserving_nullity": int(
            candidate_basis.shape[1]),
        "strongest_carrier_W_quotient_singular_value": float(singular[0]),
        "base_rest_residual_rms": float(
            np.linalg.norm(residual(current)) / np.sqrt(current.size)),
        "rows": rows,
        "verdict": verdict,
        "claim_boundary": (
            "Range-corrected continuation of one strongest canonical "
            "carrier-W quotient direction. A numeric branch still requires "
            "a full augmented solve, spectrum, current, and box refinement."),
    }
    output = HERE / "canonical_noncoplanar_branch_probe.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    np.savez_compressed(
        HERE / "canonical_noncoplanar_branch_probe.npz",
        rest=current, direction=direction, fractions=np.asarray(FRACTIONS),
        states=np.asarray(states))
    print(json.dumps({
        "verdict": verdict,
        "base_rest_residual_rms": result["base_rest_residual_rms"],
        "strongest_carrier_W_quotient_singular_value": result[
            "strongest_carrier_W_quotient_singular_value"],
        "last_rows": [{key: row[key] for key in (
            "fraction_of_rest_norm", "final_total_rms",
            "final_null_solvability_norm", "psi_locked_fidelity",
            "maximum_opposite_pair_error", "carrier_W_third_over_second")}
            for row in rows[-4:]],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
