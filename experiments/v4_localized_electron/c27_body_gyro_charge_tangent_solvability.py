"""Full-field Fredholm test for nearby charge/frequency rest families.

The rigid two-amplitude ansatz leaves residuals when either native clock is
changed.  This test allows every Psi and chi coordinate, including halo and
exterior sites, to deform.  For

  F(q, thetaPsi, thetaChi) = DEL(R^- q, q, R^+ q),

it solves J_q dq = -F_theta at the exact rest orbit.  A small range residual
is the necessary and sufficient first-order condition for a nearby relative-
equilibrium family; the solution is a derived collective tangent, not a
runtime projection or background.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import c27_body_gyro_basis_probe as candidate
import c27_body_gyro_rest_spectrum as spectrum
import relative_kinetic_probe as relative


HERE = Path(__file__).resolve().parent
STEP = 1e-6


def rotation(theta_psi: float, theta_chi: float, exponent: int):
    return relative.raw_rotation_matrix(
        spectrum.SIDE, theta_psi, theta_chi, exponent)


def main() -> None:
    spectrum.configure()
    candidate.base.inverse.rank.lock_inertia()
    _, current, _, theta0 = candidate.base.inverse.rest_layers()
    q0 = torch.tensor(current, dtype=torch.float64)
    a = 2 * spectrum.DT / np.sin(theta0)
    coefficients = np.asarray((
        a + candidate.base.inverse.rank.B / 4,
        a - candidate.base.inverse.rank.B / 4,
        -a / 2,
        0.0,
    ))
    del_residual, _ = candidate.make_derivatives(1.0, coefficients)

    def force(q: torch.Tensor, theta_psi: float, theta_chi: float):
        minus = torch.tensor(
            rotation(theta_psi, theta_chi, -1), dtype=torch.float64)
        plus = torch.tensor(
            rotation(theta_psi, theta_chi, 1), dtype=torch.float64)
        return del_residual(minus @ q, q, plus @ q)

    fixed_minus = torch.tensor(rotation(theta0, 0.0, -1), dtype=torch.float64)
    fixed_plus = torch.tensor(rotation(theta0, 0.0, 1), dtype=torch.float64)

    def fixed_force(q):
        return del_residual(fixed_minus @ q, q, fixed_plus @ q)

    jacobian = torch.autograd.functional.jacobian(
        fixed_force, q0.detach().requires_grad_(True), vectorize=True
    ).detach().numpy()
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = singular[0] * 1e-10
    rank = int(np.sum(singular > threshold))
    rows = []
    tangents = {}
    sources = {}
    remainders = {}
    for name, dtheta_psi, dtheta_chi in (
            ("psi_clock", 1.0, 0.0),
            ("chi_internal_clock", 0.0, 1.0)):
        plus = force(q0, theta0 + STEP * dtheta_psi,
                     STEP * dtheta_chi).detach().numpy()
        minus = force(q0, theta0 - STEP * dtheta_psi,
                      -STEP * dtheta_chi).detach().numpy()
        source = (plus - minus) / (2 * STEP)
        tangent = np.linalg.lstsq(
            jacobian, -source, rcond=1e-10)[0]
        remainder = jacobian @ tangent + source
        psi, chi = relative.unpack_raw(tangent, spectrum.SIDE)
        psi_weight = float(np.sum(abs(psi)**2))
        chi_weight = float(np.sum(abs(chi)**2))
        total = max(psi_weight + chi_weight, 1e-300)
        rest_psi, _ = relative.unpack_raw(current, spectrum.SIDE)
        occupied = np.sum(abs(rest_psi)**2, axis=-1) > 1e-12
        carrier_weight = float(
            np.sum(abs(psi[occupied])**2) + np.sum(abs(chi[occupied])**2))
        tangents[name] = tangent
        sources[name] = source
        remainders[name] = remainder
        rows.append({
            "name": name,
            "source_norm": float(np.linalg.norm(source)),
            "tangent_norm": float(np.linalg.norm(tangent)),
            "relative_range_remainder": float(
                np.linalg.norm(remainder) / max(np.linalg.norm(source), 1e-300)),
            "psi_fraction": psi_weight / total,
            "chi_fraction": chi_weight / total,
            "carrier_fraction": carrier_weight / total,
            "exterior_fraction": 1 - carrier_weight / total,
        })
    overlap = float(abs(np.dot(
        tangents["psi_clock"], tangents["chi_internal_clock"])) / max(
            np.linalg.norm(tangents["psi_clock"])
            * np.linalg.norm(tangents["chi_internal_clock"]), 1e-300))
    obstruction = np.column_stack((
        remainders["psi_clock"], remainders["chi_internal_clock"]))
    _, obstruction_singular, obstruction_vh = np.linalg.svd(
        obstruction, full_matrices=False)
    linked_direction = obstruction_vh[-1]
    if abs(linked_direction[0]) > 1e-15:
        linked_direction = linked_direction / linked_direction[0]
    source_matrix = np.column_stack((
        sources["psi_clock"], sources["chi_internal_clock"]))
    linked_source = source_matrix @ linked_direction
    linked_tangent = np.linalg.lstsq(
        jacobian, -linked_source, rcond=1e-10)[0]
    linked_remainder = jacobian @ linked_tangent + linked_source
    linked_relative = float(
        np.linalg.norm(linked_remainder)
        / max(np.linalg.norm(linked_source), 1e-300))
    independent_closed = all(
        row["relative_range_remainder"] < 1e-8 for row in rows)
    linked_closed = linked_relative < 1e-8
    np.savez_compressed(
        HERE / "c27_body_gyro_charge_tangent_solvability.npz",
        psi_clock=tangents["psi_clock"],
        chi_internal_clock=tangents["chi_internal_clock"],
        linked_clock_direction=linked_direction,
        linked_clock_tangent=linked_tangent,
        jacobian_singular_values=singular,
    )
    result = {
        "status": "V4_C27_BODY_GYRO_CHARGE_TANGENT_SOLVABILITY",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "side": spectrum.SIDE,
        "dt": spectrum.DT,
        "finite_difference_step": STEP,
        "jacobian_rank": rank,
        "jacobian_dimension": int(jacobian.shape[0]),
        "jacobian_nullity": int(jacobian.shape[1] - rank),
        "jacobian_condition_on_range": float(
            singular[0] / max(singular[rank - 1], 1e-300)),
        "rows": rows,
        "tangent_overlap": overlap,
        "full_field_family_tangents_independent": overlap < 1 - 1e-8,
        "obstruction_singular_values": [
            float(value) for value in obstruction_singular],
        "linked_clock_direction_dthetaPsi_dthetaChi": [
            float(value) for value in linked_direction],
        "linked_clock_relative_range_remainder": linked_relative,
        "independent_clock_families_close": independent_closed,
        "linked_clock_family_closes": linked_closed,
        "verdict": (
            "C27_BODY_GYRO_LINKED_CLOCK_FAMILY_TANGENT_PASS"
            if linked_closed else
            "C27_BODY_GYRO_INDEPENDENT_CLOCK_TANGENTS_PASS"
            if independent_closed else
            "C27_BODY_GYRO_FULL_FIELD_CHARGE_TANGENT_OBSTRUCTION"),
        "claim_boundary": (
            "First-order full-field Fredholm condition only. A passing tangent "
            "does not prove finite-amplitude continuation, nonlinear stability, "
            "or moving-carrier closure."),
    }
    output = HERE / "c27_body_gyro_charge_tangent_solvability.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "jacobian_rank": rank,
        "jacobian_dimension": int(jacobian.shape[0]),
        "tangent_overlap": overlap,
        "linked_clock_direction": [float(value) for value in linked_direction],
        "linked_clock_relative_range_remainder": linked_relative,
        "rows": rows,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
