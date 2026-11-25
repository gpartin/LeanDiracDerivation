"""First-order rest-orbit solvability of the rotor-wake kinetic candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.func import grad

import canonical_noncoplanar_bifurcation_audit as small
import canonical_noncoplanar_box_transfer as box
import canonical_noncoplanar_large_box_audit as large
import canonical_transfer_wake_kinetic_gate as wake


HERE = Path(__file__).resolve().parent
STRENGTH = 1e-4


def main() -> None:
    archive = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")
    rest = archive["rest"]
    theta = float(np.arccos(1-small.DT**2*small.CHI0**2/2))
    phase = np.exp(1j*theta)

    canonical_del = grad(
        lambda p, c, f: box.interval(p, c, large.SIDE)
        + box.interval(c, f, large.SIDE), argnums=1)

    def canonical_map(raw: torch.Tensor):
        return canonical_del(
            box.phase_rotate(raw, np.conjugate(phase), large.SIDE), raw,
            box.phase_rotate(raw, phase, large.SIDE))

    tensor = torch.tensor(rest, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        canonical_map, tensor, vectorize=True).detach().numpy()
    u, singular, vh = np.linalg.svd(jacobian, full_matrices=True)
    scale = max(float(singular[0]), 1e-300)
    active = singular > 1e-8*scale
    left_null = u[:, ~active]
    right_range = vh[active].T
    left_range = u[:, active]
    range_values = singular[active]

    rows = []
    for mode in ("transfer_even", "transfer_odd", "area_gradient"):
        added_del = grad(
            lambda p, c, f: wake.extra_interval(p, c, mode)
            + wake.extra_interval(c, f, mode), argnums=1)

        def added_map(raw: torch.Tensor):
            return added_del(
                box.phase_rotate(raw, np.conjugate(phase), large.SIDE), raw,
                box.phase_rotate(raw, phase, large.SIDE))

        force = added_map(tensor).detach().numpy()
        null_coordinates = left_null.T@force
        range_coordinates = left_range.T@force
        correction = -right_range@(range_coordinates/range_values)
        candidate = rest+STRENGTH*correction

        candidate_del = grad(
            lambda p, c, f: box.interval(p, c, large.SIDE)
            + box.interval(c, f, large.SIDE)
            + STRENGTH*(wake.extra_interval(p, c, mode)
                        + wake.extra_interval(c, f, mode)), argnums=1)

        def candidate_map(raw: torch.Tensor):
            previous = box.phase_rotate(
                raw, np.conjugate(phase), large.SIDE)
            following = box.phase_rotate(raw, phase, large.SIDE)
            return candidate_del(previous, raw, following)

        candidate_tensor = torch.tensor(
            candidate, dtype=torch.float64, requires_grad=True)
        residual = candidate_map(candidate_tensor).detach().numpy()
        psi0, _chi0 = large.unpack_np(rest)
        psi, chi = large.unpack_np(candidate)
        core = np.sum(abs(psi0)**2, axis=-1) > 1e-12
        rows.append({
            "mode": mode,
            "added_force_norm": float(np.linalg.norm(force)),
            "null_solvability_norm": float(np.linalg.norm(null_coordinates)),
            "relative_null_obstruction": float(
                np.linalg.norm(null_coordinates)
                / max(np.linalg.norm(force), 1e-300)),
            "unit_first_order_correction_over_rest": float(
                np.linalg.norm(correction)/np.linalg.norm(rest)),
            "screen_strength": STRENGTH,
            "corrected_state_displacement_over_rest": float(
                np.linalg.norm(candidate-rest)/np.linalg.norm(rest)),
            "corrected_orbit_residual_rms": float(
                np.linalg.norm(residual)/np.sqrt(residual.size)),
            "residual_over_strength_squared": float(
                np.linalg.norm(residual)/np.sqrt(residual.size)/STRENGTH**2),
            "halo_psi_power_fraction": float(
                np.sum(abs(psi[~core])**2)/max(np.sum(abs(psi)**2), 1e-300)),
            "halo_chi_deviation_norm": float(np.linalg.norm(
                chi[~core]-np.asarray((0, 0, small.CHI0)))),
        })

    viable = [row for row in rows[:2]
              if row["relative_null_obstruction"] < 1e-8
              and row["corrected_orbit_residual_rms"] < 1e-4]
    result = {
        "status": "V4_CANONICAL_TRANSFER_WAKE_FREDHOLM",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "canonical_DEL_jacobian_nullity": int(np.sum(~active)),
        "rows": rows,
        "verdict": (
            "TRANSFER_WAKE_HAS_FIRST_ORDER_REST_BRANCH"
            if viable else "TRANSFER_WAKE_REST_BRANCH_OBSTRUCTED"),
        "claim_boundary": (
            "Fredholm and one-step range correction at strength 1e-4. A pass "
            "is local existence evidence only, not a converged nonlinear "
            "branch, selected coefficient, stable action, or motion pass."),
    }
    output = HERE / "canonical_transfer_wake_fredholm.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
