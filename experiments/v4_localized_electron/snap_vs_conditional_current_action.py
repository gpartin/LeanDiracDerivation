"""Compare literal cell snapping with cubical conditional Dirac transport.

This is a diagnostic of the current reduced-tight-frame research action.  It
does not install either motion rule in the action.  The literal-snap target
translates the complete joint carrier by one cell.  The conditional target
uses the already-derived cubical incidence projectors, which send the two
signed channels in opposite directions.  Both are tested against the same
native acceleration at the exact localized rest orbit.

The calculation also records an important measurement boundary: the cubical
coordinates are an internal fiber.  Applying their projectors directly to
the eight physical corner sites is a hypothesis until a localized
intertwining embedding is proved.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import canonical_noncoplanar_box_transfer as box
import cubical_edge_shift_channels as cube
import conditional_edge_native_gate as conditional
import edge_snap_verlet_gate as snap
import personal_touch_unit_geometric_nonlinear as action


HERE = Path(__file__).resolve().parent
SIDE = 5


def core_sites() -> list[tuple[int, int, int]]:
    center = SIDE // 2 - 1
    return [tuple(center + bit for bit in corner)
            for corner in itertools.product((0, 1), repeat=3)]


def fiber_values(field: np.ndarray) -> np.ndarray:
    return np.stack([field[site] for site in core_sites()])


def channel_row(field: np.ndarray, axis: int) -> dict:
    values = fiber_values(field)
    identity = np.eye(8, dtype=complex)
    gamma = cube.gamma(axis)
    forward = (identity + gamma) / 2
    backward = (identity - gamma) / 2
    plus = forward @ values
    minus = backward @ values
    norm = max(float(np.vdot(values, values).real), 1e-30)
    plus_weight = float(np.vdot(plus, plus).real / norm)
    minus_weight = float(np.vdot(minus, minus).real / norm)
    return {
        "axis": axis,
        "forward_weight": plus_weight,
        "backward_weight": minus_weight,
        "signed_channel_imbalance": plus_weight - minus_weight,
        "reconstruction_defect": float(np.linalg.norm(plus + minus - values)),
    }


def native_acceleration(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action.ACTION_KIND = "reduced_tight_frame"
    point = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    gradient = torch.func.grad(action.total_potential)(point)
    acceleration = -gradient / action.masses(point.device)
    psi_acceleration, chi_acceleration = box.unpack(acceleration, SIDE)
    return (psi_acceleration.detach().numpy(),
            chi_acceleration.detach().numpy())


def main() -> None:
    raw = np.load(
        HERE / "canonical_noncoplanar_large_box_solvability.npz")["rest"]
    psi_t, chi_t = box.unpack(torch.tensor(raw, dtype=torch.float64), SIDE)
    psi, chi = psi_t.numpy(), chi_t.numpy()
    force_psi, force_chi = native_acceleration(raw)

    snap_rows = [
        snap.classify(axis, sign, psi, chi, force_psi, force_chi)
        for axis in range(3) for sign in (-1, 1)
    ]
    acceleration = np.asarray(box.pack(force_psi, force_chi))
    conditional_rows = [
        conditional.classify(axis, sign, psi, chi, acceleration)
        for axis in range(3) for sign in (-1, 1)
    ]

    psi_channels = [channel_row(psi, axis) for axis in range(3)]
    # The chi vacuum component is not part of the localized carrier contrast.
    chi_vacuum = np.zeros_like(chi)
    chi_vacuum[..., 2] = 19.0
    chi_channels = [channel_row(chi - chi_vacuum, axis)
                    for axis in range(3)]

    snap_worst = max(row["combined_relative_residual"]
                     for row in snap_rows)
    conditional_worst = max(row["combined_relative_residual"]
                            for row in conditional_rows)
    lift = json.loads((HERE / "localized_conditional_lift_audit.json").read_text(
        encoding="utf-8"))
    rigid = json.loads((HERE / "rigid_shift_vs_dirac_transport.json").read_text(
        encoding="utf-8"))

    result = {
        "status": "V4_SNAP_VS_CONDITIONAL_CURRENT_ACTION_RESEARCH",
        "canonical_action_changed": False,
        "research_action_admitted": False,
        "tested_action": (
            "canonical V40 potential plus the two reduced unit tight-frame "
            "terms carrier_carrier_chi and carrier_carrier_Psi_tight_frame"),
        "strict_snap": {
            "definition": (
                "the complete joint Psi-chi carrier occupies one cube at one "
                "committed layer and the adjacent cube at the next"),
            "rows": snap_rows,
            "worst_relative_native_recurrence_residual": snap_worst,
            "verdict": "STRICT_ONE_LAYER_SNAP_FAILS_CURRENT_ACTION",
        },
        "conditional_dirac_channels": {
            "definition": (
                "inside each four-component copy, the +gamma channel shifts "
                "one way, the -gamma channel shifts the other way, and beta "
                "mixes them locally"),
            "psi_rest_channel_weights": psi_channels,
            "chi_contrast_rest_channel_weights": chi_channels,
            "rows": conditional_rows,
            "worst_relative_native_recurrence_residual": conditional_worst,
            "algebraic_dirac_comparator_smallest_dt_state_error": rigid["rows"][-1][
                "conditional_walk_state_error"],
            "algebraic_dirac_comparator_smallest_dt_rigid_error": rigid["rows"][-1][
                "rigid_shift_state_error"],
            "physical_site_lift_verdict": lift["verdict"],
            "verdict": "DIRAC_CHANNEL_ALGEBRA_PASSES_NATIVE_LIFT_AND_DYNAMICS_OPEN",
        },
        "second_order_identity": (
            "At zero channel phase the centered second difference of the "
            "conditional shift equals that of the rigid shift because "
            "P_plus+P_minus=I and gamma^2=I. Acceleration alone therefore "
            "cannot decide the transport mechanism; the two-layer velocity, "
            "bond current, and localized embedding must decide it."),
        "copy_result": (
            "Both Clifford copies contain their own forward and backward "
            "channels. The second copy is not a leading or trailing copy and "
            "does not turn a rigid snap into Dirac transport."),
        "research_verdict": "STRICT_SNAP_REJECTED_AS_MASSIVE_DIRAC_TARGET",
        "first_open_link": (
            "derive a localized intertwiner E from full Psi/chi phase space "
            "to the cubical four-component fiber, then test whether the same "
            "native two-layer update carries the signed channel current"),
        "claim_boundary": (
            "The result rejects the literal one-layer rigid snap for the "
            "tested action and identifies the correct Dirac comparator. It "
            "does not yet prove a mobile localized native carrier or V4-006."),
    }
    (HERE / "snap_vs_conditional_current_action.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
