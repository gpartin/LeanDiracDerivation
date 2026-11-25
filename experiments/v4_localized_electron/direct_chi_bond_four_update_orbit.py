"""Four-update handoff with the geometry-normalized direct chi bond trial.

The action delta is the positive reciprocal, cubic C19 energy

  Delta U = (B/6) * sum_(i,n) w_n |chi_(i+n)-chi_i|^2 / (2*24).

The factor 1/6 is the quadratic C19 continuum normalization.  This script
keeps the current g=19 nonlinear transfer kinetic metric and the previously
optimized endpoint carrier.  Consequently it is a movement-response gate,
not an exact-rest or action-admission result.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

import density_mold_connection_probe as local
import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import edge_snap_transfer_four_update_orbit as transfer


def direct_chi_bond_potential(raw: torch.Tensor) -> torch.Tensor:
    _, chi = local.unpack_tensor(raw, four.orbit.SIDE)
    bond = torch.zeros((), dtype=torch.float64)
    for offset, weight in rank.native.DIRECTIONS:
        difference = chi - torch.roll(
            chi, shifts=offset, dims=(0, 1, 2))
        bond = bond + weight * torch.sum(local.squared_norm(difference))
    return (rank.B / 6) * bond / (2 * rank.C19_WEIGHT_SUM)


def interval_action(left: torch.Tensor, right: torch.Tensor, dt: float,
                    coefficient: float) -> torch.Tensor:
    baseline = transfer.interval_action(left, right, dt, coefficient)
    delta = dt * (
        direct_chi_bond_potential(left)
        + direct_chi_bond_potential(right)) / 2
    return baseline - delta


def main(iterations: int = 600) -> None:
    rank.lock_inertia()
    transfer.GEOMETRIC_FACTOR = 19.0
    transfer.FILTER_FACTOR = 0.0
    transfer.COLLECTIVE_FACTOR = 0.0
    transfer.RADIAL_FACTOR = 0.0
    transfer.NORMAL_FACTOR = -1.0
    transfer.BOND_FRAME_FACTOR = 0.0
    original = four.orbit.interval_action
    original_charge_weight = four.CHARGE_WEIGHT
    four.orbit.interval_action = interval_action
    four.CHARGE_WEIGHT = 1e-4
    try:
        checkpoint = four.HERE / "edge_snap_transfer_charge_four_update_orbit.npz"
        initial_middle = np.load(checkpoint)["middle"]
        four.main(
            iterations=iterations,
            output_stem="direct_chi_bond_four_update_orbit",
            action_variant=(
                "g19_transfer_plus_positive_direct_chi_C19_B_over_6"),
            initial_middle=initial_middle)
    finally:
        four.orbit.interval_action = original
        four.CHARGE_WEIGHT = original_charge_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    main(args.iterations)
