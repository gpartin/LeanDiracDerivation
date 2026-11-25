"""Four-update free handoff with the transfer-activated chi bond."""

from __future__ import annotations

import argparse
import numpy as np
import torch

import edge_snap_action_rank as rank
import edge_snap_four_update_orbit as four
import edge_snap_transfer_four_update_orbit as transfer
import transfer_activated_chi_bond_probe as activated


COEFFICIENT = 0.5


def interval_action(left: torch.Tensor, right: torch.Tensor, dt: float,
                    coefficient: float) -> torch.Tensor:
    baseline = transfer.interval_action(left, right, dt, coefficient)
    return baseline + COEFFICIENT * activated.interval_extra_action(
        left, right, dt)


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
            output_stem="transfer_activated_chi_bond_four_update_orbit",
            action_variant=(
                "g19_transfer_plus_positive_density_rate_activated_chi_bond"),
            initial_middle=initial_middle)
    finally:
        four.orbit.interval_action = original
        four.CHARGE_WEIGHT = original_charge_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()
    main(args.iterations)
