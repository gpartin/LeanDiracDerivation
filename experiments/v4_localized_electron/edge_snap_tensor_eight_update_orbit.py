"""Eight-update refinement of the tensor edge-snap relative orbit."""

from __future__ import annotations

import numpy as np

import edge_snap_four_update_orbit as orbit_search
import edge_snap_tensor_four_update_orbit as tensor_search


def refined_initial_path() -> np.ndarray | None:
    checkpoint = orbit_search.HERE / "edge_snap_tensor_four_update_orbit.npz"
    if not checkpoint.exists():
        return None
    data = np.load(checkpoint)
    coarse = [data["q0"], *list(data["middle"]), data["qn"]]
    fine = []
    for interval in range(4):
        fine.append((coarse[interval] + coarse[interval + 1]) / 2)
        if interval < 3:
            fine.append(coarse[interval + 1])
    return np.stack(fine)


def main() -> None:
    original_action = orbit_search.orbit.interval_action
    original_updates = orbit_search.UPDATES
    original_dt = orbit_search.DT
    orbit_search.orbit.interval_action = tensor_search.tensor_interval_action
    orbit_search.UPDATES = 8
    orbit_search.DT = 0.05
    try:
        orbit_search.main(
            iterations=350,
            output_stem="edge_snap_tensor_eight_update_orbit",
            action_variant="local_plus_traceless_density_mold",
            initial_middle=refined_initial_path())
    finally:
        orbit_search.orbit.interval_action = original_action
        orbit_search.UPDATES = original_updates
        orbit_search.DT = original_dt


if __name__ == "__main__":
    main()
