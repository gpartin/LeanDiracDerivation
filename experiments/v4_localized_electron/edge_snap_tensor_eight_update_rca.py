"""Classify the timestep-refined tensor edge-snap residual."""

from edge_snap_four_update_rca import main


if __name__ == "__main__":
    main(
        input_stem="edge_snap_tensor_eight_update_orbit",
        output_stem="edge_snap_tensor_eight_update_rca",
    )
