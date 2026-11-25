"""Fixed-readout motion diagnostic for the cube-symplectic candidate."""

from __future__ import annotations

import os


os.environ.setdefault(
    "LFM_TANGENT_CACHE", "cube_symplectic_movement_tangent_blocks.npz")
os.environ.setdefault(
    "LFM_MOTION_OUTPUT", "cube_symplectic_movement_motion_diagnostic.json")

from wake_common_velocity_motion_diagnostic import main  # noqa: E402


if __name__ == "__main__":
    main()
