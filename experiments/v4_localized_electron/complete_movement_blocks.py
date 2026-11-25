"""Assemble exact tangent blocks for ``complete_movement_action``."""

from pathlib import Path
import os

import numpy as np


HERE = Path(__file__).resolve().parent
COHERENT_SCALE = float(os.environ.get("LFM_COHERENT_MOVEMENT_SCALE", "1"))


def main() -> None:
    incumbent = np.load(
        HERE/"wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
    coherent = np.load(HERE/"curl_overlap_movement_blocks.npz")
    symplectic = np.load(
        HERE/"cube_symplectic_movement_complete_tangent_blocks.npz")
    blocks = {
        name: (incumbent[name]
               + COHERENT_SCALE*(coherent[name]-incumbent[name])
               + (symplectic[name]-incumbent[name]))
        for name in ("previous", "current", "following")
    }
    np.savez_compressed(HERE/"complete_movement_tangent_blocks.npz", **blocks)


if __name__ == "__main__":
    main()
