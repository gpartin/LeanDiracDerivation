"""Rescale the exact triangle delta without rebuilding its Hessian."""

from pathlib import Path
import os

import numpy as np


HERE = Path(__file__).resolve().parent
SCALE = float(os.environ.get("LFM_TRIANGLE_MOVEMENT_SCALE", str(1/19)))


def main():
    incumbent = np.load(
        HERE/"wake_stabilized_completed_loop_1_relative_0_tangent_blocks.npz")
    full = np.load(HERE/"triangle_movement_blocks_full.npz")
    blocks = {name: incumbent[name]+SCALE*(full[name]-incumbent[name])
              for name in ("previous", "current", "following")}
    np.savez_compressed(HERE/"triangle_movement_blocks.npz", **blocks)


if __name__ == "__main__":
    main()
