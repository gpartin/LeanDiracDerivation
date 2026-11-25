"""Long Dirac-tangent gate for the cube-symplectic movement action."""

from __future__ import annotations

from pathlib import Path

import cube_symplectic_movement_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = (Path(__file__).resolve().parent
              / "cube_symplectic_movement_tangent_blocks.npz")


if __name__ == "__main__":
    gate.main()
