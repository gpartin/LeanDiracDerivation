"""Long tangent gate for the C19 triangle movement action."""

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import triangle_movement_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent / "triangle_movement_blocks.npz"


if __name__ == "__main__":
    gate.main()
