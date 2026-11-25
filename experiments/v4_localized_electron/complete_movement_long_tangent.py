"""Long tangent gate for the complete movement action."""

from pathlib import Path

import complete_movement_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = Path(__file__).resolve().parent / "complete_movement_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
