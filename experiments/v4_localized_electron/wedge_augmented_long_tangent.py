"""Long tangent gate for the jointly inferred area/material velocity."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import wedge_augmented_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"wedge_augmented_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
