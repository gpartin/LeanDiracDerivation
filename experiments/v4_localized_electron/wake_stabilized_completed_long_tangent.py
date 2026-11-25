"""Long tangent gate for direct-prong transport plus wake control."""

from __future__ import annotations

import os
from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import wake_stabilized_completed_action as action


gate.action = action
loop = os.environ.get("LFM_LOOP_CURVATURE", "1").replace(".", "p")
relative = os.environ.get("LFM_RELATIVE_PRONG_WEIGHT", "0.75").replace(".", "p")
gate.CACHE = (Path(__file__).resolve().parent
              / f"wake_stabilized_completed_loop_{loop}_relative_{relative}_tangent_blocks.npz")


if __name__ == "__main__":
    gate.main()
