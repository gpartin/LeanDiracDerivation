"""Long tangent response under an existing-area coefficient scan."""

from __future__ import annotations

import os
from pathlib import Path

import area_scaled_completed_action as action
import mixed_prong_dirac_long_tangent as gate


scale = os.environ.get("LFM_AREA_SCALE", "1").replace(".", "p")
gate.action = action
gate.CACHE = (Path(__file__).resolve().parent
              / f"area_scaled_completed_{scale}_tangent_blocks.npz")


if __name__ == "__main__":
    gate.main()
