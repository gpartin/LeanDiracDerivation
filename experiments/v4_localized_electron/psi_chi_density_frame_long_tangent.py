"""Long tangent gate for density-activated live-frame completion."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import psi_chi_density_frame_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"psi_chi_density_frame_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
