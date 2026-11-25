"""Long tangent gate for complete live-frame C19 bonds."""

from __future__ import annotations

from pathlib import Path

import complete_frame_bond_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"complete_frame_bond_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
