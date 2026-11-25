"""Long tangent gate for live parallel-overlap transport."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import overlap_material_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"overlap_material_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
