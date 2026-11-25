"""Long tangent gate for the oriented-area material completion."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import wedge_material_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"wedge_material_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
