"""Long tangent gate for the induced oriented-area connection."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import wedge_connection_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"wedge_connection_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
