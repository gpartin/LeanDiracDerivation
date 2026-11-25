"""Long tangent gate for the signed complete-frame connection."""

from __future__ import annotations

from pathlib import Path

import complete_frame_connection_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = Path(__file__).resolve().parent/"complete_frame_connection_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
