"""Long tangent gate for the curl--overlap movement completion."""

from pathlib import Path

import curl_overlap_movement_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = Path(__file__).resolve().parent / "curl_overlap_movement_blocks.npz"


if __name__ == "__main__":
    gate.main()
