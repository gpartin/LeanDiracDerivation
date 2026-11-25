"""Run the long tangent gate with the variationally completed action."""

from __future__ import annotations

from pathlib import Path

import completed_mixed_prong_action as action
import mixed_prong_dirac_long_tangent as gate


gate.action = action
gate.CACHE = (Path(__file__).resolve().parent
              / "completed_mixed_prong_tangent_blocks.npz")


if __name__ == "__main__":
    gate.main()
