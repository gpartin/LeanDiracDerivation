"""Run the shared long-tangent gate with the cumulative stable action."""

from __future__ import annotations

from pathlib import Path

import mixed_prong_dirac_long_tangent as gate
import stabilized_mixed_prong_action as action


gate.action = action
gate.CACHE = Path(__file__).resolve().parent / "stabilized_mixed_prong_tangent_blocks.npz"


if __name__ == "__main__":
    gate.main()
